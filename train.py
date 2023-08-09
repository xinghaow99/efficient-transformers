import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    DefaultDataCollator,
    set_seed,
    TrainingArguments,
    TrainerCallback,
)
from transformers.trainer_pt_utils import get_model_param_count
from datasets import load_dataset
import evaluate
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import argparse
import torch
import copy
from torch.nn.utils.rnn import pad_sequence
from retnet.modeling_retnet import RetNetModelWithLMHead
from retnet.configuration_retnet import RetNetConfig
from linformer.configuration_linformer import LinformerConfig
from linformer.modeling_linformer import LinformerForCausalLM
from performer.modeling_performerllama import PerformerLlamaForCausalLM
from performer.configuration_performerllama import PerformerLlamaConfig
import math
import yaml
@dataclass
class ModelArguments:
    model_type: Optional[str] = field(
        default="retnet",
        metadata={"help": "Model type, e.g. retnet, linformer, gpt2, etc."}
    )
    model_name_or_path: Optional[str] = field(
        default="EleutherAI/pythia-12b"
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables using Huggingface auth token from Git Credentials."}
    )

@dataclass
class DataArguments:
    eval_dataset_size: int = field(
        default=1024, metadata={"help": "Size of validation dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_length: Optional[int] = field(
        default=2048,
        metadata={"help": "Maximum input length for the source text."}
    )
    dataset: str = field(
        default='alpaca',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )
    dataset_format: Optional[str] = field(
        default=None,
        metadata={"help": "Which dataset format is used. [alpaca|chip2|self-instruct|hh-rlhf]"}
    )

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(
        default=None
    )
    max_memory_MB: int = field(
        default=80000,
        metadata={"help": "Free memory per gpu."}
    )
    report_to: str = field(
        default='none',
        metadata={"help": "To use wandb or something else for reporting."}
    )
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=2, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=8, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    max_steps: int = field(default=10000, metadata={"help": 'How many optimizer update steps to take'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=False, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='no', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=40, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})

metric = evaluate.load("perplexity")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits[..., :-1, :].reshape(-1, logits.shape[-1])
    labels = labels[..., 1:].reshape(-1)
    return metric.compute(predictions=predictions, references=labels)
class LogPerplexityCallback(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % args.logging_steps == 0 and state.global_step > 0:
            # Get the loss and calculate perplexity
            train_loss = state.log_history[-1]['loss']
            perplexity = math.exp(train_loss)
            state.log_history[-1]['perplexity'] = perplexity
class MyTrainer(Trainer):
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss
            loss = tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged)
            logs["loss"] = round(loss, 4)
            logs["perplexity"] = round(math.exp(loss), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

def load_config_from_yaml(config_file, config):
    with open(config_file, 'r') as f:
        config = config.from_dict(yaml.load(f, Loader=yaml.FullLoader))
    return config

if __name__ == "__main__":
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments
    ))
    model_args, data_args, training_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    set_seed(args.seed)
    # config = AutoConfig.from_pretrained("EleutherAI/pythia-160m")
    # model = AutoModelForCausalLM.from_config(config)
    if args.model_type == 'retnet':
        config = load_config_from_yaml(f"retnet-100m.yaml", RetNetConfig)
        model = RetNetModelWithLMHead(config)
    elif args.model_type == 'linformer':
        config = load_config_from_yaml(f"linformer-100m.yaml", LinformerConfig)
        model = LinformerForCausalLM(config)
    elif args.model_type == 'performer':
        config = load_config_from_yaml('performer-100m.yaml', PerformerLlamaConfig)
        model = PerformerLlamaForCausalLM(config)
    model.config.use_cache = False # `use_cache=True` is incompatible with gradient checkpointing.
    # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.model_max_length = 16384
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.unk_token = tokenizer.eos_token
    tokenizer.bos_token = tokenizer.eos_token
    dataset = load_dataset('EleutherAI/pile', streaming=True)
    def tokenize_function(examples):
        inputs = tokenizer(examples["text"], padding='max_length', truncation=True, max_length=args.max_length)
        examples["input_ids"] = inputs["input_ids"]
        examples["attention_mask"] = inputs["attention_mask"]
        examples["labels"] = copy.deepcopy(inputs["input_ids"])
        return examples
    dataset = dataset.map(tokenize_function, batched=False, remove_columns=dataset["train"].column_names)
    print('Model Size: ', get_model_param_count(model))
    train_datset = dataset["train"]
    val_dataset = dataset["validation"]
    if args.max_train_samples is not None:
        train_datset = train_datset.select(range(args.max_train_samples))
    if args.max_eval_samples is not None:
        val_dataset = val_dataset.select(range(args.max_eval_samples))
    trainer = MyTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_datset,
        eval_dataset=val_dataset,
        args=training_args,
        compute_metrics=compute_metrics,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )
    trainer.train()