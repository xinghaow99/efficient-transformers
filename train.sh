export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_PROJECT=efficient-transformers
accelerate launch --num_processes 4 --main_process_port 29502 train.py \
    --per_device_train_batch_size 10 \
    --num_accumulation_steps 48 \
    --learning_rate 6e-4 \
    --report_to wandb \
    --model_type performer