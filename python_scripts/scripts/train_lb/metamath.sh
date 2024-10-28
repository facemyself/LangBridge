#!/bin/env bash
#export TRANSFORMERS_CACHE=/data1/rzw/CACHE/huggingface/hub
export HUGGINGFACE_HUB_CACHE=/data1/rzw/CACHE/huggingface/hub
export CUDA_LAUNCH_BLOCKING=1
#export CUDA_VISIBLE_DEVICES=7
NUM_GPU=4
#google/mt5-xl 
#facebook/xglm-1.7B
# ARGS="
# --n_gpu $NUM_GPU
# --strategy deepspeed_stage_2
# --output_dir checkpoints/MetaMath-qwen2.5-stage1-10k
# --run_name MetaMath-qwen2.5-stage1-10k
# --seed 42
# --train_set_path /data1/rzw/CODE/LangBridge/data/metamath-200k
# --output_exists True
# --enc_name_or_path Qwen/Qwen2.5-1.5B-Instruct
# --lm_name_or_path meta-math/MetaMath-7B-V1.0
# --alignments linear
# --enc_hidden_size 1536
# --lm_hidden_size 4096
# --max_length 128
# --max_length_enc 1024
# --freeze_language_model True
# --freeze_encoder True
# --learning_rate_alignment 6e-4
# --learning_rate_enc 2e-5
# --w_decay_alignment 0.0
# --w_decay_enc 0.1
# --warmup_steps 0
# --per_device_train_batch_size 4
# --per_device_eval_batch_size 16
# --gradient_accumulation_steps 8
# --logging_steps 10
# --num_train_epochs 1
# --dataloader_num_workers 16
# --bf16 True
# --use_wandb True
# --enc_output_index 27
# --lm_input_index 0
# --lm_output_index 31
# --dec_input_index 0
# --training_stage 1
# "

# echo $ARGS
# if [ $NUM_GPU == 1 ]; then
#     echo "running on a single GPU"
#     python train_langbridge.py $ARGS
# else
#     echo "running on multiple GPUs"
#     torchrun --nproc_per_node $NUM_GPU train_langbridge.py $ARGS
# fi


ARGS="
--n_gpu $NUM_GPU
--strategy deepspeed_stage_2
--output_dir checkpoints/MetaMath-qwen2.5-stage2
--run_name MetaMath-qwen2.5-stage2-10k
--seed 42
--output_exists True
--enc_name_or_path Qwen/Qwen2.5-1.5B-Instruct
--lm_name_or_path meta-math/MetaMath-7B-V1.0
--hf_checkpoint_path checkpoints/MetaMath-qwen2.5-stage1-10k/epoch=1
--alignments linear
--enc_hidden_size 1536
--lm_hidden_size 4096
--max_length 128
--max_length_enc 1024
--freeze_language_model True
--freeze_encoder True
--learning_rate_alignment 6e-4
--learning_rate_enc 2e-5
--w_decay_alignment 0.0
--w_decay_enc 0.1
--warmup_steps 0
--per_device_train_batch_size 4
--per_device_eval_batch_size 16
--gradient_accumulation_steps 8
--logging_steps 10
--num_train_epochs 1
--dataloader_num_workers 16
--bf16 True
--use_wandb False
--enc_output_index 27
--lm_input_index 0
--lm_output_index 31
--dec_input_index 0
--training_stage 2
"

echo $ARGS
if [ $NUM_GPU == 1 ]; then
    echo "running on a single GPU"
    python train_langbridge.py $ARGS
else
    echo "running on multiple GPUs"
    torchrun --nproc_per_node $NUM_GPU train_langbridge.py $ARGS
fi