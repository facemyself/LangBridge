#!/bin/env bash
#export TRANSFORMERS_CACHE=/data1/rzw/CACHE/huggingface/hub
#export HUGGINGFACE_HUB_CACHE=/data1/rzw/CACHE/huggingface/hub
export HUGGINGFACE_HUB_CACHE=/workspace/CACHE
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1
random_port(){
    # Random port
    MASTER_PORT=$((30000 + RANDOM % (99999-30000+1)))
    echo "MASTER_PORT=$MASTER_PORT"
}

check_4090() {
    # Check if the GPU is RTX 40 series
    if nvidia-smi | grep -q 'RTX 40'; then
        echo "RTX 40 series GPU detected, disabling NCCL P2P and IB"
        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1
    fi
}


export_world_info() {
    # Set world info for deepspeed
    # ref: https://github.com/microsoft/DeepSpeed/issues/1331
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        echo "CUDA_VISIBLE_DEVICES is not set"
        NUM_GPUS=$(nvidia-smi -L | wc -l)
        # generate GPUS from index 0
        CUDA_VISIBLE_DEVICES=$(seq -s ',' 0 $((NUM_GPUS - 1)))
        echo "Use all GPUs"
        export "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        WID=`echo  {\"localhost\": [$CUDA_VISIBLE_DEVICES]} | base64`
    else
        # count CUDA_VISIBLE_DEVICES
        NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
        echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        WID=`echo  {\"localhost\": [$CUDA_VISIBLE_DEVICES]} | base64`
    fi
}
random_port
check_4090
export_world_info
BATCH_SIZE_PER_GPU=2
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
# google/mt5-xl 
# facebook/xglm-1.7B
ARGS="
--n_gpu $NUM_GPUS
--strategy deepspeed_stage_2
--output_dir checkpoints/metamath-qwen2.5-stage1-6-6-25-21
--run_name metamath-qwen2.5-stage1-6-6-25-21
--seed 42
--train_set_path /data1/rzw/CODE/LangBridge/data/metamath-200k
--output_exists True
--enc_name_or_path Qwen/Qwen2.5-1.5B-Instruct
--lm_name_or_path meta-math/MetaMath-7B-V1.0
--alignments ffn
--enc_hidden_size 1536
--lm_hidden_size 4096
--max_length 128
--max_length_enc 1024
--freeze_language_model True
--freeze_encoder True
--learning_rate_alignment 4e-5
--learning_rate_enc 2e-5
--w_decay_alignment 0.0
--w_decay_enc 0.1
--warmup_steps 0
--per_device_train_batch_size $BATCH_SIZE_PER_GPU
--per_device_eval_batch_size $BATCH_SIZE_PER_GPU
--gradient_accumulation_steps $GRADIENT_ACC_STEPS
--logging_steps 10
--num_train_epochs 1
--dataloader_num_workers 16
--bf16 True
--use_wandb False
--enc_output_index 6
--lm_input_index 6
--lm_output_index 25
--dec_input_index 21
--training_stage 1
"

echo $ARGS
if [ $NUM_GPUS == 1 ]; then
    echo "running on a single GPU"
    python train_langbridge.py $ARGS
else
    echo "running on multiple GPUs"
    torchrun --nproc_per_node $NUM_GPUS train_langbridge.py $ARGS
fi


random_port
check_4090
export_world_info
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

ARGS="
--n_gpu $NUM_GPUS
--strategy deepspeed_stage_2
--output_dir checkpoints/metamath-qwen2.5-stage2-6-6-25-21
--run_name metamath-qwen2.5-stage2-6-6-25-21
--seed 42
--output_exists True
--enc_name_or_path Qwen/Qwen2.5-1.5B-Instruct
--lm_name_or_path meta-math/MetaMath-7B-V1.0
--hf_checkpoint_path checkpoints/metamath-qwen2.5-stage1-6-6-25-21/epoch=1
--alignments ffn
--enc_hidden_size 1536
--lm_hidden_size 4096
--max_length 128
--max_length_enc 1024
--freeze_language_model True
--freeze_encoder True
--learning_rate_alignment 4e-5
--learning_rate_enc 2e-5
--w_decay_alignment 0.0
--w_decay_enc 0.1
--warmup_steps 0
--per_device_train_batch_size $BATCH_SIZE_PER_GPU
--per_device_eval_batch_size $BATCH_SIZE_PER_GPU
--gradient_accumulation_steps $GRADIENT_ACC_STEPS
--logging_steps 10
--num_train_epochs 1
--dataloader_num_workers 16
--bf16 True
--use_wandb True
--enc_output_index 6
--lm_input_index 6
--lm_output_index 25
--dec_input_index 21
--training_stage 2
"

echo $ARGS
if [ $NUM_GPUS == 1 ]; then
    echo "running on a single GPU"
    python train_langbridge.py $ARGS
else
    echo "running on multiple GPUs"
    torchrun --nproc_per_node $NUM_GPUS train_langbridge.py $ARGS
fi


random_port
check_4090
export_world_info
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

ARGS="
--n_gpu $NUM_GPUS
--strategy deepspeed_stage_2
--output_dir checkpoints/metamath-qwen2.5-stage3-6-6-25-21
--run_name metamath-qwen2.5-stage3-6-6-25-21
--seed 42
--output_exists True
--enc_name_or_path Qwen/Qwen2.5-1.5B-Instruct
--lm_name_or_path meta-math/MetaMath-7B-V1.0
--hf_checkpoint_path checkpoints/metamath-qwen2.5-stage2-6-6-25-21/epoch=1
--alignments ffn
--enc_hidden_size 1536
--lm_hidden_size 4096
--max_length 128
--max_length_enc 1024
--freeze_language_model True
--freeze_encoder True
--learning_rate_alignment 4e-5
--learning_rate_enc 2e-5
--w_decay_alignment 0.0
--w_decay_enc 0.1
--warmup_steps 0
--per_device_train_batch_size $BATCH_SIZE_PER_GPU
--per_device_eval_batch_size $BATCH_SIZE_PER_GPU
--gradient_accumulation_steps $GRADIENT_ACC_STEPS
--logging_steps 10
--num_train_epochs 1
--dataloader_num_workers 16
--bf16 True
--use_wandb True
--enc_output_index 6
--lm_input_index 6
--lm_output_index 25
--dec_input_index 21
--training_stage 3
"

echo $ARGS
if [ $NUM_GPUS == 1 ]; then
    echo "running on a single GPU"
    python train_langbridge.py $ARGS
else
    echo "running on multiple GPUs"
    torchrun --nproc_per_node $NUM_GPUS train_langbridge.py $ARGS
fi
