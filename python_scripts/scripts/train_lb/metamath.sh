#!/bin/env bash
#export OMP_NUM_THREADS=8
export HUGGINGFACE_HUB_CACHE=/workspace/CACHE


export CUDA_VISIBLE_DEVICES=0,1
NUM_GPU=2
BATCH_SIZE_PER_GPU=4
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPU/$BATCH_SIZE_PER_GPU))
ARGS="
--n_gpu $NUM_GPU
--strategy deepspeed_stage_2
--output_dir checkpoints/metamath-lb-9b
--run_name metamath-lb-9b
--seed 42
--train_set_path /data1/rzw/CODE/LangBridge/data/metamath-200k
--output_exists True
--enc_name_or_path google/mt5-xl
--lm_name_or_path meta-math/MetaMath-7B-V1.0
--alignments linear
--enc_hidden_size 2048
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
"

echo $ARGS
if [ $NUM_GPU == 1 ]; then
    echo "running on a single GPU"
    python train_langbridge.py $ARGS
else
    echo "running on multiple GPUs"
    torchrun --nproc_per_node $NUM_GPU train_langbridge.py $ARGS
fi