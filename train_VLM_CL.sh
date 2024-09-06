#!/bin/bash
# CIL CONFIG
NOTE="ours_generator_512_8_8_bs4_saveoptim_l2distill0.1_fishernormmerge_lr3e-4_sc22_4tasks_5rounds_itr125" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="ours_generator"
MODEL_ARCH="llava" # llava bunny_3b bunny_8b
RND_SEED=1

# fed args
SCENARIO=22
NUM_ROUNDS=5
NUM_TASKS=4
NUM_CLIENTS=4
MODEL_MAX_LEN=20000
NUM_ITER=125
MEMORY_SIZE=100000
IS_STREAMONLY=True

LORA_ENABLE=False
IA3_ENABLE=True

USE_TASK_ID=False
USE_PROMPT=False

SAVE_OPTIM=True

USE_TASK_VECTOR=False
USE_FISHER=True

GENERATOR_OUTPUT_SIZE=512
GENERATOR_HIDDEN_DIM=8
GENERATOR_HIDDEN_FEATURE=8
KEY_EMBED_SIZE=64
POOL_SIZE=12
PROMPT_TOP_K=3
EMA_RATIO=0.9

BATCHSIZE=4

LR=3e-4
MM_PROJECTOR_LR=$LR #3e-4
FINAL_LR=$LR #3e-4
MM_FINAL_LR=$LR #3e-4
OPT_NAME="adamw_torch" # adam8bit_bnb adamw_torch
SCHED_NAME="constant" #cosine
WARMUP_RATIO=0.1 # SHOULD BE 0.03 / NUM_ROUNDS
DECAY_RATIO=0.9

if [ "$MODEL_ARCH" == "llava" ]; then
    MODEL_NAME="./llava-v1.5-7b"
    VERSION="v1"
    VISION_TOWER="./clip-vit-large-patch14-336"
    MODEL_TYPE="llama"
    BITS=16

elif [ "$MODEL_ARCH" == "bunny_3b" ]; then
    MODEL_NAME="BAAI/Bunny-v1_0-3B"
    VERSION="bunny"
    VISION_TOWER="google/siglip-so400m-patch14-384"
    MODEL_TYPE="phi-2"
    BITS=16
elif [ "$MODEL_ARCH" == "bunny_8b" ]; then
    MODEL_NAME="BAAI/Bunny-Llama-3-8B-V"
    VERSION="llama"
    VISION_TOWER="google/siglip-so400m-patch14-384"
    MODEL_TYPE="llama3-8b"
    BITS=8
else
    echo "Undefined setting"
    exit 1
fi
# --master_port 29500
# --num_gpus=4
deepspeed --master_port 29504 \
    --include localhost:4 \
    train_VLM_CL.py \
    --deepspeed ./deepspeed_script/zero2.json \
    --model_name_or_path $MODEL_NAME \
    --model_name_for_dataarg $MODEL_NAME \
    --model_type $MODEL_TYPE \
    --version $VERSION \
    --num_clients $NUM_CLIENTS \
    --model_max_length $MODEL_MAX_LEN \
    --num_rounds $NUM_ROUNDS \
    --num_tasks $NUM_TASKS \
    --scenario $SCENARIO \
    --vision_tower $VISION_TOWER \
    --gradient_checkpointing True \
    --num_train_epochs 1 \
    --num_iter $NUM_ITER \
    --gradient_accumulation_steps 1 \
    --bits $BITS \
    --bf16 True \
    --tf32 True \
    --mode $MODE --dataloader_num_workers 2 \
    --seed $RND_SEED \
    --optim $OPT_NAME \
    --lr_scheduler_type $SCHED_NAME \
    --weight_decay 0. \
    --warmup_ratio $WARMUP_RATIO \
    --decay_ratio $DECAY_RATIO \
    --learning_rate $LR --per_gpu_train_batch_size $BATCHSIZE \
    --final_lr $FINAL_LR --mm_final_lr $MM_FINAL_LR \
    --mm_projector_lr $MM_PROJECTOR_LR \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --logging_steps 2 \
    --note $NOTE \
    --memory_size $MEMORY_SIZE \
    --is_streamonly $IS_STREAMONLY \
    --prompt_num 1 \
    --lora_enable $LORA_ENABLE \
    --ia3_enable $IA3_ENABLE \
    --use_task_id $USE_TASK_ID \
    --get_prompt $USE_PROMPT \
    --generator_output_size $GENERATOR_OUTPUT_SIZE \
    --generator_hidden_dim $GENERATOR_HIDDEN_DIM \
    --generator_hidden_feature $GENERATOR_HIDDEN_FEATURE \
    --ema_ratio $EMA_RATIO \
    --key_embed_size $KEY_EMBED_SIZE \
    --pool_size $POOL_SIZE \
    --prompt_top_k $PROMPT_TOP_K \
    --save_optim $SAVE_OPTIM \
    --use_task_vector $USE_TASK_VECTOR \
    --use_fisher $USE_FISHER \
    --output_dir "./results/test/" > ./nohup/${NOTE}.log 2>&1 &

# --eval_period $EVAL_PERIOD
# lr_scheduler_type