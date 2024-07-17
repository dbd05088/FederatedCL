#/bin/bash
# CIL CONFIG
NOTE="sft_nomem_sc20_lr5e-5_4tasks_5rounds_itr125" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="sft"
MODEL_ARCH="llava" # llava bunny_3b bunny_8b
RND_SEED=1

# fed args
SCENARIO=20
NUM_ROUNDS=5
NUM_TASKS=4
NUM_CLIENTS=10
MODEL_MAX_LEN=15000
NUM_ITER=125

BATCHSIZE=4

LR=5e-5
MM_PROJECTOR_LR=5e-5
FINAL_LR=5e-5
MM_FINAL_LR=5e-5
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
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --master_port 29500 \
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
    --lora_enable True \
    --output_dir "./results/test/" > ./nohup/sft_nomem_sc20_lr5e-5_4tasks_5rounds_itr125.log 2>&1 &

# --eval_period $EVAL_PERIOD
# lr_scheduler_type