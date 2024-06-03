#/bin/bash
# CIL CONFIG
NOTE="fedavg_demon_lr1e-6_test" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="fedavg"
MODEL_ARCH="llava" # llava bunny_3b bunny_8b
RND_SEED=1

# fed args
SCENARIO=4
NUM_ROUNDS=10
NUM_CLIENTS=10
MODEL_MAX_LEN=6000

BATCHSIZE=8

LR=2e-4
MM_PROJECTOR_LR=2e-5
OPT_NAME="adamw_torch" # adam8bit_bnb adamw_torch
SCHED_NAME="cosine" #cosine
WARMUP_RATIO=0.003 # SHOULD BE 0.03 / NUM_ROUNDS

if [ "$MODEL_ARCH" == "llava" ]; then
    MODEL_NAME="liuhaotian/llava-v1.5-7b"
    VERSION="v1"
    VISION_TOWER="openai/clip-vit-large-patch14-336"
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

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed train_VLM.py \
    --deepspeed ./deepspeed_script/zero2.json \
    --model_name_or_path $MODEL_NAME \
    --model_name_for_dataarg $MODEL_NAME \
    --model_type $MODEL_TYPE \
    --version $VERSION \
    --num_clients $NUM_CLIENTS \
    --model_max_length $MODEL_MAX_LEN \
    --num_rounds $NUM_ROUNDS \
    --scenario $SCENARIO \
    --vision_tower $VISION_TOWER \
    --gradient_checkpointing True \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 2 \
    --bits $BITS \
    --bf16 True \
    --tf32 True \
    --mode $MODE --dataloader_num_workers 4 \
    --seed $RND_SEED \
    --optim $OPT_NAME --lr_scheduler_type $SCHED_NAME \
    --weight_decay 0. \
    --warmup_ratio $WARMUP_RATIO \
    --learning_rate $LR --per_gpu_train_batch_size $BATCHSIZE \
    --mm_projector_lr $MM_PROJECTOR_LR \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --logging_steps 1 \
    --note $NOTE \
    --output_dir "./results/test/" > ./nohup/fedavg_demon_lr1e-6_test.log 2>&1 &

# --eval_period $EVAL_PERIOD