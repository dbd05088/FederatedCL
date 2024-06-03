#/bin/bash
# sysctl -w vm.max_map_count=262144
sudo sysctl -w vm.max_map_count=262144
# CIL CONFIG
NOTE="fedavg_demon8_lr1e-6_iter2" # experiment name *****All the models are saved in client_states_$NOTE folder*******
MODE="fedavg" # method name
MODEL_ARCH="bunny_3b" # llava bunny_3b bunny_8b
RND_SEED=1

# fed args
SCENARIO=6 # run scenario-$SCENARIO.json from scenarios folder
NUM_ROUNDS=10
NUM_CLIENTS=8 # should be the same as the number of clients in scenario-$SCENARIO.json
MODEL_MAX_LEN=6000

MEM_SIZE=50000
ONLINE_ITER=2
BATCHSIZE=4
TEMP_BATCHSIZE=1

LR=1e-6
MM_PROJECTOR_LR=1e-6
OPT_NAME="adamw_torch" # adam8bit_bnb adamw_torch
SCHED_NAME="cosine_with_restarts" #cosine
WARMUP_RATIO=0.03 # SHOULD BE 0.03 / NUM_ROUNDS

if [ "$MODEL_ARCH" == "llava" ]; then
    MODEL_NAME="liuhaotian/llava-v1.5-7b"
    VERSION="v1"
    VISION_TOWER="openai/clip-vit-large-patch14-336"
    MODEL_TYPE="llama"
    BITS=8

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

CUDA_VISIBLE_DEVICES=3,4,5,6,7 python main_VLM.py \
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
    --gradient_accumulation_steps 4 \
    --bits $BITS \
    --bf16 True \
    --tf32 True \
    --mode $MODE --dataloader_num_workers 2 \
    --memory_size $MEM_SIZE \
    --seed $RND_SEED \
    --optim $OPT_NAME --lr_scheduler_type $SCHED_NAME \
    --weight_decay 0. \
    --warmup_ratio $WARMUP_RATIO \
    --learning_rate $LR --per_gpu_train_batch_size $BATCHSIZE \
    --mm_projector_lr $MM_PROJECTOR_LR \
    --temp_batchsize $TEMP_BATCHSIZE \
    --online_iter $ONLINE_ITER \
    --note $NOTE \
    --output_dir "./nohup" > ./nohup/fedavg_demon8_lr1e-6_iter2.log 2>&1 &

# --eval_period $EVAL_PERIOD