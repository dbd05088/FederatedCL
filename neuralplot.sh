#/bin/bash

# CIL CONFIG
NOTE="fedavg_demon_lr5e-6_cosine_nodist" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="fedavg"
MODEL_ARCH="bunny_3b" # llava bunny_3b bunny_8b

# fed args
SCENARIO=5
NUM_ROUNDS=10
NUM_CLIENTS=10
MODEL_MAX_LEN=6000
MAX_NEW_TOKENS=256

# adam8bit_bnb adamw_torch

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

CUDA_VISIBLE_DEVICES=5 python neuralflow.py \
    --is_eval True \
    --model_name_or_path $MODEL_NAME \
    --model_name_for_dataarg $MODEL_NAME \
    --model_type $MODEL_TYPE \
    --version $VERSION \
    --scenario $SCENARIO \
    --num_rounds $NUM_ROUNDS \
    --num_clients $NUM_CLIENTS \
    --model_max_length $MODEL_MAX_LEN \
    --max_new_tokens $MAX_NEW_TOKENS \
    --vision_tower $VISION_TOWER \
    --bits $BITS \
    --bf16 True \
    --tf32 True \
    --note $NOTE \
    --mode $MODE \
    --round_to_eval 1 \
    --output_dir "./nohup" #> ./nohup/fedavg_demon_nodist_lr5e-6_bs128_bs4_round10_eval.log 2>&1 &

# --eval_period $EVAL_PERIOD
