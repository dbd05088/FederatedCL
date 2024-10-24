#!/bin/bash

# CIL CONFIG
NOTE="fedours_bs4_saveoptim_lr6e-3_lastdownmean_freq5_fishercossimsoftmax_mean_sc0_4tasks_5rounds_fixitr100"
MODE="fedours"
MODEL_ARCH="llava" # llava bunny_3b bunny_8b

# fed args
SCENARIO=1
NUM_ROUNDS=5
NUM_TASKS=7
NUM_CLIENTS=1
MODEL_MAX_LEN=20000
MAX_NEW_TOKENS=512


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

# ROUND_TO_EVALS=$2
ROUND_TO_EVALS=(20)
ITER_TO_EVAL=0

for ((index=0; index<${#ROUND_TO_EVALS[@]}; index++)); do
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=$1 python eval_VLM_CL.py \
        --is_eval True \
        --model_name_or_path $MODEL_NAME \
        --model_name_for_dataarg $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --version $VERSION \
        --scenario $SCENARIO \
        --num_rounds $NUM_ROUNDS \
        --num_tasks $NUM_TASKS \
        --num_clients $NUM_CLIENTS \
        --model_max_length $MODEL_MAX_LEN \
        --max_new_tokens $MAX_NEW_TOKENS \
        --vision_tower $VISION_TOWER \
        --bits $BITS \
        --bf16 True \
        --tf32 True \
        --note $NOTE \
        --mode $MODE \
        --eval_server True \
        --unseen_task True \
        --zeroshot False \
        --lora_enable False \
        --ia3_enable True \
        --generator_output_size 512 \
        --generator_hidden_dim 8 \
        --generator_hidden_feature 8 \
        --key_embed_size 64 \
        --prompt_top_k 1 \
        --pool_size 40 \
        --set_state "gate" \
        --is_prompt False \
        --use_task_vector False \
        --round_to_eval ${ROUND_TO_EVALS[$index]} \
        --eval_iter $ITER_TO_EVAL \
        --output_dir "./nohup" #> ./nohup/${NOTE}_eval_round${ROUND_TO_EVALS[$index]}_iter${ITER_TO_EVAL}.log 2>&1 &
done
# --eval_period $EVAL_PERIOD