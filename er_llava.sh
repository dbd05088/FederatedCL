#/bin/bash

# CIL CONFIG
NOTE="fedavg" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="fedavg"
SIGMA=10
REPEAT=1
INIT_CLS=100
# GPU_TRANSFORM="--gpu_transform"
# USE_AMP="--use_amp"
SEEDS="1"

# if [ "$DATASET" == "cifar10" ]; then
MEM_SIZE=50000 ONLINE_ITER=1
MODEL_NAME="resnet18" EVAL_PERIOD=200
BATCHSIZE=4; LR=2e-5 OPT_NAME="adamw_torch" SCHED_NAME="cosine" IMP_UPDATE_PERIOD=1


for RND_SEED in $SEEDS
do
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_llava.py \
    --gradient_checkpointing True \
    --gradient_accumulation_steps 4 \
    --bf16 True \
    --tf32 True \
    --mode $MODE --dataloader_num_workers 2 \
    --sigma $SIGMA --repeat $REPEAT --init_cls $INIT_CLS \
    --memory_size $MEM_SIZE \
    --seed $RND_SEED \
    --optim $OPT_NAME --lr_scheduler_type $SCHED_NAME \
    --learning_rate $LR --per_gpu_train_batch_size $BATCHSIZE \
    --per_device_eval_batch_size $BATCHSIZE \
    --online_iter $ONLINE_ITER \
    --note $NOTE --eval_period $EVAL_PERIOD \
    --output_dir "./nohup" > ./nohup/fedavg_bs16.log 2>&1 &
done

    # --deepspeed ./deepspeed_script/zero3.json \
# --sigma $SIGMA --repeat $REPEAT --init_cls $INIT_CLS \
# --model_name $MODEL_NAME
# --imp_update_period $IMP_UPDATE_PERIOD $USE_AMP \
# $GPU_TRANSFORM
# --memory_size $MEM_SIZE

    