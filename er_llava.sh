#/bin/bash

# CIL CONFIG
NOTE="er_test" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="er"
DATASET="cifar10" # cifar10, cifar100, tinyimagenet, imagenet
SIGMA=10
REPEAT=1
INIT_CLS=100
# GPU_TRANSFORM="--gpu_transform"
# USE_AMP="--use_amp"
SEEDS="1"

if [ "$DATASET" == "cifar10" ]; then
    MEM_SIZE=50000 ONLINE_ITER=1
    MODEL_NAME="resnet18" EVAL_PERIOD=100
    BATCHSIZE=16; LR=2e-5 OPT_NAME="adamw_torch" SCHED_NAME="cosine" IMP_UPDATE_PERIOD=1

# elif [ "$DATASET" == "cifar100" ]; then
#     MEM_SIZE=2000 ONLINE_ITER=3
#     MODEL_NAME="resnet18" EVAL_PERIOD=100
#     BATCHSIZE=16; LR=3e-4 OPT_NAME="adamw_torch" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

# elif [ "$DATASET" == "tinyimagenet" ]; then
#     MEM_SIZE=4000 ONLINE_ITER=3
#     MODEL_NAME="resnet18" EVAL_PERIOD=100
#     BATCHSIZE=32; LR=3e-4 OPT_NAME="adamw_torch" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

# elif [ "$DATASET" == "imagenet" ]; then
#     MEM_SIZE=1281167 ONLINE_ITER=0.03125
#     MODEL_NAME="resnet18" EVAL_PERIOD=1000
#     BATCHSIZE=1024; LR=3e-4 OPT_NAME="adamw_torch" SCHED_NAME="default" IMP_UPDATE_PERIOD=10

else
    echo "Undefined setting"
    exit 1
fi

for RND_SEED in $SEEDS
do
    CUDA_VISIBLE_DEVICES=1 deepspeed main_llava.py \
    --deepspeed ./deepspeed_script/zero3.json \
    --mode $MODE --dataloader_num_workers 8 \
    --dataset $DATASET \
    --seed $RND_SEED \
    --optim $OPT_NAME --lr_scheduler_type $SCHED_NAME \
    --learning_rate $LR --per_gpu_train_batch_size $BATCHSIZE \
    --per_device_eval_batch_size $BATCHSIZE \
    --online_iter $ONLINE_ITER \
    --note $NOTE --eval_period $EVAL_PERIOD \
    --output_dir "./nohup.out"
done

# --sigma $SIGMA --repeat $REPEAT --init_cls $INIT_CLS \
# --model_name $MODEL_NAME
# --imp_update_period $IMP_UPDATE_PERIOD $USE_AMP \
# $GPU_TRANSFORM
# --memory_size $MEM_SIZE