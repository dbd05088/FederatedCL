#!/bin/bash

base_dir="pfedpg/PT_nomem_sc20_lr1e-2_4tasks_5rounds_itr125"

# Associate each integer value with a list of corresponding dataset names
declare -A dataset_map
# List of datasets that are not multi-choice
dataset_map[3]="WebQA-0 AQUA-0 AQUA-1"
dataset_map[4]="Bongard-OpenWorld-0"
dataset_map[6]="mPLUG-0 mPLUG-1 mPLUG-2 mPLUG-4" #
dataset_map[7]="Spot-the-Diff-0 Birds-to-Words-0 IEdit-0 CLEVR-Change-0"
dataset_map[8]="PororoSV-0 FlintstonesSV-0 VIST-0 AESOP-0"

round=20

# Iterate over the keys (integer values) in the associative array
for i in "${!dataset_map[@]}"; do
    datasets=(${dataset_map[$i]})
    for dataset in "${datasets[@]}"; do
        input_file1="./eval_results/${base_dir}/client${i}_round${round}_${dataset}.json"
        input_file2="./eval_results/${base_dir}/server_round${round}_${dataset}.json"
        output_file1="./eval_results_gpt/${base_dir}_correctness/client${i}_round${round}_${dataset}.jsonl"
        output_file2="./eval_results_gpt/${base_dir}_correctness/server_round${round}_${dataset}.jsonl"
        
        echo "Processing ${dataset} (client ${i})..."
        OPENAI_API_KEY="" python eval_gpt_explainfirst.py -r "$input_file1" -o "$output_file1"
        # python eval_gpt_explainfirst.py -r "$input_file2" -o "$output_file2"
    done
done

echo "All datasets processed."