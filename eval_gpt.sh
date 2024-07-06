#!/bin/bash

base_dir="fedper_8/fedper_8_llava_sc12_lr1e-4_bs16_itr100_constant_round10"

# Associate each dataset name with its corresponding integer value
declare -A dataset_map
dataset_map["Bongard-OpenWorld-0"]=0
dataset_map["Birds-to-Words-0"]=1
dataset_map["DiDeMoSV-0"]=3
dataset_map["Mementos-0"]=5
dataset_map["SciCap-1"]=6
dataset_map["AQUA-1"]=9

round=10

# Iterate over the keys (dataset names) in the associative array
for dataset in "${!dataset_map[@]}"; do
    i=${dataset_map[$dataset]}
    input_file1="./eval_results/${base_dir}/client${i}_round${round}_${dataset}.json"
    input_file2="./eval_results/${base_dir}/server_round${round}_${dataset}.json"
    output_file1="./eval_results_gpt/${base_dir}_correctness/client${i}_round${round}_${dataset}.jsonl"
    output_file2="./eval_results_gpt/${base_dir}_correctness/server_round${round}_${dataset}.jsonl"
    
    echo "Processing ${dataset} (client ${i})..."
    python eval_gpt_explainfirst.py -r "$input_file1" -o "$output_file1"
    # python eval_gpt_explainfirst.py -r "$input_file2" -o "$output_file2"
done

echo "All datasets processed."