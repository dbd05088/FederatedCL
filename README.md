# environment setting
- Make sure your nvidia-smi and nvcc version all over cuda 11.6
```
conda create -n fcl python=3.10
conda activate fcl
pip install transformers==4.40.0
pip install torch==2.2.1 torchvision xformers==0.0.25 --index-url https://download.pytorch.org/whl/cu118
pip install flash-attn==2.5.7 --no-build-isolation
pip install peft bitsandbytes pandas kornia opencv-python timm torch_optimizer easydict pycocoevalcap sentencepiece protobuf trl==0.8.6 deepspeed==0.14.0 loguru captum POT jsonlines
pip install -U scikit-learn
```

```
conda install -c conda-forge cudatoolkit-dev -y
sudo apt install openjdk-11-jdk or conda install conda-forge::openjdk=8
sudo apt-get install build-essential
gdrive files download 14W8eYNFpCkJvQN8zyNSmTTxboT3le1eC
```

# Dataset Preparation
<details>
<summary>Click to expand</summary>

1. In `dataset` folder, run the following script files:
```bash
bash MMCloze.sh
bash HRVQA.sh
bash MultiVQA_large.sh
bash MultiVQA_small.sh
bash mPLUG.sh
bash Bongard.sh
bash KGQA.sh
bash Visual_Relation.sh
bash Visual_storytelling.sh
```

2. Run the following preprocessing python codes:
- MMCloze
```bash
python ./preprocess/preprocess_RecipeQA_TextCloze.py
python ./preprocess/preprocess_RecipeQA_VisualCloze.py
python ./preprocess/preprocess_COMICS_Panel.py
python ./preprocess/preprocess_COMICS_Dialogue.py
```
- HRVQA
```bash
cd ./dataset/HRVQA/jsons
python task_split.py
cd ../../..
python ./preprocess/preprocess_HRVQA.py
```
- MultiVQA_large
```bash
python ./preprocess/preprocess_RecipeQA_ImageCoherence.py
python ./preprocess/preprocess_Fashion200K.py
python ./preprocess/preprocess_NLVR2.py
```
- MultiVQA_small
```bash
python ./preprocess/preprocess_VISION.py
python ./preprocess/preprocess_VizWiz.py
python ./preprocess/preprocess_MIT.py
```
- mPLUG
```bash
python ./preprocess/preprocess_mPLUG.py
```

- Bongard
```bash
python ./preprocess/preprocess_Bongard.py
python ./preprocess/preprocess_Bongard_query.py
python ./preprocess/preprocess_Bongard_HOI.py
python ./preprocess/preprocess_Bongard_HOI_query.py
```

- KGQA
```bash
python ./preprocess/preprocess_WebQA.py
python ./preprocess/preprocess_TQA.py
python ./preprocess/preprocess_AQUA.py
```

- Visual_Relation
```bash
python ./preprocess/preprocess_SpotDiff.py
python ./preprocess/preprocess_Bird2Words.py
python ./preprocess/preprocess_CLEVR.py
python ./preprocess/preprocess_IEdit.py
```

- Visual_storytelling
```bash
python ./preprocess/preprocess_PororoSV.py
python ./preprocess/preprocess_FlintstonesSV.py
python ./preprocess/preprocess_VIST.py
python ./preprocess/preprocess_AESOP.py
```
</details>

-----------------------------------------------------------------
## New data instruction

1. In `dataset` folder, run the following script files:
```bash
bash Fashion.sh
bash HRVQA.sh
bash Pair_TF.sh
bash KGQA.sh
bash Bongard.sh
bash iconqa.sh
bash CoInstruct.sh
bash Visual_Relation.sh
bash Visual_storytelling.sh
bash MultiVQA_small.sh
```

2. Run the following preprocessing python codes:
- Fashion
```bash
python ./preprocess/preprocess_Fashion200K.py
```
- HRVQA
```bash
cd ./dataset/HRVQA/jsons
python task_split.py
cd ../../..
python ./preprocess/preprocess_HRVQA.py
```
- Pair_TF
```bash
python ./preprocess/preprocess_NLVR2.py
python ./preprocess/preprocess_PatternCom.py
```

- KGQA
```bash
python ./preprocess/preprocess_WebQA.py
python ./preprocess/preprocess_TQA.py
python ./preprocess/preprocess_AQUA.py
```

- Bongard
```bash
python ./preprocess/preprocess_Bongard2.py
python ./preprocess/preprocess_Bongard_query.py
python ./preprocess/preprocess_Bongard_HOI.py
python ./preprocess/preprocess_Bongard_HOI_query.py
```

- IconQA
```bash
python ./preprocess/preprocess_iconqa.py
```

- CoInstruct
```bash
python ./preprocess/preprocess_coinstruct.py
```

- Visual_Relation
```bash
python ./preprocess/preprocess_SpotDiff.py
python ./preprocess/preprocess_Bird2Words.py
python ./preprocess/preprocess_CLEVR.py
python ./preprocess/preprocess_IEdit.py
```

- Visual_storytelling
```bash
python ./preprocess/preprocess_PororoSV.py
python ./preprocess/preprocess_FlintstonesSV.py
python ./preprocess/preprocess_VIST.py
python ./preprocess/preprocess_AESOP.py
```

- MultiVQA_small
```bash
python ./preprocess/preprocess_VISION.py
python ./preprocess/preprocess_VizWiz.py
python ./preprocess/preprocess_MIT.py
```

# How to run

1. Train
- Client for-loop continual-learning: `bash train_VLM_CL.sh`
    - Arguments:
        - NOTE: Experiment name **All the models are saved in client_states_$NOTE folder**
        - MODE: Name of method (method in `federated_methods/method_manager.py`)
        - NUM_ITER: MAX number of updates per round in each client
        - NUM_TASKS: Number of tasks that clients learn sequentially
        - NUM_ROUNDS: Number of rounds per task
        - SCENARIO: Data configuration to use (`scenarios/scenario-{$SCENARIO}.json`)
        - PROMPT_NUM: prompt length (for l2p, dap, pfedpg)
        - LORA_ENABLE: False for prompt-based methods (l2p, dap, pfedpg)
        - IS_STREAMONLY: choose between stream-only or memory-only
        - MEMORY_SIZE: buffer size per client

2. Eval
    1. `bash eval.sh`
        - Arguments:
            - NOTE: experiment name **Models in client_states_$NOTE folder will be loaded**
            - SCENARIO
            - ROUND_TO_EVAL: model weights ends with round{round_to_eval}.pth / round{round_to_eval-1}.pth for client/server will be loaded

    2. `bash eval_gpt.sh`: gpt eval
        - Arguments:
            - base_dir: directory of generated jsons exist
            - round: round to eval
            - OPENAI_API_KEY: openai api key

    3. `python combined_score.py`
        - Arguments:
            - mode: method name
            - Method: note of method
            - num_rounds: round to eval
        - The results will be saved in `results.csv`
        
# Baseline list

- L2P (layerwise, visual feature only)
- CODA_Prompt                           
- DAP                                   
- EvoPrompt                             
- L2P_T (concat visual + text feature) 
- CODA_Prompt_T                         
- DAP_T                                 
- EvoPrompt_T                           
- LAE                                   
- Federated methods combined version
    - FedAvg, FedPer, Ditto, Feddat, ...
- L2P_FedAvg / L2P_T_FedAvg
- CODA_Prompt_FedAvg / CODA_Prompt_T_FedAvg
- DAP_FedAvg / DAP_T_FedAvg
- 
