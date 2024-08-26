# environment setting
- Make sure your nvidia-smi and nvcc version all over cuda 11.6
```
conda create -n fcl python=3.10
conda activate fcl
pip install transformers==4.40.0
pip install torch==2.2.1 torchvision xformers==0.0.25 --index-url https://download.pytorch.org/whl/cu118
pip install flash-attn==2.5.7 --no-build-isolation
pip install peft bitsandbytes pandas kornia opencv-python timm torch_optimizer easydict pycocoevalcap sentencepiece protobuf trl==0.8.6 deepspeed==0.14.0 loguru captum POT
```

```
conda install -c conda-forge cudatoolkit-dev -y
sudo apt install openjdk-11-jdk or conda install conda-forge::openjdk=8
sudo apt-get install build-essential
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
python preprocess_RecipeQA_TextCloze.py
python preprocess_RecipeQA_VisualCloze.py
python preprocess_COMICS_Panel.py
python preprocess_COMICS_Dialogue.py
```
- HRVQA
```bash
cd ./dataset/HRVQA/jsons
python task_split.py
cd ../../..
python preprocess_HRVQA.py
```
- MultiVQA_large
```bash
python preprocess_RecipeQA_ImageCoherence.py
python preprocess_Fashion200K.py
python preprocess_NLVR2.py
```
- MultiVQA_small
```bash
python preprocess_VISION.py
python preprocess_VizWiz.py
python preprocess_MIT.py
```
- mPLUG
```bash
python preprocess_mPLUG.py
```

- Bongard
```bash
python preprocess_Bongard.py
python preprocess_Bongard_query.py
python preprocess_Bongard_HOI.py
python preprocess_Bongard_HOI_query.py
```

- KGQA
```bash
python preprocess_WebQA.py
python preprocess_TQA.py
python preprocess_AQUA.py
```

- Visual_Relation
```bash
python preprocess_SpotDiff.py
python preprocess_Bird2Words.py
python preprocess_CLEVR.py
python preprocess_IEdit.py
```

- Visual_storytelling
```bash
python preprocess_PororoSV.py
python preprocess_FlintstonesSV.py
python preprocess_VIST.py
python preprocess_AESOP.py
```
</details>

# How to run

1. Train
- Client for-loop running: `bash train_VLM.sh`
    - Arguments:
        - NOTE: Experiment name **All the models are saved in client_states_$NOTE folder**
        - MODE: Name of method (method in `federated_methods/method_manager.py`)
        - NUM_ITER: Number of updates per round in each client

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

- L2P (layerwise, visual feature only) -> done
- CODA_Prompt                           -> hyperparameter searching
- DAP                                   -> done
- EvoPrompt                             -> done
- L2P_T (concat visual + text feature) -> done
- CODA_Prompt_T                         -> hyperparameter searching
- DAP_T                                 -> done
- EvoPrompt_T                           -> done
- LAE                                   -> done
- Federated methods combined version
    - FedAvg, Ditto, Feddat, ...
- L2P_FedAvg / L2P_T_FedAvg
- CODA_Prompt_FedAvg / CODA_Prompt_T_FedAvg
- DAP_FedAvg / DAP_T_FedAvg
- 
