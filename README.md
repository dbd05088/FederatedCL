# environment setting
- Make sure your nvidia-smi and nvcc version all over cuda 11.6
```
conda create -n fcl python=3.10
conda activate fcl
pip install transformers==4.40.0
pip install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu118
pip install packaging
pip install flash-attn --no-build-isolation
pip install peft
pip install bitsandbytes
pip install pandas kornia opencv-python timm torch_optimizer easydict pycocoevalcap sentencepiece protobuf
pip install trl==0.8.6
pip install deepspeed==0.14.0
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

2. Eval
- `bash eval.sh`
    - Arguments:
        - NOTE: experiment name **Models in client_states_$NOTE folder will be loaded**
        - SCENARIO
        - ROUND_TO_EVAL: model weights ends with round{round_to_eval}.pth / round{round_to_eval-1}.pth for client/server will be loaded

- `bash eval_gpt.sh`: gpt eval