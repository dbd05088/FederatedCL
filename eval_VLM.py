import logging.config
import os
import random

import numpy as np
import torch
from configuration.VLM_config import ModelArguments, DataArguments, TrainingArguments
import transformers
from utils.train_utils import get_VLMmodel

from utils.method_manager_VLM import select_method
# from torch.utils.tensorboard import SummaryWriter

from torch import multiprocessing
import copy
import torch.distributed as dist
import json
from transformers import BitsAndBytesConfig

from utils.data_loader_VLM import GenerationDataset, GenerationDataset2
from torch.utils.data import DataLoader
from utils.eval_metrics import NLPEvaluator, matching_token_num
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

from transformers import StoppingCriteria, StoppingCriteriaList

class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, repeat_len = 2):
      self.n = repeat_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        should_stop =False
        if input_ids.shape[1] > self.n*3:
            last_n_ids = input_ids[0][-self.n:]		# 마지막으로 생성한 n개의 토큰
            lastlast_n_ids = input_ids[0][-self.n*2:-self.n]
            lastlastlast_n_ids = input_ids[0][-self.n*2:-self.n]
            for i in range(self.n):
                if lastlastlast_n_ids[i] != lastlast_n_ids[i] or lastlast_n_ids[i] != last_n_ids[i]: # stop sequence와 비교
                    should_stop = False
                    break
                else :
                    should_stop = True
        return should_stop

def evaluate(dataset, dataname, round, model, tokenizer, device, model_args, training_args, logger, client_id=None):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    stopping_criteria = CustomStoppingCriteria()
    stopping_criteria = StoppingCriteriaList([stopping_criteria])
    # img_feat_size = 729
    model.eval()
    predictions = []
    n_word_total = 0
    n_generated_word_total = 0
    n_word_correct = 0
    cnt = 0
    with torch.no_grad():
        for i, (inputs, imgs, gold, prompt, img_file) in enumerate(tqdm(dataloader)):
            inputs = inputs.to(device=device, non_blocking=True)
            imgs = imgs.to(device=device, dtype=torch.bfloat16, non_blocking=True)
            image_sizes = [x.shape[-2:] for x in imgs]
            
            with torch.inference_mode():
                output_ids = model.generate(
                    inputs,
                    images=imgs,
                    # image_sizes=image_sizes,
                    do_sample=True,# if args.temperature > 0 else False,
                    temperature=0.2,#args.temperature,
                    top_p=None,#args.top_p,
                    num_beams=1,#args.num_beams,
                    max_new_tokens=model_args.max_new_tokens,#args.max_new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria = stopping_criteria
                )
            if 'bunny' in model_args.model_name_or_path.lower():
                input_token_len = inputs.shape[1]
                output_ids = output_ids[:,input_token_len:]
            
            pred_sentence = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            
            input_label = tokenizer.encode(gold[0])
            output_id = tokenizer.encode(pred_sentence)
            n_word = len(set(input_label))
            n_generated_word = len(set(output_id))
            n_correct = matching_token_num(output_id, input_label)
            # print(pred_sentence)
            predictions.append({"image_file":img_file[0], "input":prompt[0], "sentence":pred_sentence, "gt_sentence":gold[0].strip()})
            
            n_word_total += n_word
            n_generated_word_total += n_generated_word
            n_word_correct += n_correct
            cnt += 1
    scores = NLPEvaluator(predictions).evaluate()
    scores["precision"] = n_word_correct / n_word_total
    scores["recall"] = n_word_correct / n_generated_word_total
    
    predictions.append(scores)
    #save predictions
    if client_id is not None:
        logger.info(f"Test (Client id {client_id}) | Data {dataname} | precision {scores['precision']:.4f} | recall {scores['recall']:.4f} | Bleu_1 {scores['Bleu_1']} | Bleu_2 {scores['Bleu_2']} | Bleu_3 {scores['Bleu_3']} |Bleu_4 {scores['Bleu_4']} | METEOR {scores['METEOR']} | ROUGE_L {scores['ROUGE_L']} | CIDEr {scores['CIDEr']} |")
        with open(f"./eval_results/{training_args.mode}/{training_args.note}/client{client_id}_round{round}_{dataname}.json", 'w') as fp:
            json.dump(predictions, fp, indent=4)
    else:
        logger.info(f"Test (Server) | Data {dataname} | precision {scores['precision']:.4f} | recall {scores['recall']:.4f} | Bleu_1 {scores['Bleu_1']} | Bleu_2 {scores['Bleu_2']} | Bleu_3 {scores['Bleu_3']} |Bleu_4 {scores['Bleu_4']} | METEOR {scores['METEOR']} | ROUGE_L {scores['ROUGE_L']} | CIDEr {scores['CIDEr']} |")
        with open(f"./eval_results/{training_args.mode}/{training_args.note}/server_round{round}_{dataname}.json", 'w') as fp:
            json.dump(predictions, fp, indent=4)

    

def main():
    ##################################
    round_to_eval = 10
    ##################################
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    logging.config.fileConfig("./configuration/logging.conf")
    logger = logging.getLogger()

    os.makedirs(f"eval_results/{training_args.mode}/{training_args.note}", exist_ok=True)
    fileHandler = logging.FileHandler(f'eval_results/{training_args.mode}/{training_args.note}/round_{round_to_eval}.log', mode="w")

    # writer = SummaryWriter(f'tensorboard/{training_args.mode}/{training_args.note}/federated')

    formatter = logging.Formatter(
        "[%(levelname)s] %(filename)s:%(lineno)d > %(message)s"
    )
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.info(training_args)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    logger.info(f"Set the device ({device})")

    # Fix the random seeds
    torch.manual_seed(training_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(training_args.seed)
    random.seed(training_args.seed)

    model, tokenizer, data_args = get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args)
    
    train_datalists, test_datalists = get_datalists(training_args, training_args.scenario)
    samples_per_round_per_client = [len(train_datalists[i]) // training_args.num_rounds for i in range(training_args.num_clients)]
    
    
    logger.info(f'Evaluatiing clients and server at round {round_to_eval}')
    
    server_eval_key = []
    server_state_dict = torch.load(f'./client_states_{training_args.note}/server_model_round{round_to_eval-1}.pth', map_location='cpu')
    for client_id in range(training_args.num_clients):
        # load client weight
        client_state_dict = torch.load(f'./client_states_{training_args.note}/{client_id}_client_model_round{round_to_eval}.pth', map_location='cpu')
        test_datalist = test_datalists[client_id]
        for data_info in test_datalist:
            if samples_per_round_per_client[client_id]*round_to_eval > data_info['eval_cnt']:
                # breakpoint()
                model.load_state_dict(client_state_dict, strict=False)
                dataset = GenerationDataset2(data_info['data'], tokenizer, data_args)
                # evaluate(data_info['data'], data_info['data_name'], round_to_eval, model, tokenizer, data_args, device, model_args, training_args, logger, client_id)
                evaluate(dataset, data_info['data_name'], round_to_eval, model, tokenizer, device, model_args, training_args, logger, client_id)
                if data_info['data_name'] not in server_eval_key:
                    model.load_state_dict(server_state_dict, strict=False)
                    # evaluate(data_info['data'], data_info['data_name'], round_to_eval, model, tokenizer, data_args, device, model_args, training_args, logger, None)
                    evaluate(dataset, data_info['data_name'], round_to_eval, model, tokenizer, device, model_args, training_args, logger, None)
                    server_eval_key.append(data_info['data_name'])
    
def get_datalists(args, scenario_num):
    with open(f"./scenarios/scenario-{scenario_num}.json") as fp:
        scenario = json.load(fp)
    assert args.num_clients == len(scenario)
    
    train_datalists = {}
    test_datalists = {}
    
    for client_data in scenario:
        client_id = client_data['client_id']
        train_datalist = []
        test_datalist = []
        eval_cnt = 0
        for data in client_data['datasets']:
            with open(f"./dataset/{data['dataset']}/train/dataset-{str(data['subset_id'])}.json") as fp:
                datalist = json.load(fp)
            random.shuffle(datalist)
            train_datalist.extend(datalist)
            
            with open(f"./dataset/{data['dataset']}/test/dataset-{str(data['subset_id'])}.json") as fp:
                datalist = json.load(fp)
            test_datalist.append({
                "data_name": f"{data['dataset']}-{data['subset_id']}",
                "data": datalist,
                "eval_cnt": eval_cnt})
            eval_cnt += len(datalist)
            
        train_datalists[client_id] = train_datalist
        test_datalists[client_id] = test_datalist
    
    return train_datalists, test_datalists

if __name__ == "__main__":
    main()
