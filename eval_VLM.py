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

# import warnings
# warnings.filterwarnings('ignore')

def main():
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

    os.makedirs(f"results/{training_args.mode}/{training_args.note}", exist_ok=True)
    os.makedirs(f"tensorboard/{training_args.mode}/{training_args.note}", exist_ok=True)
    fileHandler = logging.FileHandler(f'results/{training_args.mode}/{training_args.note}/seed_{training_args.seed}.log', mode="w")

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
    breakpoint()
    
    train_datalists, test_datalists = get_datalists(training_args, training_args.scenario)
    # breakpoint()
    
    from utils.data_loader_VLM import DataCollatorForSupervisedDataset, LazySupervisedDataset, GenerationDataset
    from torch.utils.data import DataLoader
    from models.llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
    from utils.eval_metrics import NLPEvaluator, matching_token_num
    from collections.abc import Mapping
    def _prepare_input(data):
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: _prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(_prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": device}
            return data.to(**kwargs)
        return data

    def _prepare_inputs(inputs):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = _prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                "training dataset contains keys expected by the model"
            )
        return inputs

    # dataset = LazySupervisedDataset(test_datalists[5][0]['data'], tokenizer, data_args, preprocess=False)
    # dataloader = DataLoader(dataset, batch_size= 1, collate_fn=DataCollatorForSupervisedDataset(tokenizer=tokenizer))
    dataset = GenerationDataset(test_datalists[5][0]['data'], tokenizer, data_args)
    dataloader = DataLoader(dataset, batch_size= 1)
    # img_feat_size = 729
    model.eval()
    predictions = []
    total_loss = 0
    n_word_total = 0
    n_word_correct = 0
    cnt = 0
    with torch.no_grad():
        for i, (inputs, imgs, gold) in enumerate((dataloader)):
            # * prepare data
            # batch = _prepare_inputs(batch)
            # inputs = batch['input_ids']
            # input_labels = batch['labels']
            # imgs = batch['images']
            inputs = inputs.to(device)
            imgs = imgs.to(device=device, dtype=torch.bfloat16)
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
                    max_new_tokens=512,#args.max_new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            # breakpoint()
            # valid_label_mask = input_labels[0].ne(IGNORE_INDEX)
            input_token_len = inputs.shape[1] #[:,input_token_len:]
            
            # n_word = len(torch.unique(input_labels[0][valid_label_mask]))
            # n_correct = matching_token_num(output_ids[0], input_labels[0][valid_label_mask])
            pred_sentence = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0] #[:,input_token_len:]
            # gold_sentence = tokenizer.decode(input_labels[0][valid_label_mask], skip_special_tokens=True)
            predictions.append({"sentence":pred_sentence, "gt_sentence":gold})
            print(pred_sentence)
            print(gold)
            breakpoint()
            # n_word_total += n_word
            # n_word_correct += n_correct
            cnt += 1
    scores = NLPEvaluator(predictions).evaluate()
    scores["precision"] = n_word_correct / n_word_total
    scores["loss"] = total_loss / cnt

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
