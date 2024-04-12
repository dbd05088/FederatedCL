import logging.config
import os
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from configuration import config
from configuration.llava_config import ModelArguments, DataArguments, TrainingArguments
import transformers
from utils.data_loader import get_test_datalist
from utils.data_loader import get_train_datalist

from utils.method_manager_new import select_method


def main():
    # args = config.base_parser()

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
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


    num_samples = {'cifar10': 50000, 'cifar100': 50000, 'clear10':30000, 'clear100':100000, 'tinyimagenet': 100000, 'imagenet': 1281165}
    logging.config.fileConfig("./configuration/logging.conf")
    logger = logging.getLogger()

    os.makedirs(f"results/{training_args.dataset}/{training_args.note}", exist_ok=True)
    os.makedirs(f"tensorboard/{training_args.dataset}/{training_args.note}", exist_ok=True)
    fileHandler = logging.FileHandler(f'results/{training_args.dataset}/{training_args.note}/seed_{training_args.seed}.log', mode="w")

    formatter = logging.Formatter(
        "[%(levelname)s] %(filename)s:%(lineno)d > %(message)s"
    )
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    #writer = SummaryWriter(f'tensorboard/{args.dataset}/{args.note}/seed_{args.seed}')

    logger.info(training_args)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Set the device ({device})")

    # Fix the random seeds
    torch.manual_seed(training_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(training_args.seed)
    random.seed(training_args.seed)


    # get datalist
    print("args.dataset", training_args.dataset, "num_samples", num_samples[training_args.dataset])
    # train_datalist, cls_dict, cls_addition = get_train_datalist(training_args.dataset, args.sigma, args.repeat, args.init_cls, args.seed)
    # test_datalist = get_test_datalist(args.dataset)
    # samples_cnt = 0
    train_datalist = None
    test_datalist = None

    # Reduce datalist in Debug mode
    if training_args.debug:
        random.shuffle(train_datalist)
        train_datalist = train_datalist[:5000]
        random.shuffle(test_datalist)
        test_datalist = test_datalist[:2000]


    logger.info(f"Select a CIL method ({training_args.mode})")
    method = select_method(training_args, train_datalist, test_datalist, device, model_args=model_args, training_args=training_args,bnb_model_from_pretrained_args=bnb_model_from_pretrained_args)

    print("\n###flops###\n")
    #method.get_flops_parameter()

    eval_results = defaultdict(list)

    samples_cnt = 0
    task_id = 0

    for i, data in enumerate(train_datalist):

        # explicit task boundary for twf
        if samples_cnt % args.samples_per_task == 0 and training_args.mode in ["bic", "xder", "der_lider", "er_lider", "xder_lider", "co2l", "trire"]:
            method.online_before_task(task_id)
            task_id += 1

        samples_cnt += 1
        method.online_step(data, samples_cnt, training_args.dataloader_num_workers)
        if samples_cnt % args.eval_period == 0:
            eval_dict = method.online_evaluate(test_datalist, samples_cnt, 512, training_args.dataloader_num_workers, cls_dict,
                                               cls_addition, data["time"])
            eval_results["test_acc"].append(eval_dict['avg_acc'])
            eval_results["percls_acc"].append(eval_dict['cls_acc'])
            eval_results["data_cnt"].append(samples_cnt)
        
        if (args.mode in ["remind"]) and samples_cnt == args.baseinit_samples:
            method.finalize_baseinit()
        
        if samples_cnt % args.samples_per_task == 0 and (training_args.mode in ["memo", "xder", "afec", "sparcl", "trire"]) and samples_cnt != num_samples[args.dataset]:
            method.online_after_task()
        
    if eval_results["data_cnt"][-1] != samples_cnt:
        eval_dict = method.online_evaluate(test_datalist, samples_cnt, 512, training_args.dataloader_num_workers, cls_dict, cls_addition,
                                           data["time"])

    A_last = eval_dict['avg_acc']

    if training_args.mode == 'gdumb':
        eval_results = method.evaluate_all(test_datalist, args.memory_epoch, training_args.per_device_eval_batch_size, training_args.dataloader_num_workers, cls_dict, cls_addition)

    np.save(f'results/{training_args.dataset}/{training_args.note}/seed_{training_args.seed}_eval.npy', eval_results['test_acc'])
    np.save(f'results/{training_args.dataset}/{training_args.note}/seed_{training_args.seed}_eval_per_cls.npy', eval_results['percls_acc'])
    np.save(f'results/{training_args.dataset}/{training_args.note}/seed_{training_args.seed}_eval_time.npy', eval_results['data_cnt'])

    # Accuracy (A)
    A_auc = np.mean(eval_results["test_acc"])

    # KLR_avg = np.mean(method.knowledge_loss_rate[1:])
    # KGR_avg = np.mean(method.knowledge_gain_rate)
    KLR_avg = 0.0
    KGR_avg = 0.0

    logger.info(f"======== Summary =======")
    logger.info(f"A_auc {A_auc:6f} | A_last {A_last:6f} | KLR_avg {KLR_avg:6f} | KGR_avg {KGR_avg:6f} | Total FLOPs {method.total_flops:4f}")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
