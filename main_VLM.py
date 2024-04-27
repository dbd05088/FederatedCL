import logging.config
import os
import random

import numpy as np
import torch
from configuration.VLM_config import ModelArguments, DataArguments, TrainingArguments
import transformers
from utils.data_loader import get_test_datalist
from utils.data_loader import get_train_datalist
from utils.train_utils import get_VLMmodel

from utils.method_manager_VLM import select_method
from torch.utils.tensorboard import SummaryWriter

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

    train_datalists, test_datalists = get_datalists(training_args, training_args.scenario)
    
    # model, _, _ = get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args)
    # breakpoint()
    
    # from collections import OrderedDict
    # state_dict = OrderedDict()
    # for name, parameters in model.named_parameters():
    #     if 'vision_tower' in name or 'mm_projector' in name:
    #         state_dict[name] = parameters.cpu()
    
    # torch.save(state_dict, 'llava_vision_tower_mm_projector.pth')
    
    
    # create folder
    if not os.path.exists(training_args.state_dir):
        os.makedirs(training_args.state_dir)

    # start multiprocessing
    multiprocessing.set_start_method('spawn')
    processes = []

    num_process = min(training_args.n_gpu, training_args.num_clients + 1)
    client2server_queue = multiprocessing.Queue()
    server2client_queues = [
        multiprocessing.Queue() for _ in range(1, num_process)
    ]
    
    server, client = select_method(training_args.mode)

    server_rank = 0
    server = server(
                train_datalists=train_datalists,
                test_datalists=test_datalists,
                device=device,
                data_args=data_args,
                model_args=model_args,
                args=training_args,
                bnb_model_from_pretrained_args=bnb_model_from_pretrained_args,
                receive_channel=client2server_queue,
                send_channel=server2client_queues,
                logger=logger
            )
    server_process = multiprocessing.Process(
        target=run,
        args=(
            server_rank,
            num_process,
            'localhost',
            8001,
            server
        )
    )
    server_process.start()
    processes.append(server_process)

    for rank in range(1, num_process):
        args_copied = copy.deepcopy(training_args)
        bnb_model_from_pretrained_args_copied = copy.deepcopy(bnb_model_from_pretrained_args)
        device = torch.device(f"cuda:{rank}")
        args_copied.device = device
        bnb_model_from_pretrained_args_copied['device_map'] = {"":args_copied.device}
        client_runner = client(
            rank,
            device,
            data_args,
            model_args,
            args=args_copied,
            bnb_model_from_pretrained_args=bnb_model_from_pretrained_args_copied,
            receive_channel=server2client_queues[rank-1],
            send_channel=client2server_queue,
            logger=logger
        )
        p = multiprocessing.Process(target=run,
                args=(
                    rank,
                    num_process,
                    'localhost',
                    8001,
                    client_runner
                ))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def run(rank, world_size, master_addr, master_port, runner):
    print("Process {} start to run".format(rank))
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    # server process
    runner.setup()
    runner.run()
    
def get_datalists(args, scenario_num):
    with open(f"./scenarios/scenario-{scenario_num}.json") as fp:
        scenario = json.load(fp)
    assert args.num_clients == len(scenario)
    
    train_datalists = {}
    test_datalists = {}
    
    for data in scenario:
        with open(f"./scenarios/{data['dataset']}-{str(data['subset_id'])}.json") as fp:
            datalist = json.load(fp)
        train_datalists[data['client_id']] = datalist
        
        test_datalist = get_test_datalist(data['dataset'])
        test_datalists[data['client_id']] = {data['dataset']:test_datalist}
    
    return train_datalists, test_datalists

if __name__ == "__main__":
    main()
