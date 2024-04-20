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
from methods.cl_manager_server import CLManagerServer
from methods.cl_manager_client import CLManagerClient

from utils.train_utils import get_llavamodel
from torch import multiprocessing
import copy
import torch.distributed as dist
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
            # load_in_4bit=training_args.bits == 4,
            # load_in_8bit=training_args.bits == 8,
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


    # num_samples = {'cifar10': 50000, 'cifar100': 50000, 'clear10':30000, 'clear100':100000, 'tinyimagenet': 100000, 'imagenet': 1281165}
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


    # get datalist
    # print("args.dataset", training_args.dataset, "num_samples", num_samples[training_args.dataset])
    # train_datalist, cls_dict, cls_addition = get_train_datalist(training_args.dataset, training_args.sigma, training_args.repeat, training_args.init_cls, training_args.seed)
    # test_datalist = get_test_datalist(training_args.dataset)
    train_datalist = get_train_datalist(training_args.dataset)
    test_datalist = get_test_datalist(training_args.dataset)
    print(len(train_datalist))
    print(len(test_datalist))
    samples_cnt = 0

    # FIXME
    # client별로 할당받을 datalist 얻기
    # train_datalists = list[train_datalist], len = num_clients
    # train_datalist = None
    # test_datalist = None

    # Reduce datalist in Debug mode
    if training_args.debug:
        random.shuffle(train_datalist)
        train_datalist = train_datalist[:5000]
        random.shuffle(test_datalist)
        test_datalist = test_datalist[:2000]

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

    server_rank = 0
    server = CLManagerServer(
                train_datalists=train_datalist,
                test_datalists=test_datalist,
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
        client_runner = CLManagerClient(
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

if __name__ == "__main__":
    main()
