from typing import Callable, Tuple, Type, Dict

from federated_methods.fedavg import fedavg_load_state_dict, fedavg_aggregate_state_dict, fedavg_create_trainer
from federated_methods.sft import sft_load_state_dict
from federated_methods.fedper import fedper_set_state_dict, fedper_load_state_dict
from federated_methods.scaffold import scaffold_set_state_dict, scaffold_aggregate_state_dict, scaffold_create_trainer
from federated_methods.feddyn import feddyn_set_state_dict, feddyn_aggregate_state_dict, feddyn_create_trainer
from federated_methods.pfedpg import pfedpg_set_state_dict, pfedpg_aggregate_state_dict, pfedpg_create_trainer
from federated_methods.fedyogi import fedyogi_set_state_dict, fedyogi_aggregate_state_dict
from federated_methods.feddat import feddat_set_state_dict, feddat_load_state_dict, feddat_create_trainer, feddat_aggregate_state_dict

def dummy_function(*args):
    return {}

def select_method(mode: str) -> Tuple[Callable, Callable, Callable, Callable, Dict]:
    extra_modules = {}
    if mode == 'sft':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, sft_load_state_dict, fedavg_create_trainer, fedavg_aggregate_state_dict
    elif mode == 'fedavg':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedavg_load_state_dict, fedavg_create_trainer, fedavg_aggregate_state_dict
    elif mode == 'fedper':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = fedper_set_state_dict, fedper_load_state_dict, fedavg_create_trainer, fedavg_aggregate_state_dict
    elif mode == 'scaffold':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = scaffold_set_state_dict, fedavg_load_state_dict, scaffold_create_trainer, scaffold_aggregate_state_dict
    elif mode == 'feddyn':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = feddyn_set_state_dict, fedavg_load_state_dict, feddyn_create_trainer, feddyn_aggregate_state_dict
    elif mode == 'pfedpg':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = pfedpg_set_state_dict, dummy_function, pfedpg_create_trainer, pfedpg_aggregate_state_dict
    elif mode == 'fedyogi':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = fedyogi_set_state_dict, fedavg_load_state_dict, fedavg_create_trainer, fedyogi_aggregate_state_dict
    elif mode == 'feddat':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = feddat_set_state_dict, feddat_load_state_dict, feddat_create_trainer, feddat_aggregate_state_dict
    else:
        raise NotImplementedError(mode)
    return set_state_dict, load_state_dict, create_trainer, aggregate_state_dict, extra_modules