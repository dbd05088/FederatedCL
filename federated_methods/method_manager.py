from typing import Callable, Tuple, Type, Dict

from federated_methods.fedavg import fedavg_load_state_dict, fedavg_aggregate_state_dict, fedavg_create_trainer
from federated_methods.sft import sft_load_state_dict
from federated_methods.fedper import fedper_set_state_dict, fedper_load_state_dict, fedper_half_set_state_dict, fedper_8_set_state_dict
from federated_methods.scaffold import scaffold_set_state_dict, scaffold_aggregate_state_dict, scaffold_create_trainer
from federated_methods.feddyn import feddyn_set_state_dict, feddyn_aggregate_state_dict, feddyn_create_trainer
from federated_methods.pfedpg import pfedpg_set_state_dict, pfedpg_aggregate_state_dict, pfedpg_create_trainer
from federated_methods.fedyogi import fedyogi_set_state_dict, fedyogi_aggregate_state_dict
from federated_methods.feddat import feddat_set_state_dict, feddat_create_trainer, feddat_aggregate_state_dict
from federated_methods.fedadapter import fedadapter_create_trainer
from federated_methods.fedprox import fedprox_set_state_dict, fedprox_create_trainer
from federated_methods.pfedme import pfedme_set_state_dict, pfedme_create_trainer, pfedme_aggregate_state_dict
from federated_methods.fedsim import fedsim_set_state_dict, fedsim_create_trainer
from federated_methods.ditto import ditto_create_trainer
from federated_methods.apfl import apfl_create_trainer
from federated_methods.dap_attn import dap_attn_create_trainer

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
    elif mode == 'fedper_half':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = fedper_half_set_state_dict, fedper_load_state_dict, fedavg_create_trainer, fedavg_aggregate_state_dict
    elif mode == 'fedper_8':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = fedper_8_set_state_dict, fedper_load_state_dict, fedavg_create_trainer, fedavg_aggregate_state_dict
    elif mode == 'scaffold':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = scaffold_set_state_dict, fedavg_load_state_dict, scaffold_create_trainer, scaffold_aggregate_state_dict
    elif mode == 'feddyn':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = feddyn_set_state_dict, fedavg_load_state_dict, feddyn_create_trainer, feddyn_aggregate_state_dict
    elif mode == 'pfedpg':
        # set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = pfedpg_set_state_dict, dummy_function, pfedpg_create_trainer, pfedpg_aggregate_state_dict
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = pfedpg_set_state_dict, sft_load_state_dict, pfedpg_create_trainer, fedavg_aggregate_state_dict
        # set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = pfedpg_set_state_dict, fedper_load_state_dict, pfedpg_create_trainer, fedavg_aggregate_state_dict
    elif mode == 'fedyogi':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = fedyogi_set_state_dict, fedavg_load_state_dict, fedavg_create_trainer, fedyogi_aggregate_state_dict
    elif mode == 'feddat':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = feddat_set_state_dict, fedper_load_state_dict, feddat_create_trainer, feddat_aggregate_state_dict
    elif mode == 'fedadapter':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedavg_load_state_dict, fedadapter_create_trainer, fedavg_aggregate_state_dict
    elif mode == 'fedprox':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = fedprox_set_state_dict, fedavg_load_state_dict, fedprox_create_trainer, fedavg_aggregate_state_dict
    elif mode =='pfedme':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = pfedme_set_state_dict, fedavg_load_state_dict, pfedme_create_trainer, pfedme_aggregate_state_dict
    elif mode == 'fedsim':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = fedsim_set_state_dict, fedper_load_state_dict, fedsim_create_trainer, fedavg_aggregate_state_dict
        # set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = fedsim_set_state_dict, fedper_load_state_dict, fedavg_create_trainer, fedavg_aggregate_state_dict
    elif mode =='ditto':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = fedsim_set_state_dict, fedper_load_state_dict, ditto_create_trainer, fedavg_aggregate_state_dict
    elif mode =='apfl':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = fedsim_set_state_dict, fedper_load_state_dict, apfl_create_trainer, fedavg_aggregate_state_dict
    elif mode =='l2p' or mode =='layer_l2p' or mode =='dap' or mode =='layer_l2p_text' or mode =='l2p_text':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, sft_load_state_dict, pfedpg_create_trainer, fedavg_aggregate_state_dict
        # set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, fedavg_load_state_dict, pfedpg_create_trainer, fedavg_aggregate_state_dict
    elif mode =='dap_attn' or mode == 'layer_l2p_attn':
        set_state_dict, load_state_dict, create_trainer, aggregate_state_dict = dummy_function, sft_load_state_dict, dap_attn_create_trainer, fedavg_aggregate_state_dict
    else:
        raise NotImplementedError(mode)
    return set_state_dict, load_state_dict, create_trainer, aggregate_state_dict, extra_modules