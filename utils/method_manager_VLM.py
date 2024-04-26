import logging
from methods.FedAvg import FedAvg_client, FedAvg_server
from methods.FedUpperbound import FedUpperbound_client, FedUpperbound_server


def select_method(mode):
    if mode == "fedavg" :
        server, client = FedAvg_server, FedAvg_client
    elif mode == "fedupperbound" :
        server, client = FedUpperbound_server, FedUpperbound_client
    else:
        raise NotImplementedError(mode)

    return server, client
