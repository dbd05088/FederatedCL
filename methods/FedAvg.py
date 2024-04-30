from methods.cl_manager_server import CLManagerServer
from methods.cl_manager_client import CLManagerClient
from collections import OrderedDict
import torch

class FedAvg_server(CLManagerServer):
    def setup(self):
        super().setup()
        
        self.state_dicts = []
        self.mean_state_dict = OrderedDict()
    
    def server_msg(self):
        return self.mean_state_dict
    
    def handle_msg_per_client(self, msg):
        assert isinstance(msg, OrderedDict)
        
        self.state_dicts.append(msg)
    
    def do_server_work(self, curr_round):
        num_clients = len(self.state_dicts)
        if num_clients == 0:
            return
        keys = self.state_dicts[0].keys()
        
        mean_state_dict = OrderedDict()
        for k in keys:
            if isinstance(self.state_dicts[0][k], torch.Tensor):
                new_tensor = 0
                for i in range(len(self.state_dicts)):
                    new_tensor += self.state_dicts[i][k]
                mean_state_dict[k] = new_tensor / num_clients
            else:
                mean_state_dict[k] = self.state_dicts[0][k]
        
        self.model.load_state_dict(mean_state_dict, strict=False)
        self.mean_state_dict = mean_state_dict
        
        self.evaluate_seendata(curr_round)
        
class FedAvg_client(CLManagerClient): 
    def client_msg(self):
        state_dict = OrderedDict()
        with torch.no_grad():
            for name, parameters in self.model.named_parameters():
                if isinstance(parameters, torch.Tensor) and parameters.requires_grad:
                        state_dict[name] = parameters.detach().cpu()
        return state_dict
    
    def handle_server_msg(self, server_msg):
        assert isinstance(server_msg, OrderedDict)
        
        self.model.load_state_dict(server_msg, strict=False)