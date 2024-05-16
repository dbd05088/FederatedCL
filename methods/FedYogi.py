from methods.cl_manager_server import CLManagerServer
from methods.cl_manager_client import CLManagerClient
from collections import OrderedDict
import torch

class FedYogi_server(CLManagerServer):
    def setup(self):
        super().setup()
        
        self.state_dicts = []
        self.mean_state_dict = OrderedDict()
        self.proxy_dict = OrderedDict()
        self.opt_proxy_dict = OrderedDict()
        self.tau = 1e-3
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.eta = 1e-3
        with torch.no_grad():
            for name, parameters in self.model.named_parameters():
                if isinstance(parameters, torch.Tensor) and parameters.requires_grad:
                        self.mean_state_dict[name] = parameters.detach().cpu()
                        self.proxy_dict[name] = torch.zeros_like(parameters).cpu()
                        self.opt_proxy_dict[name] = (torch.ones_like(parameters)*self.tau**2).cpu()
    
    def server_msg(self, client_id=None):
        return self.mean_state_dict
    
    def handle_msg_per_client(self, msg):
        assert isinstance(msg, OrderedDict)
        
        self.state_dicts.append(msg)
    
    def do_server_work(self, curr_round):
        num_clients = len(self.state_dicts)
        if num_clients == 0:
            return

        for key, param in self.opt_proxy_dict.items():
            delta_w = sum([state_dict[key] - self.mean_state_dict[key] for state_dict in self.state_dicts]) / len(self.state_dicts)
            self.proxy_dict[key] = self.beta1 * self.proxy_dict[key] + (1 - self.beta1) * delta_w if curr_round > 0 else delta_w
            delta_square = torch.square(self.proxy_dict[key])
            self.opt_proxy_dict[key] = param - (1-self.beta2)*delta_square*torch.sign(param - delta_square)
            self.mean_state_dict[key] += self.eta * torch.div(self.proxy_dict[key], torch.sqrt(self.opt_proxy_dict[key])+self.tau)
        
        self.model.load_state_dict(self.mean_state_dict, strict=False)
        self.state_dicts = []
        
        # self.evaluate_seendata(curr_round)
        self.save_server_model(curr_round)
        
class FedYogi_client(CLManagerClient): 
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