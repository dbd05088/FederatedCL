from methods.cl_manager_server import CLManagerServer
from methods.cl_manager_client import CLManagerClient
from collections import OrderedDict
import torch

class FedProx_server(CLManagerServer):
    def setup(self):
        super().setup()
        
        self.state_dicts = []
        self.mean_state_dict = OrderedDict()
    
    def server_msg(self, client_id=None):
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
        self.state_dicts = []
        
        # self.evaluate_seendata(curr_round)
        self.save_server_model(curr_round)
        
class FedProx_client(CLManagerClient): 
    def setup(self):
        super().setup()
        
        self.global_model_param = OrderedDict()
        self.mu = 0.01
        
    # def before_optimizer_step(self):
    #     model_params = OrderedDict(self.model.named_parameters())
    #     for name, global_param in self.global_model_param.items():
    #         if model_params[name].grad is not None:
    #             global_param = global_param.to(self.device)
    #             model_params[name].grad.data += self.mu*torch.abs(model_params[name].data - global_param)

    def after_optimizer_step(self):
        model_params = OrderedDict(self.model.named_parameters())
        with torch.no_grad():
            for name, global_param in self.global_model_param.items():
                if 'mm_projector' in name:
                    model_params[name].copy_(model_params[name].data - self.mm_projector_lr*(self.mu*torch.abs(model_params[name].to(self.device) - global_param)))
                else:
                    model_params[name].copy_(model_params[name].data - self.lr*(self.mu*torch.abs(model_params[name].to(self.device) - global_param)))
        

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
        self.global_model_param = server_msg