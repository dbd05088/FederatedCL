from methods.cl_manager_server import CLManagerServer
from methods.cl_manager_client import CLManagerClient
from collections import OrderedDict
import torch
import copy

class Scaffold_server(CLManagerServer):
    def setup(self):
        super().setup()
        
        self.state_dicts = []
        self.aux_deltas = []
        self.mean_state_dict = OrderedDict()
        self.aux = OrderedDict()
        with torch.no_grad():
            for name, parameters in self.model.named_parameters():
                if isinstance(parameters, torch.Tensor) and parameters.requires_grad:
                        self.aux[name] = torch.zeros_like(parameters)
    
    def server_msg(self):
        return (self.mean_state_dict, self.aux)
    
    def handle_msg_per_client(self, msg):
        assert isinstance(msg, tuple) and len(msg) == 2
        
        self.state_dicts.append(msg[0])
        self.aux_deltas.appenx(msg[1])
    
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
        
        # update global aux
        for key in self.aux.keys():
            delta_auxiliary = sum([self.aux_deltas[i][key] for i in range(len(self.aux_deltas))]) 
            self.aux[key] += delta_auxiliary / len(self.aux_deltas)
        self.aux_deltas = []
        # self.evaluate_seendata(curr_round)
        self.save_server_model(curr_round)
        
class Scaffold_client(CLManagerClient): 
    def setup(self):
        super().setup()
        
        self.global_model_param = OrderedDict()
        self.local_aux = None
        self.global_aux = OrderedDict()
    
    def before_optimizer_step(self):
        model_params = self.model.state_dict()
        local_aux_params = self.local_aux.state_dict()
        for name, global_param in self.global_aux.items():
            global_param = global_param.to(self.device)
            model_params[name].grad.data += (global_param - local_aux_params[name])
        
        
    def client_msg(self):
        
        # update local_aux
        model_params = self.model.state_dict()
        original_local_aux = copy.deepcopy(self.local_aux)
        with torch.no_grad():
            for name, local_aux_param in self.local_aux.items():
                local_aux_param.data = local_aux_param - self.global_aux[name].to(self.device) + (self.global_model_param[name].to(self.device) - model_params[name]) / (self.samples_per_round*self.lr)
        
        state_dict = OrderedDict()
        with torch.no_grad():
            for name, parameters in self.model.named_parameters():
                if isinstance(parameters, torch.Tensor) and parameters.requires_grad:
                    state_dict[name] = parameters.detach().cpu()
        
        delta_aux = OrderedDict()
        with torch.no_grad():
            for k in self.local_aux.keys():
                delta_aux[k] = (self.local_aux[k] - original_local_aux[k]).detach().cpu()
        
        return (state_dict, delta_aux)
    
    def handle_server_msg(self, server_msg):
        assert isinstance(server_msg, tuple) and len(server_msg) == 2
        
        self.model.load_state_dict(server_msg[0], strict=False)
        self.global_model_param = server_msg[0]
        self.global_aux = server_msg[1]
        
        if self.local_aux is None:
            self.local_aux = copy.deepcopy(server_msg[1])