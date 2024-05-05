from methods.cl_manager_server import CLManagerServer
from methods.cl_manager_client import CLManagerClient
from collections import OrderedDict
import torch
import copy

ALPHA = 0.01

class FedDyn_server(CLManagerServer):
    def setup(self):
        super().setup()
        
        self.state_dicts = []
        self.mean_state_dict = OrderedDict()
        self.alpha = ALPHA
        self.server_state = OrderedDict()
        with torch.no_grad():
            for name, parameters in self.model.named_parameters():
                if isinstance(parameters, torch.Tensor) and parameters.requires_grad:
                    self.server_state[name] = torch.zeros_like(parameters.detach().cpu())
    
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
        
        # update server state
        model_delta = copy.deepcopy(self.server_state)
        for name, param in model_delta.items():
            model_delta[name] = torch.zeros_like(param)

        model_params = OrderedDict(self.model.named_parameters())
        with torch.no_grad():
            for client_params in self.state_dicts:
                for name, param in model_delta.items():
                    model_delta[name] = param + (client_params[name] - model_params[name].detach().cpu()) / len(self.state_dicts)

        for name, state_param in self.server_state.items():
            self.server_state[name] = state_param - self.alpha*model_delta[name]
        
        # fedavg
        mean_state_dict = OrderedDict()
        for k in keys:
            if isinstance(self.state_dicts[0][k], torch.Tensor):
                new_tensor = 0
                for i in range(len(self.state_dicts)):
                    new_tensor += self.state_dicts[i][k]
                mean_state_dict[k] = new_tensor / num_clients
                
                mean_state_dict[k] -= (1/self.alpha)*self.server_state[k]
            else:
                mean_state_dict[k] = self.state_dicts[0][k]
        
        self.model.load_state_dict(mean_state_dict, strict=False)
        self.mean_state_dict = mean_state_dict
        
        self.state_dicts = []
        
        # self.evaluate_seendata(curr_round)
        self.save_server_model(curr_round)
        
class FedDyn_client(CLManagerClient): 
    def setup(self):
        super().setup()
        
        self.global_model_vector = None
        self.alpha = ALPHA
        old_grad = copy.deepcopy(self.model)
        old_grad = model_parameter_vector(old_grad)
        self.old_grad = torch.zeros_like(old_grad)
        
    def before_optimizer_step(self):
        if self.global_model_vector != None:
            v1 = model_parameter_vector(self.model)
            loss = self.alpha/2 * torch.norm(v1 - self.global_model_vector, 2)**2
            loss -= torch.dot(v1, self.old_grad)
            loss.backward()

    def client_msg(self):
        # update old_grad on every round 
        # FIXME: every round or every optimizer step?
        if self.global_model_vector != None:
            v1 = model_parameter_vector(self.model).detach()
            self.old_grad = self.old_grad - self.alpha * (v1 - self.global_model_vector)
        
        state_dict = OrderedDict()
        with torch.no_grad():
            for name, parameters in self.model.named_parameters():
                if isinstance(parameters, torch.Tensor) and parameters.requires_grad:
                    state_dict[name] = parameters.detach().cpu()
        return state_dict
    
    def handle_server_msg(self, server_msg):
        assert isinstance(server_msg, OrderedDict)
        
        self.model.load_state_dict(server_msg, strict=False)
        
        if len(server_msg) > 0:
            param = [p.view(-1) for p in server_msg.values()]
            self.global_model_vector = torch.cat(param, dim=0).to(self.device)
        
def model_parameter_vector(model):
    param = [p.view(-1) for p in model.parameters() if p.requires_grad]
    return torch.cat(param, dim=0)