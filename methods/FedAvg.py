from methods.cl_manager_server import CLManagerServer
from methods.cl_manager_client import CLManagerClient
from collections import OrderedDict
from peft.tuners.lora import LoraLayer

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
    
    def do_server_work(self):
        num_clients = len(self.state_dicts)
        if num_clients == 0:
            return
        keys = self.state_dicts[0].keys()
        
        mean_state_dict = OrderedDict()
        for k in keys:
            new_tensor = 0
            for i in range(len(self.state_dicts)):
                new_tensor += self.state_dicts[i][k]
            mean_state_dict[k] = new_tensor / num_clients
        
        # self.model.load_state_dict(mean_state_dict, strict=False)
        self.mean_state_dict = mean_state_dict
        
class FedAvg_client(CLManagerClient): 
    def client_msg(self):
        state_dict = OrderedDict()
        for name, module in self.model.named_modules():
            if isinstance(module, LoraLayer) or 'vision_tower' in name or 'mm_projector' in name:
                state_dict.update(module.state_dict().cpu())
        return state_dict
    
    def handle_server_msg(self, server_msg):
        assert isinstance(server_msg, OrderedDict)
        
        self.model.load_state_dict(server_msg, strict=False)