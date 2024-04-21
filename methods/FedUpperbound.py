from methods.cl_manager_server import CLManagerServer
from methods.cl_manager_client import CLManagerClient
from collections import OrderedDict
from peft.tuners.lora import LoraLayer
from torch.utils.data import DataLoader
from utils.data_loader_llava import LazySupervisedDataset, DataCollatorForSupervisedDataset
import copy

class FedUpperbound_server(CLManagerServer):
    def setup(self):
        super().setup()
        
        self.client_data = []
    
    def server_msg(self):
        state_dict = OrderedDict()
        for name, module in self.model.named_modules():
            if isinstance(module, LoraLayer) or 'vision_tower' in name or 'mm_projector' in name:
                state_dict.update(module.state_dict())
        for k, v in state_dict.items():
            state_dict[k] = v.cpu()
        return state_dict
    
    def handle_msg_per_client(self, msg):
        assert isinstance(msg, list)
        
        self.client_data.extend(msg)
    
    def do_server_work(self):
        dataset = LazySupervisedDataset(self.client_data, self.tokenizer, self.data_args)
        dataloader = DataLoader(dataset, batch_size= self.batch_size, num_workers=self.n_worker, collate_fn=DataCollatorForSupervisedDataset(tokenizer=self.tokenizer))
        self.total_samples = len(dataloader)
        
        # train for one epoch
        self.model.train()
        for i, data in enumerate(dataloader):
            self.optimizer.zero_grad()
            data = self._prepare_inputs(data)
            loss = self.compute_loss(self.model, data)
            loss.backward()
            self.optimizer.step()
            
            self.report_training(i, loss)
        
        for i in range(len(self.test_datalists)):
            self.evaluate(i, self.test_datalists[i])
        
class FedUpperbound_client(CLManagerClient): 
    def setup(self):
        super().setup()
        
        self.trained_data = []
    
    def client_msg(self):
        return self.trained_data
    
    def train_one_round(self, curr_round, train_datalist, test_datalist):
        self.state['round_cnt'] += 1
        self.state['curr_round'] = curr_round
        
        # FIXME
        samples_per_round = 1000

        seen_so_far = self.state['sample_cnt']
        
        for i, data in enumerate(train_datalist[seen_so_far:seen_so_far+samples_per_round]):
            # explicit task boundary for twf
            # if samples_cnt % training_args.samples_per_task == 0 and training_args.mode in ["bic", "xder", "der_lider", "er_lider", "xder_lider", "co2l", "trire"]:
            #     method.online_before_task(task_id)
            #     task_id += 1
            self.trained_data.append(copy.deepcopy(data))

            self.state['sample_cnt'] += 1
            self.online_step(data, self.state['sample_cnt'], self.args.dataloader_num_workers)
            if self.state['sample_cnt'] % self.eval_period == 0:
                self.evaluate(test_datalist, 128, self.n_worker)

        self.save_state()
    
    def handle_server_msg(self, server_msg):
        assert isinstance(server_msg, OrderedDict)
        
        self.model.load_state_dict(server_msg, strict=False)
        self.train_data = []
