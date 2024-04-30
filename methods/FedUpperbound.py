from methods.cl_manager_server import CLManagerServer
from methods.cl_manager_client import CLManagerClient
from collections import OrderedDict
from torch.utils.data import DataLoader
from utils.data_loader_VLM import LazySupervisedDataset, DataCollatorForSupervisedDataset
import torch

class FedUpperbound_server(CLManagerServer):
    def setup(self):
        super().setup()
        
        self.client_data = []
        self.state_dict = None
    
    def server_msg(self):
        return self.state_dict
    
    def handle_msg_per_client(self, msg):
        assert isinstance(msg, list)
        
        self.client_data.extend(msg)
    
    def do_server_work(self):
        dataset = LazySupervisedDataset(self.client_data, self.tokenizer, self.data_args)
        dataloader = DataLoader(dataset, batch_size= self.batch_size, num_workers=self.n_worker, collate_fn=DataCollatorForSupervisedDataset(tokenizer=self.tokenizer), shuffle=True)
        self.total_samples = len(dataloader)
        
        if self.gradient_checkpointing:
            gradient_checkpointing_kwargs = {'use_reentrant':False}
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        
        # train for one epoch
        self.model.train()
        self.optimizer.zero_grad()
        for i, data in enumerate(dataloader):
            data = self._prepare_inputs(data)
            loss = self.compute_loss(self.model, data)
            loss.backward()
            if (i + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                # self.lr_scheduler.step()
                self.optimizer.zero_grad()
            
            self.report_training(i+1, loss)
        
        state_dict = OrderedDict()
        
        with torch.no_grad():
            for name, parameters in self.model.named_parameters():
                if isinstance(parameters, torch.Tensor) and parameters.requires_grad:
                        state_dict[name] = parameters.detach().cpu()

        self.state_dict = state_dict
        
        self.evaluate_seendata()
        
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
        samples_per_round = len(train_datalist) // self.num_rounds # 4

        seen_so_far = self.state['sample_cnt']
        
        if self.gradient_checkpointing:
            gradient_checkpointing_kwargs = {'use_reentrant':False}
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        
        self.optimizer.zero_grad()
        for i, data in enumerate(train_datalist[seen_so_far:seen_so_far+samples_per_round]):
            self.trained_data.append(data)

            self.state['sample_cnt'] += 1
            self.online_step(data, self.state['sample_cnt'], self.args.dataloader_num_workers)
            # if self.state['sample_cnt'] % self.eval_period == 0:
            
        # eval at the end of each round
        for data_info in test_datalist:
            if self.state['sample_cnt'] > data_info['eval_cnt']:
                self.evaluate(data_info['data_name'], data_info['data'])

        self.save_state()
    
    def handle_server_msg(self, server_msg):
        if not isinstance(server_msg, OrderedDict):
            return
        
        self.model.load_state_dict(server_msg, strict=False)
        self.trained_data = []
