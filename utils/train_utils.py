import torch_optimizer
from easydict import EasyDict as edict
from torch import optim
import torch
import pandas as pd
from models import mnist, cifar, imagenet
from torch.utils.data import DataLoader
# from onedrivedownloader import download as dn
from torch.optim import SGD
import numpy as np
import timm
import copy


from utils.data_loader import get_train_datalist, ImageDataset, StreamDataset, MemoryDataset, cutmix_data, get_statistics, get_test_datalist
import torch.nn.functional as F
import kornia.augmentation as K
import torch.nn as nn
from torch import Tensor
from utils.my_augment import Kornia_Randaugment
from torchvision import transforms
from tqdm import tqdm

def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i
            
class CrossEntropyLossMaybeSmooth(nn.CrossEntropyLoss):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    def __init__(self, smooth_eps=0.0):
        super(CrossEntropyLossMaybeSmooth, self).__init__()
        self.smooth_eps = smooth_eps

    def forward(self, output, target, smooth=False):
        if not smooth:
            return F.cross_entropy(output, target)

        target = target.contiguous().view(-1)
        n_class = output.size(1)
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
        smooth_one_hot = one_hot * (1 - self.smooth_eps) + (1 - one_hot) * self.smooth_eps / (n_class - 1)
        log_prb = F.log_softmax(output, dim=1)
        loss = -(smooth_one_hot * log_prb).sum(dim=1).mean()
        return loss


class DataAugmentation(nn.Module):

    def __init__(self, inp_size, mean, std) -> None:
        super().__init__()
        self.randaugmentation = Kornia_Randaugment()
        self.inp_size = inp_size
        self.mean = mean
        self.std = std

        additional_aug = self.randaugmentation.form_transforms()
        self.transforms = nn.Sequential(
            K.Resize(size = (inp_size,inp_size)),
            K.RandomCrop(size = (inp_size,inp_size)),
            K.RandomHorizontalFlip(p=1.0),
            *additional_aug,
            K.Normalize(mean, std)
            )
        #self.cutmix = K.RandomCutMix(p=0.5)

    def set_cls_magnitude(self, option, current_cls_loss, class_count):
        self.randaugmentation.set_cls_magnitude(option, current_cls_loss, class_count)

    def get_cls_magnitude(self):
        return self.randaugmentation.get_cls_magnitude()

    def get_cls_num_ops(self):
        return self.randaugmentation.get_cls_num_ops()

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Tensor, labels=None) -> Tensor:
        #if labels is None or len(self.randaugmentation.cls_num_ops) == 0:
        additional_aug = self.randaugmentation.form_transforms()
        self.transforms = nn.Sequential(
            K.Resize(size = (self.inp_size, self.inp_size)),
            K.RandomCrop(size = (self.inp_size, self.inp_size)),
            K.RandomHorizontalFlip(p=1.0),
            *additional_aug,
            K.Normalize(self.mean, self.std)
            )
        #print("transform")
        #print(self.transforms)
        x_out = self.transforms(x)  # BxCxHxW
        '''
        else:
            additional_aug = self.randaugmentation.form_transforms(list(set((labels))))
            
            self.before_transforms = nn.Sequential(
                K.Resize(size = (self.inp_size, self.inp_size)),
                K.RandomCrop(size = (self.inp_size, self.inp_size)),
                K.RandomHorizontalFlip(p=1.0)
                )
            x_out = self.before_transforms(x)
            
            for i in range(len(x)):
                add_transform = nn.Sequential(*additional_aug[labels[i]])
                x_out[i] = add_transform(x_out[i])
                
            self.after_transforms = nn.Sequential(
                K.Normalize(self.mean, self.std)
                )
            x_out = self.transforms(x_out)  # BxCxHxW
        '''
        ##### check transforms
        # print("self.transform")
        # print(self.transforms)

        #x_out, _ = self.cutmix(x_out)
        return x_out


def get_transform(dataset, transform_list, gpu_transform, use_kornia=True):
    mean, std, n_classes, inp_size, _ = get_statistics(dataset=dataset)
    if use_kornia:
        train_transform = DataAugmentation(inp_size, mean, std)
    else:
        train_transform = []
        if "cutout" in transform_list:
            train_transform.append(Cutout(size=16))
            if gpu_transform:
                gpu_transform = False
                print("cutout not supported on GPU!")
        if "randaug" in transform_list:
            train_transform.append(transforms.RandAugment())
            
        if "autoaug" in transform_list:
            if hasattr(transform_list, 'AutoAugment'):
                if 'cifar' in dataset:
                    train_transform.append(transforms.AutoAugment(transforms.AutoAugmentPolicy('cifar10')))
                elif 'imagenet' in dataset:
                    train_transform.append(transforms.AutoAugment(transforms.AutoAugmentPolicy('imagenet')))
            else:
                train_transform.append(select_autoaugment(dataset))
                gpu_transform = False
        if "trivaug" in transform_list:
            train_transform.append(transforms.TrivialAugmentWide())
        if gpu_transform:
            train_transform = transforms.Compose([
                transforms.RandomCrop(inp_size, padding=4),
                transforms.RandomHorizontalFlip(),
                *train_transform,
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean, std),
            ])
        else:
            train_transform = transforms.Compose(
                [
                    transforms.Resize((inp_size, inp_size)),
                    transforms.RandomCrop(inp_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    *train_transform,
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
    print(f"Using train-transforms {train_transform}")

    test_transform = transforms.Compose(
        [
            transforms.Resize((inp_size, inp_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_transform, test_transform

def select_optimizer(opt_name, lr, model):
    if hasattr(model, 'fc'):
        fc_name = 'fc'
    elif hasattr(model, 'head'):
        fc_name = 'head'
    if "adam" in opt_name:
        params = [param for name, param in model.named_parameters() if fc_name not in name]
        opt = optim.Adam(params, lr=lr, weight_decay=0)
    elif "sgd" in opt_name:
        params = [param for name, param in model.named_parameters() if fc_name not in name]
        opt = optim.SGD(
            params, lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4
        )
    else:
        raise NotImplementedError("Please select the opt_name [adam, sgd]")
    if 'freeze_fc' not in opt_name:
        opt.add_param_group({'params': getattr(model, fc_name).parameters()})
    return opt

def select_scheduler(sched_name, opt, hparam=None):
    if "exp" in sched_name:
        scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=hparam)
    elif sched_name == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=1, T_mult=2
        )
    elif sched_name == "anneal":
        scheduler = optim.lr_scheduler.ExponentialLR(opt, 1 / 1.1, last_epoch=-1)
    elif sched_name == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(
            opt, milestones=[30, 60, 80, 90], gamma=0.1
        )
    elif sched_name == "const":
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1)
    else:
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1)
    return scheduler



def get_ckpt_remote_url(pre_dataset):
    if pre_dataset == "cifar100":
        return '<iframe src="https://onedrive.live.com/embed?cid=D3924A2D106E0039&resid=D3924A2D106E0039%21108&authkey=AFsCv4BR-bmTUII" width="98" height="120" frameborder="0" scrolling="no"></iframe>',"rs18_cifar100.pth"

    elif pre_dataset == "tinyimgR":
        return '<iframe src="https://onedrive.live.com/embed?cid=D3924A2D106E0039&resid=D3924A2D106E0039%21106&authkey=AKTxp5LFQJ9z9Ok" width="98" height="120" frameborder="0" scrolling="no"></iframe>', "erace_pret_on_tinyr.pth"

    elif pre_dataset == "imagenet":
        return '<iframe src="https://onedrive.live.com/embed?cid=D3924A2D106E0039&resid=D3924A2D106E0039%21107&authkey=ADHhbeg9cUoqJ0M" width="98" height="120" frameborder="0" scrolling="no"></iframe>',"rs50_imagenet_full.pth"

    else:
        raise ValueError("Unknown auxiliary dataset")


def load_initial_checkpoint(pre_dataset, model, device, load_cp_path = None):
    url, ckpt_name = get_ckpt_remote_url(pre_dataset)
    load_cp_path = load_cp_path if load_cp_path is not None else './checkpoints/'
    print("Downloading checkpoint file...")
    dn(url, load_cp_path)
    print(f"Downloaded in: {load_cp}")
    net = load_cp(load_cp_path, model, device, moco=True)
    print("Loaded!")
    return net

def generate_initial_checkpoint(net, pre_dataset, pre_epochs, num_aux_classes, device, opt_args):
    aux_dset, aux_test_dset = get_aux_dataset()
    net.fc = torch.nn.Linear(net.fc.in_features, num_aux_classes).to(device)
    net.train()
    opt = SGD(net.parameters(), lr=opt_args["lr"], weight_decay=opt_args["optim_wd"], momentum=opt_args["optim_mom"])
    sched = None
    if self.args.pre_dataset.startswith('cub'):
        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[80, 150, 250], gamma=0.5)
    elif 'tinyimg' in self.args.pre_dataset.lower():
        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[20, 30, 40, 45], gamma=0.5)

    for e in range(pre_epochs):
        for i, (x, y, _) in tqdm(enumerate(aux_dl), desc='Pre-training epoch {}'.format(e), leave=False, total=len(aux_dl)):
            y = y.long()
            opt.zero_grad()
            x = x.to(self.device)
            y = y.to(self.device)
            aux_out = net(x)
            aux_loss = loss(aux_out, y)
            aux_loss.backward()
            opt.step()

        if sched is not None:
            sched.step()
        if e % 5 == 4:
            print(e, f"{self.mini_eval()*100:.2f}%")
    from datetime import datetime
    # savwe the model
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    modelpath = "my_checkpoint" + '_' + now + '.pth'
    torch.save(net.state_dict(), modelpath)
    print(modelpath)

def load_cp(cp_path, net, device, moco=False) -> None:
    """
    Load pretrain checkpoint, optionally ignores and rebuilds final classifier.

    :param cp_path: path to checkpoint
    :param new_classes: ignore and rebuild classifier with size `new_classes`
    :param moco: if True, allow load checkpoint for Moco pretraining
    """
    print("net")
    print([name for name, _ in net.named_parameters()])
    s = torch.load(cp_path, map_location=device)
    print("s keys", s.keys())
    '''
    if 'state_dict' in s:  # loading moco checkpoint
        if not moco:
            raise Exception(
                'ERROR: Trying to load a Moco checkpoint without setting moco=True')
        s = {k.replace('encoder_q.', ''): i for k,
             i in s['state_dict'].items() if 'encoder_q' in k}
    '''

    #if not ignore_classifier: # online CL이므로 fc out-dim을 1부터 시작
    net.fc = torch.nn.Linear(
        net.fc.in_features, 1).to(device) # online이므로 num_aux_classes => 1

    for k in list(s):
        if 'fc' in k:
            s.pop(k)
    for k in list(s):
        if 'net' in k:
            s[k[4:]] = s.pop(k)
    for k in list(s):
        if 'wrappee.' in k:
            s[k.replace('wrappee.', '')] = s.pop(k)
    for k in list(s):
        if '_features' in k:
            s.pop(k)

    try:
        net.load_state_dict(s)
    except:
        _, unm = net.load_state_dict(s, strict=False)
        print("unm")
        print(unm)
        '''
        if new_classes is not None or ignore_classifier:
            assert all(['classifier' in k for k in unm]
                       ), f"Some of the keys not loaded where not classifier keys: {unm}"
        else:
            assert unm is None, f"Missing keys: {unm}"
        '''

    return net
'''
def partial_distill_loss(model, net_partial_features: list, pret_partial_features: list,
                         targets, teacher_forcing: list = None, extern_attention_maps: list = None):

    assert len(net_partial_features) == len(
        pret_partial_features), f"{len(net_partial_features)} - {len(pret_partial_features)}"

    if teacher_forcing is None or extern_attention_maps is None:
        assert teacher_forcing is None
        assert extern_attention_maps is None

    loss = 0
    attention_maps = []

    for i, (net_feat, pret_feat) in enumerate(zip(net_partial_features, pret_partial_features)):
        assert net_feat.shape == pret_feat.shape, f"{net_feat.shape} - {pret_feat.shape}"

        adapter = getattr(
            model, f"adapter_{i+1}")

        pret_feat = pret_feat.detach()

        if teacher_forcing is None:
            curr_teacher_forcing = torch.zeros(
                len(net_feat,)).bool().to(self.device)
            curr_ext_attention_map = torch.ones(
                (len(net_feat), adapter.c)).to(self.device)
        else:
            curr_teacher_forcing = teacher_forcing
            curr_ext_attention_map = torch.stack(
                [b[i] for b in extern_attention_maps], dim=0).float()

        adapt_loss, adapt_attention = adapter(net_feat, pret_feat, targets,
                                              teacher_forcing=curr_teacher_forcing, attention_map=curr_ext_attention_map)

        loss += adapt_loss
        attention_maps.append(adapt_attention.detach().cpu().clone().data)

    return loss / (i + 1), attention_maps
'''
def get_data_loader(opt_dict, dataset, pre_train=False):
    if pre_train:
        batch_size = 128
    else:
        batch_size = opt_dict['batchsize']

    # pre_dataset을 위한 dataset 불러오고 dataloader 생성
    train_transform, test_transform = get_transform(dataset, opt_dict['transforms'], opt_dict['gpu_transform'])

    test_datalist = get_test_datalist(dataset)
    train_datalist, cls_dict, cls_addition = get_train_datalist(dataset, opt_dict["sigma"], opt_dict["repeat"], opt_dict["init_cls"], opt_dict["rnd_seed"])

    # for debugging!
    # train_datalist = train_datalist[:2000]

    exp_train_df = pd.DataFrame(train_datalist)
    exp_test_df = pd.DataFrame(test_datalist)

    train_dataset = ImageDataset(
        exp_train_df,
        dataset=dataset,
        transform=train_transform,
        preload = True,
        use_kornia=True,
        #cls_list=exposed_classes, #cls_list none이면 알아서 label로 train
        data_dir=opt_dict["data_dir"]
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=opt_dict["n_worker"],
    )

    test_dataset = ImageDataset(
        exp_test_df,
        dataset=dataset,
        transform=test_transform,
        #cls_list=exposed_classes, #cls_list none이면 알아서 label로 train
        data_dir=opt_dict["data_dir"]
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=batch_size,#opt_dict["batchsize"],
        num_workers=opt_dict["n_worker"],
    )

    return train_loader, test_loader

def select_model(model_name, dataset, num_classes=None, opt_dict=None, G=False, F=False, ver2=False, channel_constant=1, kwinner = False):
    if model_name == 'vit':
        return timm.create_model('vit_base_patch16_224', pretrained=False)
    
    model_imagenet = False
    opt = edict(
        {
            "depth": 18,
            "num_classes": num_classes,
            "in_channels": 3,
            "bn": True,
            "normtype": "BatchNorm",
            "activetype": "ReLU",
            "pooltype": "MaxPool2d",
            "preact": False,
            "affine_bn": True,
            "bn_eps": 1e-6,
            "compression": 0.5,
        }
    )

    if "mnist" in dataset:
        model_class = getattr(mnist, "MLP")
    elif "cifar" in dataset:
        model_class = getattr(cifar, "ResNet")
        if G:
            model_class = getattr(cifar, "ResNet_G")
            opt["ver2"] = ver2
        elif F:
            model_class = getattr(cifar, "ResNet_F")
            opt["ver2"] = ver2
    elif "imagenet" in dataset or "clear" in dataset:
        #model_class = getattr(imagenet, "ResNet")
        model_imagenet=True
        model_class = getattr(cifar, "ResNet")
        if G:
            model_class = getattr(cifar, "ResNet_G")
            opt["ver2"] = ver2
        elif F:
            model_class = getattr(cifar, "ResNet_F")
            opt["ver2"] = ver2
    else:
        raise NotImplementedError(
            "Please select the appropriate datasets (mnist, cifar10, cifar100, imagenet)"
        )
    if model_name == "resnet18":
        opt["depth"] = 18
    elif model_name == "resnet34":
        opt["depth"] = 34
    elif model_name == "resnet32":
        opt["depth"] = 32
    elif model_name == "resnet20":
        opt["depth"] = 20
    elif model_name == "resnet44":
        opt["depth"] = 44
    elif model_name == "resnet56":
        opt["depth"] = 56
    elif model_name == "mlp400":
        opt["width"] = 400
    else:
        raise NotImplementedError(
            "Please choose the model name in [resnet18, resnet32, resnet34]"
        )

    model = model_class(opt, model_imagenet, channel_constant, kwinner)

    # TODO initial check
    initial = False

    # pre_dataset 설정
    pre_dataset = None
    pre_dataset_num_class = 0
    path_load_cp = None
    if dataset == "cifar10":
        pre_dataset = "cifar100"
        path_load_cp = "res18_cifar100_pretrained_model.pth" #"checkpoint/rs18_cifar100_new.pth"
        pre_dataset_num_class = 100
    elif dataset == "cifar100":
        pre_dataset = "tiny_imagenet"
        pre_dataset_num_class = 1000
    else:
        pre_dataset = "anything"

    assert pre_dataset is not None # none이면 설정이 덜 된것
    if opt_dict is not None:

        if initial:
            device = opt_dict['device']
            '''
            # pre_dataset을 위한 dataset 불러오고 dataloader 생성
            train_transform, test_transform = get_transform(pre_dataset, opt_dict['transforms'], opt_dict['gpu_transform'])

            test_datalist = get_test_datalist(pre_dataset)
            train_datalist, cls_dict, cls_addition = get_train_datalist(pre_dataset, opt_dict["sigma"], opt_dict["repeat"], opt_dict["init_cls"], opt_dict["rnd_seed"])

            # for debugging!
            # train_datalist = train_datalist[:2000]

            exp_train_df = pd.DataFrame(train_datalist)
            exp_test_df = pd.DataFrame(test_datalist)

            train_dataset = ImageDataset(
                exp_train_df,
                dataset=pre_dataset,
                transform=train_transform,
                preload = True,
                use_kornia=True,
                #cls_list=exposed_classes, #cls_list none이면 알아서 label로 train
                data_dir=opt_dict["data_dir"]
            )
            train_loader = DataLoader(
                train_dataset,
                shuffle=True,
                batch_size=128,
                num_workers=opt_dict["n_worker"],
            )
            test_dataset = ImageDataset(
                exp_test_df,
                dataset=pre_dataset,
                transform=test_transform,
                #cls_list=exposed_classes, #cls_list none이면 알아서 label로 train
                data_dir=opt_dict["data_dir"]
            )
            test_loader = DataLoader(
                test_dataset,
                shuffle=False,
                batch_size=128,#opt_dict["batchsize"],
                num_workers=opt_dict["n_worker"],
            )
            '''
            train_loader, test_loader = get_data_loader(opt_dict, pre_dataset, pre_train=True)

            model.fc = torch.nn.Linear(model.fc.in_features, pre_dataset_num_class).to(device)
            model.to(device)
            model.train()
            #opt = SGD(model.parameters(), lr=opt_dict["lr"], weight_decay=opt_dict["optim_wd"], momentum=opt_dict["optim_mom"])
            opt = optim.Adam(model.parameters(), lr=opt_dict["lr"])
            criterion = F.cross_entropy

            for epoch in range(opt_dict["pre_epoch"]):
            #for epoch in range(10):
                correct = 0
                num_data = 0
                total_loss = 0.0
                iteration = 0

                for i, data in tqdm(enumerate(train_loader), desc='Pre-training epoch {}'.format(epoch), leave=False, total=len(train_loader)):
                    model.train()
                    x = data["image"]
                    y = data["label"]
                    x = x.to(device)
                    y = y.to(device)
                    logit = model(x)
                    loss = criterion(logit, y)
                    loss.backward()
                    opt.step()

                    _, preds = logit.topk(1, 1, True, True)
                    correct += torch.sum(preds == y.unsqueeze(1)).item()
                    num_data += y.size(0)
                    total_loss += loss.item()
                    iteration+=1

                print(f"[TRAIN] epoch{epoch} loss",  total_loss / iteration, "accuracy", correct / num_data)

                if epoch % 10 == 0:
                    model.eval()
                    for i, data in enumerate(test_loader):
                        x = data["image"]
                        y = data["label"]
                        x = x.to(device)
                        y = y.to(device)
                        logit = model(x)
                        loss = criterion(logit, y)

                        _, preds = logit.topk(1, 1, True, True)
                        correct += torch.sum(preds == y.unsqueeze(1)).item()
                        num_data += y.size(0)
                        total_loss += loss.item()
                        iteration+=1

                    print("[TEST] epoch{epoch} loss",  total_loss / iteration, "accuracy", correct / num_data)

            torch.save(model.state_dict(), "res18_cifar100_pretrained_model.pth")        
        else:
            model = load_initial_checkpoint(pre_dataset, model, opt_dict["device"], load_cp_path = path_load_cp)
    return model

##### for ASER #####
def compute_knn_sv(model, eval_x, eval_y, cand_x, cand_y, k, device="cpu"):
    """
        Compute KNN SV of candidate data w.r.t. evaluation data.
            Args:
                model (object): neural network.
                eval_x (tensor): evaluation data tensor.
                eval_y (tensor): evaluation label tensor.
                cand_x (tensor): candidate data tensor.
                cand_y (tensor): candidate label tensor.
                k (int): number of nearest neighbours.
                device (str): device for tensor allocation.
            Returns
                sv_matrix (tensor): KNN Shapley value matrix of candidate data w.r.t. evaluation data.
    """
    # Compute KNN SV score for candidate samples w.r.t. evaluation samples
    n_eval = eval_x.size(0)
    n_cand = cand_x.size(0)
    # Initialize SV matrix to matrix of -1
    sv_matrix = torch.zeros((n_eval, n_cand), device=device)
    # Get deep features
    eval_df, cand_df = deep_features(model, eval_x, n_eval, cand_x, n_cand)
    # Sort indices based on distance in deep feature space
    sorted_ind_mat = sorted_cand_ind(eval_df, cand_df, n_eval, n_cand)

    # Evaluation set labels
    el = eval_y
    el_vec = el.repeat([n_cand, 1]).T
    # Sorted candidate set labels
    cl = cand_y[sorted_ind_mat]

    # Indicator function matrix
    indicator = (el_vec == cl).float()
    indicator_next = torch.zeros_like(indicator, device=device)
    indicator_next[:, 0:n_cand - 1] = indicator[:, 1:]
    indicator_diff = indicator - indicator_next

    cand_ind = torch.arange(n_cand, dtype=torch.float, device=device) + 1
    denom_factor = cand_ind.clone()
    denom_factor[:n_cand - 1] = denom_factor[:n_cand - 1] * k
    numer_factor = cand_ind.clone()
    numer_factor[k:n_cand - 1] = k
    numer_factor[n_cand - 1] = 1
    factor = numer_factor / denom_factor

    indicator_factor = indicator_diff * factor
    indicator_factor_cumsum = indicator_factor.flip(1).cumsum(1).flip(1)

    # Row indices
    row_ind = torch.arange(n_eval, device=device)
    row_mat = torch.repeat_interleave(row_ind, n_cand).reshape([n_eval, n_cand])

    # Compute SV recursively
    sv_matrix[row_mat, sorted_ind_mat] = indicator_factor_cumsum

    return sv_matrix


def deep_features(model, eval_x, n_eval, cand_x, n_cand):
    """
        Compute deep features of evaluation and candidate data.
            Args:
                model (object): neural network.
                eval_x (tensor): evaluation data tensor.
                n_eval (int): number of evaluation data.
                cand_x (tensor): candidate data tensor.
                n_cand (int): number of candidate data.
            Returns
                eval_df (tensor): deep features of evaluation data.
                cand_df (tensor): deep features of evaluation data.
    """
    # Get deep features
    if cand_x is None:
        num = n_eval
        total_x = eval_x
    else:
        num = n_eval + n_cand
        total_x = torch.cat((eval_x, cand_x), 0)

    # compute deep features with mini-batches
    total_x = maybe_cuda(total_x)
    deep_features_ = mini_batch_deep_features(model, total_x, num)

    eval_df = deep_features_[0:n_eval]
    cand_df = deep_features_[n_eval:]
    return eval_df, cand_df


def sorted_cand_ind(eval_df, cand_df, n_eval, n_cand):
    """
        Sort indices of candidate data according to
            their Euclidean distance to each evaluation data in deep feature space.
            Args:
                eval_df (tensor): deep features of evaluation data.
                cand_df (tensor): deep features of evaluation data.
                n_eval (int): number of evaluation data.
                n_cand (int): number of candidate data.
            Returns
                sorted_cand_ind (tensor): sorted indices of candidate set w.r.t. each evaluation data.
    """
    # Sort indices of candidate set according to distance w.r.t. evaluation set in deep feature space
    # Preprocess feature vectors to facilitate vector-wise distance computation
    eval_df_repeat = eval_df.repeat([1, n_cand]).reshape([n_eval * n_cand, eval_df.shape[1]])
    cand_df_tile = cand_df.repeat([n_eval, 1])
    # Compute distance between evaluation and candidate feature vectors
    distance_vector = euclidean_distance(eval_df_repeat, cand_df_tile)
    # Turn distance vector into distance matrix
    distance_matrix = distance_vector.reshape((n_eval, n_cand))
    # Sort candidate set indices based on distance
    sorted_cand_ind_ = distance_matrix.argsort(1)
    return sorted_cand_ind_


#### For x_der ###
def normalize(x, mean, std):
    assert len(x.shape) == 4
    return (x - torch.tensor(mean).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)) \
        / torch.tensor(std).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)


def random_flip(x):
    assert len(x.shape) == 4
    mask = torch.rand(x.shape[0]) < 0.5
    x[mask] = x[mask].flip(3)
    return x


def random_grayscale(x, prob=0.2):
    assert len(x.shape) == 4
    mask = torch.rand(x.shape[0]) < prob
    x[mask] = (x[mask] * torch.tensor([[0.299, 0.587, 0.114]]).unsqueeze(2).unsqueeze(2).to(x.device)).sum(1, keepdim=True).repeat_interleave(3, 1)
    return x


class strong_aug():
    def __init__(self, size, mean, std):
        self.transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            # transforms.ToTensor()
        ])
        self.mean = mean
        self.std = std

    def __call__(self, x):
        flip = random_flip(x)
        tmp = torch.stack(
                [self.transform(a) for a in flip]
            )
        tmp2 = random_grayscale(
            tmp)
        y = normalize(tmp2, self.mean, self.std)
        return y


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, reduction='mean'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.reduction = reduction

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean(0)

        return loss.mean() if self.reduction == 'mean' else loss.sum()

def re_init_weights(param_data, shape, device, reinint_method='xavier'):
    mask = torch.empty(shape, requires_grad=False, device=device)
    if len(mask.shape) < 2:
        mask = torch.unsqueeze(mask, 1)
        renint_usnig_method(param_data, mask, reinint_method)
        mask = torch.squeeze(mask, 1)
    else:
        renint_usnig_method(param_data, mask, reinint_method)
    return mask

def renint_usnig_method(param_data, mask, method='xavier'):
    if method == 'kaiming':
        nn.init.kaiming_uniform_(mask)
    elif method == 'normal':
        std, mean = torch.std_mean(param_data)
        nn.init.normal_(mask, mean, std)
    elif method == 'xavier':
        nn.init.xavier_uniform_(mask)

def create_dense_mask_0(net, device, value, fc=False):
    for name, param in net.named_parameters():
        if fc:
            if "fc" in name:
                param.data[param.data == param.data] = value
        else:
            param.data[param.data == param.data] = value
    net.to(device)
    return net

def test_sparsity(model, column=True, channel=True, filter=True, kernel=False):

    # --------------------- total sparsity --------------------
    total_zeros = 0
    total_nonzeros = 0
    layer_cont = 1

    for name, weight in model.named_parameters():
        if (len(weight.size()) == 4):# and "shortcut" not in name):
            zeros = np.sum(weight.cpu().detach().numpy() == 0)
            total_zeros += zeros
            non_zeros = np.sum(weight.cpu().detach().numpy() != 0)
            total_nonzeros += non_zeros
            print("(empty/total) weights of {}({}) is: ({}/{}). irregular sparsity is: {:.4f}".format(
                name, layer_cont, zeros, zeros+non_zeros, zeros / (zeros+non_zeros)))

        layer_cont += 1

    comp_ratio = float((total_zeros + total_nonzeros)) / float(total_nonzeros)
    total_sparsity = total_zeros / (total_zeros + total_nonzeros)

    print("---------------------------------------------------------------------------")
    print("total number of zeros: {}, non-zeros: {}, zero sparsity is: {:.4f}".format(
        total_zeros, total_nonzeros, total_zeros / (total_zeros + total_nonzeros)))
    print("only consider conv layers, compression rate is: {:.4f}".format(
        (total_zeros + total_nonzeros) / total_nonzeros))
    print("===========================================================================\n\n")

    # --------------------- column sparsity --------------------
    if(column):

        total_column = 0
        total_empty_column = 0
        layer_cont = 1
        for name, weight in model.named_parameters():
            if (len(weight.size()) == 4):# and "shortcut" not in name):
                weight2d = weight.reshape(weight.shape[0], -1)
                column_num = weight2d.shape[1]

                empty_column = np.sum(np.sum(np.absolute(weight2d.cpu().detach().numpy()), axis=0) == 0)
                print("(empty/total) column of {}({}) is: ({}/{}). column sparsity is: {:.4f}".format(
                    name, layer_cont, empty_column, weight.size()[1] * weight.size()[2] * weight.size()[3],
                                        empty_column / column_num))

                total_column += column_num
                total_empty_column += empty_column
            layer_cont += 1
        print("---------------------------------------------------------------------------")
        print("total number of zeros: {}, non-zeros: {}, zero sparsity is: {:.4f}".format(
            total_zeros, total_nonzeros, total_zeros / (total_zeros + total_nonzeros)))
        print("total number of column: {}, empty-column: {}, column sparsity is: {:.4f}".format(
            total_column, total_empty_column, total_empty_column / total_column))
        print("only consider conv layers, compression rate is: {:.4f}".format(
            (total_zeros + total_nonzeros) / total_nonzeros))
        print("===========================================================================\n\n")


    # --------------------- channel sparsity --------------------
    if (channel):

        total_channels = 0
        total_empty_channels = 0
        layer_cont = 1
        for name, weight in model.named_parameters():
            if (len(weight.size()) == 4):# and "shortcut" not in name):
                empty_channels = 0
                channel_num = weight.size()[1]

                for i in range(channel_num):
                    if np.sum(np.absolute(weight[:, i, :, :].cpu().detach().numpy())) == 0:
                        empty_channels += 1
                print("(empty/total) channel of {}({}) is: ({}/{}) ({}). channel sparsity is: {:.4f}".format(
                    name, layer_cont, empty_channels, weight.size()[1], weight.size()[1]-empty_channels, empty_channels / channel_num))

                total_channels += channel_num
                total_empty_channels += empty_channels
            layer_cont += 1
        print("---------------------------------------------------------------------------")
        print("total number of zeros: {}, non-zeros: {}, zero sparsity is: {:.4f}".format(
            total_zeros, total_nonzeros, total_zeros / (total_zeros + total_nonzeros)))
        print("total number of channels: {}, empty-channels: {}, channel sparsity is: {:.4f}".format(
            total_channels, total_empty_channels, total_empty_channels / total_channels))
        print("only consider conv layers, compression rate is: {:.4f}".format(
            (total_zeros + total_nonzeros) / total_nonzeros))
        print("===========================================================================\n\n")


    # --------------------- filter sparsity --------------------
    if(filter):

        total_filters = 0
        total_empty_filters = 0
        layer_cont = 1
        for name, weight in model.named_parameters():
            if (len(weight.size()) == 4):# and "shortcut" not in name):
                empty_filters = 0
                filter_num = weight.size()[0]

                for i in range(filter_num):
                    if np.sum(np.absolute(weight[i, :, :, :].cpu().detach().numpy())) == 0:
                        empty_filters += 1
                print("(empty/total) filter of {}({}) is: ({}/{}) ({}). filter sparsity is: {:.4f} ({:.4f})".format(
                    name, layer_cont, empty_filters, weight.size()[0], weight.size()[0]-empty_filters, empty_filters / filter_num, 1-(empty_filters / filter_num)))

                total_filters += filter_num
                total_empty_filters += empty_filters
            layer_cont += 1
        print("---------------------------------------------------------------------------")
        print("total number of zeros: {}, non-zeros: {}, zero sparsity is: {:.4f}".format(
            total_zeros, total_nonzeros, total_zeros / (total_zeros + total_nonzeros)))
        print("total number of filters: {}, empty-filters: {}, filter sparsity is: {:.4f}".format(
            total_filters, total_empty_filters, total_empty_filters / total_filters))
        print("only consider conv layers, compression rate is: {:.4f}".format(
            (total_zeros + total_nonzeros) / total_nonzeros))
        print("===========================================================================\n\n")

    # --------------------- kernel sparsity --------------------
    if(kernel):

        total_kernels = 0
        total_empty_kernels = 0
        layer_cont = 1
        for name, weight in model.named_parameters():
            if (len(weight.size()) == 4):# and "shortcut" not in name):
                shape = weight.shape
                npWeight = weight.cpu().detach().numpy()
                weight3d = npWeight.reshape(shape[0], shape[1], -1)

                empty_kernels = 0
                kernel_num = weight.size()[0] * weight.size()[1]

                for i in range(weight.size()[0]):
                    for j in range(weight.size()[1]):
                        if np.sum(np.absolute(weight3d[i, j, :])) == 0:
                            empty_kernels += 1
                print("(empty/total) kernel of {}({}) is: ({}/{}) ({}). kernel sparsity is: {:.4f}".format(
                    name, layer_cont, empty_kernels, kernel_num, kernel_num-empty_kernels, empty_kernels / kernel_num))

                total_kernels += kernel_num
                total_empty_kernels += empty_kernels
            layer_cont += 1
        print("---------------------------------------------------------------------------")
        print("total number of zeros: {}, non-zeros: {}, zero sparsity is: {:.4f}".format(
            total_zeros, total_nonzeros, total_zeros / (total_zeros + total_nonzeros)))
        print("total number of kernels: {}, empty-kernels: {}, kernel sparsity is: {:.4f}".format(
            total_kernels, total_empty_kernels, total_empty_kernels / total_kernels))
        print("only consider conv layers, compression rate is: {:.4f}".format(
            (total_zeros + total_nonzeros) / total_nonzeros))
        print("===========================================================================\n\n")
    return comp_ratio, total_sparsity

def copy_paste_fc(new_fc, prev_fc):
    prev_weight = copy.deepcopy(prev_fc.weight.data)
    prev_bias = copy.deepcopy(prev_fc.bias.data)
    with torch.no_grad():
        if new_fc.out_features > 1:
            new_fc.weight[:new_fc.out_features - 1] = prev_weight
            new_fc.bias[:new_fc.out_features - 1] = prev_bias
    return new_fc



import transformers
from models.llava.language_model.llava_llama import LlavaLlamaForCausalLM
from models.llava.language_model.llava_mpt import LlavaMptForCausalLM
from models.bunny import BunnyPhiForCausalLM, BunnyStableLMForCausalLM, BunnyQwen2ForCausalLM, BunnyMiniCPMForCausalLM, BunnyLlamaForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import models.llava.conversation as conversation_lib_llava
import models.bunny.conversation as conversation_lib_bunny
from transformers import Trainer
from peft.tuners.lora import LoraLayer
from models.bunny.prompt_tuning_model import Bunny_PT
from models.llava.prompt_tuning_model import Llava_PT

ACCESS_TOKEN = "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"

def get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args):
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    attn_implementation = "flash_attention_2"
    assert model_args.vision_tower is not None
    
    if training_args.mode == 'pfedpg':
        assert training_args.lora_enable == False, "no lora in pFedPG"
    
    # load tokenizer
    # for llava
    if model_args.model_type == "mpt":
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    elif model_args.model_type == 'llama': 
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    
    # for bunny
    elif (
        model_args.model_type == 'phi-1.5' or model_args.model_type == 'phi-2'
            or model_args.model_type == 'qwen1.5-1.8b' or model_args.model_type == 'minicpm'):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
        )
    elif model_args.model_type == 'llama3-8b':
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            token=ACCESS_TOKEN
        )
    elif model_args.model_type == 'stablelm-2':
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True
        )

    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    
    if model_args.model_type == 'llama3-8b':
        tokenizer.pad_token = tokenizer.eos_token
    
    if 'llava' in model_args.model_name_or_path.lower():
        # prompt tuning
        if training_args.mode == 'pfedpg':
            assert model_args.model_type != 'mpt'
            model = Llava_PT.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
            print('load pfedpg')
        if 'mpt' == model_args.model_type:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
    
    elif 'bunny' in model_args.model_name_or_path.lower():
        # prompt tuning
        if training_args.mode == 'pfedpg':
            assert model_args.model_type == 'phi-2'
            model = Bunny_PT.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **bnb_model_from_pretrained_args
            )
            print('load pfedpg')
        elif model_args.model_type == 'phi-1.5' or model_args.model_type == 'phi-2':
            model = BunnyPhiForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **bnb_model_from_pretrained_args
            )
        elif model_args.model_type == 'stablelm-2':
            model = BunnyStableLMForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        elif model_args.model_type == 'qwen1.5-1.8b':
            model = BunnyQwen2ForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        elif model_args.model_type == 'minicpm':
            model = BunnyMiniCPMForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        elif model_args.model_type == 'llama3-8b':
            model = BunnyLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                token = ACCESS_TOKEN,
                **bnb_model_from_pretrained_args
            )
        else:
            raise ValueError(f"Unknown Model Type {model_args.model_type}")    

    model.config.use_cache = False
    model.model.requires_grad_(False)

    # FIXME
    if training_args.bits >= 16:
        # print(training_args.device)
        model = model.to(training_args.device)
    
    
    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        
        # rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)


    if 'llava' in model_args.model_name_or_path.lower():
        if model_args.version in conversation_lib_llava.conv_templates:
            conversation_lib_llava.default_conversation = conversation_lib_llava.conv_templates[model_args.version]
        else:
            conversation_lib_llava.default_conversation = conversation_lib_llava.conv_templates["vicuna_v1"]
            
    elif 'bunny' in model_args.model_name_or_path.lower():
        if model_args.version in conversation_lib_bunny.conv_templates:
            conversation_lib_bunny.default_conversation = conversation_lib_bunny.conv_templates[model_args.version]
        else:
            conversation_lib_bunny.default_conversation = conversation_lib_bunny.conv_templates["default"]

    # load vision tower
    # if model_args.vision_tower is not None:
    model.get_model().initialize_vision_modules(
        model_args=model_args,
        # fsdp=training_args.fsdp
    )

    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
    # vision_tower.requires_grad_(True)
    
    # if not training_args.is_eval:
    #     data_args.img_mean = vision_tower.image_processor.image_mean
    #     data_args.img_std = vision_tower.image_processor.image_std
    #     vision_tower.image_processor.do_normalize=False
    # vision_tower.image_processor.do_rescale=False
    data_args.image_processor = vision_tower.image_processor
    
    data_args.is_multimodal = True

    model.config.image_aspect_ratio = "pad" #data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    
    if training_args.bits in [4, 8]:
        model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    
    if 'llava' in model_args.model_name_or_path.lower():
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer)or isinstance(module, torch.nn.LayerNorm):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            # if 'norm' in name and 'vision_tower' not in name:
            #     module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    
    return model, tokenizer, data_args

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names

def get_decay_parameter_names(model):
    """
    Get all parameter names that weight decay will be applied to

    Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
    apply to those modules since this function only filter out instance of nn.LayerNorm
    """
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    return decay_parameters

def deepspeed_init(model, training_args, accelerator, num_training_steps, inference=False):
    """
    Init DeepSpeed, after updating the DeepSpeed configuration with any relevant Trainer's args.

    If `resume_from_checkpoint` was passed then an attempt to resume from a previously saved checkpoint will be made.

    Args:
        trainer: Trainer object
        num_training_steps: per single gpu
        resume_from_checkpoint: path to a checkpoint if to resume from after normal DeepSpeedEngine load
        inference: launch in inference mode (no optimizer and no lr scheduler)
        auto_find_batch_size: whether to ignore the `train_micro_batch_size_per_gpu` argument as it's being
            set automatically by the auto batch size finder

    Returns: optimizer, lr_scheduler

    We may use `deepspeed_init` more than once during the life of Trainer, when we do - it's a temp hack based on:
    https://github.com/microsoft/DeepSpeed/issues/1394#issuecomment-937405374 until Deepspeed fixes a bug where it
    can't resume from a checkpoint after it did some stepping https://github.com/microsoft/DeepSpeed/issues/1612

    """
    from deepspeed.utils import logger as ds_logger

    args = training_args

    hf_deepspeed_config = accelerator.state.deepspeed_plugin.hf_ds_config

    # resume config update - some bits like `model` and `num_training_steps` only become available during train
    hf_deepspeed_config.trainer_config_finalize(args, model, num_training_steps)

    # set the Deepspeed log level consistent with the Trainer
    ds_logger.setLevel(args.get_process_log_level())

    if inference:
        # only Z3 makes sense for the inference
        if not hf_deepspeed_config.is_zero3():
            raise ValueError("ZeRO inference only makes sense with ZeRO Stage 3 - please adjust your config")

        # in case the training config is re-used for inference
        hf_deepspeed_config.del_config_sub_tree("optimizer")
        hf_deepspeed_config.del_config_sub_tree("lr_scheduler")
        optimizer, lr_scheduler = None, None
        model_parameters = None
    else:
        model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer, lr_scheduler = deepspeed_optim_sched(
            model, hf_deepspeed_config, args, num_training_steps, model_parameters
        )

    # keep for quick debug:
    # from pprint import pprint; pprint(config)

    return optimizer, lr_scheduler

def deepspeed_optim_sched(model, hf_deepspeed_config, args, num_training_steps, model_parameters):
    """
    A convenience wrapper that deals with optimizer and lr scheduler configuration.
    """
    from accelerate.utils import DummyOptim, DummyScheduler

    config = hf_deepspeed_config.config

    # Mixing and matching DS schedulers and optimizers is supported unless Offload is enabled in which case it's:
    # 1. DS scheduler + DS optimizer: Yes
    # 2. HF scheduler + HF optimizer: Mostly*
    # 3. DS scheduler + HF optimizer: Mostly*
    # 4. HF scheduler + DS optimizer: Yes
    #
    # Mostly*: All non-native DeepSpeed optimizers that have both CPU and GPU implementation should work (except LAMB)

    optimizer = None
    if "optimizer" in config:
        if args.adafactor:
            raise ValueError(
                "--adafactor was passed, but also found `optimizer` configured in the DeepSpeed config. "
                "Only one optimizer can be configured."
            )
        optimizer = DummyOptim(params=model_parameters)
    else:
        # ds supports Adam, OneBitAdam, and Lamb optimizers and can import other optimizers from torch.
        # But trainer uses AdamW by default.
        optimizer = get_llava_optimizer(model, args)
        # To use other optimizers requires voiding warranty with: `zero_allow_untested_optimizer`
        config["zero_allow_untested_optimizer"] = True

    lr_scheduler = None
    if "scheduler" in config:
        lr_scheduler = DummyScheduler(optimizer)
    else:
        if isinstance(optimizer, DummyOptim):

            def _lr_scheduler_callable(optimizer):
                # create a shallow copy first, so later modifications do not affect original trainer
                trainer_copy = copy.copy(trainer)
                # at the time _lr_scheduler_callable is called, trainer.lr_scheduler has been set
                # update it to None so that we can re-create a new scheduler
                trainer_copy.lr_scheduler = None
                lr_scheduler = trainer_copy.create_scheduler(
                    num_training_steps=num_training_steps, optimizer=optimizer
                )
                return lr_scheduler

            lr_scheduler = DummyScheduler(optimizer, lr_scheduler_callable=_lr_scheduler_callable)
        else:
            lr_scheduler = create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)

    return optimizer, lr_scheduler