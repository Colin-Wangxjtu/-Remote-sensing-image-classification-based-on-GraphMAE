from math import radians
import os
import argparse
import random
import yaml
import logging
from functools import partial
import numpy as np
import random
import dgl

import torch
import torch.nn as nn
from torch import optim as optim
from tensorboardX import SummaryWriter



logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    crt_labels, crt_times = preds[correct.bool()].unique(return_counts=True)
    label_correct_times = torch.zeros(len(y_true.unique()))
    for i, label in enumerate(crt_labels):
        label_correct_times[label] = crt_times[i]
    _, true_times = y_true.unique(return_counts=True)
    AA = label_correct_times / true_times.to('cpu')
    AA = AA.mean()
    correct = correct.sum().item()
    return correct / len(y_true), AA


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True


def get_current_lr(optimizer):
    return optimizer.state_dict()["param_groups"][0]["lr"]


def build_args():
    use_random = False
    hiper_para = {
        'max_epoch':random.randint(100, 3000),
        'num_heads':2**random.randint(0, 4),
        'num_layers':random.randint(1, 4),
        'num_hidden':2**random.randint(6, 12),
        'residual':random.choice([True, False]),
        'in_drop':random.uniform(0.1, 0.4),
        'attn_drop':random.uniform(0.05, 0.2),
        'lr':random.uniform(0.001, 0.01),
        'negative_slope':random.uniform(0.1, 0.5),
        'mask_rate':random.uniform(0.3, 0.7),
        'drop_edge_rate':random.uniform(0.03, 0.1),
        'replace_rate':random.uniform(0, 0.1),
        'alpha_l':random.uniform(2, 4),
        'pooling':random.choice(['max', 'mean']),
        'max_epoch_f':random.randint(20, 100),
        'batch_size':2**random.randint(4, 8)}
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--dataset", type=str, default="FlveoBig")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max_epoch", type=int, default=hiper_para['max_epoch'] if use_random else 399, # 200,best1300
                        help="number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=-1)

    parser.add_argument("--num_heads", type=int, default=hiper_para['num_heads'] if use_random else 4, # 4, best4
                        help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num_layers", type=int, default=hiper_para['num_layers'] if use_random else 4, # 2, best2
                        help="number of hidden layers")
    parser.add_argument("--num_hidden", type=int, default=hiper_para['num_hidden'] if use_random else 1024, # 256, best1024
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=hiper_para['residual'] if use_random else False,
                        help="use residual connection")
    parser.add_argument("--in_drop", type=float, default=hiper_para['in_drop'] if use_random else 0.3117, # .2, best.3
                        help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=hiper_para['attn_drop'] if use_random else 0.0526, # .1, best.1
                        help="attention dropout")
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--lr", type=float, default=hiper_para['lr'] if use_random else 0.00258, # 0.005, best 0.005
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument("--negative_slope", type=float, default=hiper_para['negative_slope'] if use_random else 0.149, # 0.2
                        help="the negative slope of leaky relu for GAT")
    parser.add_argument("--activation", type=str, default="prelu") # prelu
    parser.add_argument("--mask_rate", type=float, default=hiper_para['mask_rate'] if use_random else 0.4306) # 0.5, best0.45
    parser.add_argument("--drop_edge_rate", type=float, default=hiper_para['drop_edge_rate'] if use_random else 0.086) # 0.05, best0.05
    parser.add_argument("--replace_rate", type=float, default=hiper_para['replace_rate'] if use_random else 0.00909) # 0.05, best 0.06

    parser.add_argument("--encoder", type=str, default="gat")
    parser.add_argument("--decoder", type=str, default="gat")
    parser.add_argument("--loss_fn", type=str, default="sce")
    parser.add_argument("--alpha_l", type=float, default=hiper_para['alpha_l'] if use_random else 2.3776,
                         help="`pow`coefficient for `sce` loss") # 3, best3.5
    parser.add_argument("--optimizer", type=str, default="adam")
    
    parser.add_argument("--max_epoch_f", type=int, default=hiper_para['max_epoch_f'] if use_random else 150) # 30, best 40
    parser.add_argument("--lr_f", type=float, default=0.001, help="learning rate for evaluation")
    parser.add_argument("--weight_decay_f", type=float, default=0.0, help="weight decay for evaluation")
    parser.add_argument("--linear_prob", action="store_true", default=False)
    
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--logging", action="store_true")
    parser.add_argument("--scheduler", action="store_true", default=False)
    parser.add_argument("--concat_hidden", action="store_true", default=False)
    
    parser.add_argument("--s_branch_hid", type=int, default=128)
    parser.add_argument("--p_branch_hid", type=int, default=64)

    # for graph classification
    parser.add_argument("--pooling", type=str, default=hiper_para['pooling'] if use_random else 'max') # mean, best max
    parser.add_argument("--deg4feat", action="store_true", default=False, help="use node degree as input feature")
    parser.add_argument("--batch_size", type=int, default=hiper_para['batch_size'] if use_random else 128) # 32, best 64
    args = parser.parse_args()
    return args


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return nn.Identity


def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()

    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer


# -------------------
def mask_edge(graph, mask_prob):
    E = graph.num_edges()

    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


def drop_edge(graph, drop_rate, return_edges=False):
    if drop_rate <= 0:
        return graph

    n_node = graph.num_nodes()
    edge_mask = mask_edge(graph, drop_rate)
    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
    ng = ng.add_self_loop()

    dsrc = src[~edge_mask]
    ddst = dst[~edge_mask]

    if return_edges:
        return ng, (dsrc, ddst)
    return ng


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.dataset not in configs:
        logging.info("Best args not found")
        return args

    logging.info("Using best configs")
    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args


# ------ logging ------

class TBLogger(object):
    def __init__(self, log_path="./logging_data", name="run"):
        super(TBLogger, self).__init__()

        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)

        self.last_step = 0
        self.log_path = log_path
        raw_name = os.path.join(log_path, name)
        name = raw_name
        for i in range(1000):
            name = raw_name + str(f"_{i}")
            if not os.path.exists(name):
                break
        self.writer = SummaryWriter(logdir=name)

    def note(self, metrics, step=None):
        if step is None:
            step = self.last_step
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
        self.last_step = step

    def finish(self):
        self.writer.close()


class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError
        
    def forward(self, graph, x):
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias
