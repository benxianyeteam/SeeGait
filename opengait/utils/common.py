import copy
import os
import inspect
import logging
import torch
import numpy as np
import torch.nn as nn
import torch.autograd as autograd
import yaml
import random
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict, namedtuple


class NoOp:
    def __getattr__(self, *args):
        def no_op(*args, **kwargs): pass
        return no_op


class Odict(OrderedDict):
    def append(self, odict):
        dst_keys = self.keys()
        for k, v in odict.items():
            if not is_list(v):
                v = [v]
            if k in dst_keys:
                if is_list(self[k]):
                    self[k] += v
                else:
                    self[k] = [self[k]] + v
            else:
                self[k] = v


def Ntuple(description, keys, values):
    if not is_list_or_tuple(keys):
        keys = [keys]
        values = [values]
    Tuple = namedtuple(description, keys)
    return Tuple._make(values)


def get_valid_args(obj, input_args, free_keys=[]):
    if inspect.isfunction(obj):
        expected_keys = inspect.getfullargspec(obj)[0]
    elif inspect.isclass(obj):
        expected_keys = inspect.getfullargspec(obj.__init__)[0]
    else:
        raise ValueError('Just support function and class object!')
    unexpect_keys = list()
    expected_args = {}
    for k, v in input_args.items():
        if k in expected_keys:
            expected_args[k] = v
        elif k in free_keys:
            pass
        else:
            unexpect_keys.append(k)
    if unexpect_keys != []:
        logging.info("Find Unexpected Args(%s) in the Configuration of - %s -" %
                     (', '.join(unexpect_keys), obj.__name__))
    return expected_args


def get_attr_from(sources, name):
    try:
        return getattr(sources[0], name)
    except:
        return get_attr_from(sources[1:], name) if len(sources) > 1 else getattr(sources[0], name)


def is_list_or_tuple(x):
    return isinstance(x, (list, tuple))


def is_bool(x):
    return isinstance(x, bool)


def is_str(x):
    return isinstance(x, str)


def is_list(x):
    return isinstance(x, list) or isinstance(x, nn.ModuleList)


def is_dict(x):
    return isinstance(x, dict) or isinstance(x, OrderedDict) or isinstance(x, Odict)


def is_tensor(x):
    return isinstance(x, torch.Tensor)


def is_array(x):
    return isinstance(x, np.ndarray)


def ts2np(x):
    return x.cpu().data.numpy()


def ts2var(x, **kwargs):
    return autograd.Variable(x, **kwargs).cuda()


def np2var(x, **kwargs):
    return ts2var(torch.from_numpy(x), **kwargs)


def list2var(x, **kwargs):
    return np2var(np.array(x), **kwargs)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def MergeCfgsDict(src, dst):
    for k, v in src.items():
        if (k not in dst.keys()) or (type(v) != type(dict())):
            dst[k] = v
        else:
            if is_dict(src[k]) and is_dict(dst[k]):
                MergeCfgsDict(src[k], dst[k])
            else:
                dst[k] = v


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def config_loader(path):
    with open(path, 'r') as stream:
        src_cfgs = yaml.safe_load(stream)
    with open("/home/usr/dhy/mywork1_mutimodal_fusion/OpenGait-master/configs/default.yaml", 'r') as stream:
        dst_cfgs = yaml.safe_load(stream)
    MergeCfgsDict(src_cfgs, dst_cfgs)
    return dst_cfgs


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def handler(signum, frame):
    logging.info('Ctrl+c/z pressed')
    os.system(
        "kill $(ps aux | grep main.py | grep -v grep | awk '{print $2}') ")
    logging.info('process group flush!')


def ddp_all_gather(features, dim=0, requires_grad=True):
    '''
        inputs: [n, ...]
    '''

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    feature_list = [torch.ones_like(features) for _ in range(world_size)]
    torch.distributed.all_gather(feature_list, features.contiguous())

    if requires_grad:
        feature_list[rank] = features
    feature = torch.cat(feature_list, dim=dim)
    return feature


# https://github.com/pytorch/pytorch/issues/16885
class DDPPassthrough(DDP):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def get_ddp_module(module, find_unused_parameters=False, **kwargs):
    if len(list(module.parameters())) == 0:
        # for the case that loss module has not parameters.
        return module
    device = torch.cuda.current_device()
    module = DDPPassthrough(module, device_ids=[device], output_device=device,
                            find_unused_parameters=find_unused_parameters, **kwargs)
    return module


def params_count(net):
    n_parameters = sum(p.numel() for p in net.parameters())
    return 'Parameters Count: {:.5f}M'.format(n_parameters / 1e6)


def backbone_params_count_v2(net):
    # 1. 获取原始模型
    model = net.module if hasattr(net, 'module') else net

    # 2. 计算总参数
    total_params = sum(p.numel() for p in model.parameters())

    # 3. 计算分类头 (Head) 参数
    head_params = 0
    # 累加 FCs 的参数
    if hasattr(model, 'FCs'):
        head_params += sum(p.numel() for p in model.FCs.parameters())
    # 累加 BNNecks 的参数
    if hasattr(model, 'BNNecks'):
        head_params += sum(p.numel() for p in model.BNNecks.parameters())

    # 4. 相减得到 Backbone 参数
    backbone_params = total_params - head_params

    return 'Backbone Params: {:.5f}M'.format(backbone_params / 1e6)

def backbone_params_count_v1(net):
    # 1. 获取原始模型
    model = net.module if hasattr(net, 'module') else net

    # 2. 计算总参数
    total_params = sum(p.numel() for p in model.parameters())

    # 3. 计算分类头 (Head) 参数
    head_params = 0
    # 累加 FCs 的参数
    if hasattr(model, 'FCs_sil'):
        head_params += sum(p.numel() for p in model.FCs_sil.parameters())
    # 累加 BNNecks 的参数
    if hasattr(model, 'BNNecks'):
        head_params += sum(p.numel() for p in model.BNNecks.parameters())
    if hasattr(model, 'FCs_fuse'):
        head_params += sum(p.numel() for p in model.FCs_fuse.parameters())

    # 4. 相减得到 Backbone 参数
    backbone_params = total_params - head_params

    return 'Backbone Params: {:.5f}M'.format(backbone_params / 1e6)
def backbone_params_count(net):
    # 1. 处理 DDP (DistributedDataParallel) 包装的情况
    # 如果在多卡训练/测试，模型会被包裹在 .module 中
    model = net.module if hasattr(net, 'module') else net

    # 2. 定义 Backbone 包含的层名称
    # 对应 DeepGaitV2 中的 layer0 到 layer4
    backbone_names = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']

    n_backbone_parameters = 0

    # 3. 遍历这些层并累加参数
    for name in backbone_names:
        if hasattr(model, name):
            layer = getattr(model, name)
            n_backbone_parameters += sum(p.numel() for p in layer.parameters())

    return 'Backbone Params: {:.5f}M'.format(n_backbone_parameters / 1e6)