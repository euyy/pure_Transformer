#coding=utf-8
'''
Description: 
version: 
Author: Yue Yang
Date: 2022-03-23 16:50:25
LastEditors: Yue Yang
LastEditTime: 2022-04-16 15:59:50
'''
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C

__C.DATASET_NAME = 'birds' # ! 数据集名称，birds/coco/flowers
__C.CONFIG_NAME = '' # ! 配置文件名称
__C.DATA_DIR = '' # ! 数据集路径
__C.GPU_ID = 0 # ! 如果是-1则是cpu，否则是gpu
__C.CUDA = True # ! 是否用cuda加速
__C.WORKERS = 6 # ! 同时加载几个batch数据集
__C.B_VALIDATION = False # ! 是否是验证阶段

__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 64 # ! 训练的批大小
__C.TRAIN.MAX_EPOCH = 600 # ! 最多训练多少代
__C.TRAIN.SNAPSHOT_INTERVAL = 2000 # ! 间隔多少代就保存一次模型
__C.TRAIN.FLAG = True # ! 是否训练
__C.TRAIN.GENERATOR_LR = 0.01
__C.TRAIN.DISCRIMINATOR_LR = 0.01

__C.TRAIN.SMOOTH = edict()
__C.TRAIN.SMOOTH.GAMMA1 = 5.0
__C.TRAIN.SMOOTH.GAMMA3 = 10.0
__C.TRAIN.SMOOTH.GAMMA2 = 5.0
__C.TRAIN.SMOOTH.LAMBDA = 1.0

__C.TEXT = edict()  
__C.TEXT.VOCAB_SIZE = 0 # ! 词典大小
__C.TEXT.CAPTIONS_PER_IMAGE = 10 # ! 每个图像有几个句子
__C.TEXT.EMBEDDING_DIM = 256 # ! 单词向量的维度
__C.TEXT.WORDS_NUM = 18 # ! 每个句子的最长长度

__C.IMAGE = edict()
__C.IMAGE.FEAT_EMB_DIM = 32 # ! 设置图像特征提取器的通道数, NDF*8 = d_model = 256
__C.IMAGE.SIZE = 64

__C.MODEL = edict()
__C.MODEL.TEXT_ENCODER = ''
__C.MODEL.IMAGE_DECODER = ''
__C.MODEL.IMAGE_ENCODER = ''
__C.MODEL.DISCRIMINATOR = ''
__C.MODEL.R_NUM = 3

__C.TREE = edict()
__C.TREE.BRANCH_NUM = 1

# # Dataset name: flowers, birds
# __C.DATASET_NAME = 'birds'
# __C.CONFIG_NAME = ''
# __C.DATA_DIR = ''
# __C.GPU_ID = 0
# __C.CUDA = True
# __C.WORKERS = 6

# __C.RNN_TYPE = 'LSTM'   # 'GRU'
# __C.B_VALIDATION = False

# __C.TREE = edict()
# __C.TREE.BRANCH_NUM = 3
# __C.TREE.BASE_SIZE = 64


# # Training options
# __C.TRAIN = edict()
# __C.TRAIN.BATCH_SIZE = 64
# __C.TRAIN.MAX_EPOCH = 600
# __C.TRAIN.SNAPSHOT_INTERVAL = 2000
# __C.TRAIN.DISCRIMINATOR_LR = 2e-4
# __C.TRAIN.GENERATOR_LR = 2e-4
# __C.TRAIN.ENCODER_LR = 2e-4
# __C.TRAIN.RNN_GRAD_CLIP = 0.25
# __C.TRAIN.FLAG = True
# __C.TRAIN.NET_E = ''
# __C.TRAIN.NET_G = ''
# __C.TRAIN.B_NET_D = True

# __C.TRAIN.SMOOTH = edict()
# __C.TRAIN.SMOOTH.GAMMA1 = 5.0
# __C.TRAIN.SMOOTH.GAMMA3 = 10.0
# __C.TRAIN.SMOOTH.GAMMA2 = 5.0
# __C.TRAIN.SMOOTH.LAMBDA = 1.0


# # Modal options
# __C.GAN = edict()
# __C.GAN.DF_DIM = 64
# __C.GAN.GF_DIM = 128
# __C.GAN.Z_DIM = 100
# __C.GAN.CONDITION_DIM = 100
# __C.GAN.R_NUM = 2
# __C.GAN.B_ATTENTION = True
# __C.GAN.B_DCGAN = False


# __C.TEXT = edict()
# __C.TEXT.CAPTIONS_PER_IMAGE = 10
# __C.TEXT.EMBEDDING_DIM = 256
# __C.TEXT.WORDS_NUM = 18


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    _merge_a_into_b(yaml_cfg, __C)
