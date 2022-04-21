#coding=utf-8
'''
Description: 
version: 
Author: Yue Yang
Date: 2022-03-23 15:05:06
LastEditors: Yue Yang
LastEditTime: 2022-04-12 15:36:13
'''

from __future__ import print_function

from miscc.config import cfg, cfg_from_file
from datasets import TextDataset
from trainer import condGANTrainer as trainer

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np

import torch
import torchvision.transforms as transforms

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a AttnGAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird_attn2.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=-1)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


def gen_example(wordtoix, algo):
    '''
    description: example_filesnames.txt中保存的是测试用例的文件名,每个文件名对应一个图像,有5个句子
                这是用于测试阶段的
    param {*} wordtoix: 词典,从单词转成index
    param {*} algo: trainer的实例
    return {*}
    '''
    '''generate images from example sentences'''
    from nltk.tokenize import RegexpTokenizer
    filepath = '%s/example_filenames.txt' % (cfg.DATA_DIR)
    data_dic = {}
    with open(filepath, "r") as f:
        filenames = f.read().split('\n')
        for name in filenames:
            if len(name) == 0:
                continue
            filepath = '%s/%s.txt' % (cfg.DATA_DIR, name)
            with open(filepath, "r") as f:
                print('Load from:', name)
                sentences = f.read().split('\n')
                # a list of indices for a sentence
                captions = []
                cap_lens = []
                for sent in sentences:
                    if len(sent) == 0:
                        continue
                    sent = sent.replace("\ufffd\ufffd", " ")
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(sent.lower())
                    if len(tokens) == 0:
                        print('sent', sent)
                        continue

                    rev = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0 and t in wordtoix:
                            rev.append(wordtoix[t])
                    captions.append(rev)
                    cap_lens.append(len(rev))
            max_len = np.max(cap_lens)

            sorted_indices = np.argsort(cap_lens)[::-1]
            cap_lens = np.asarray(cap_lens)
            cap_lens = cap_lens[sorted_indices]
            cap_array = np.zeros((len(captions), max_len), dtype='int64')
            for i in range(len(captions)):
                idx = sorted_indices[i]
                cap = captions[idx]
                c_len = len(cap)
                cap_array[i, :c_len] = cap
            key = name[(name.rfind('/') + 1):]
            data_dic[key] = [cap_array, cap_lens, sorted_indices]
    algo.gen_example(data_dic)


if __name__ == "__main__":
    #================================================================
    # # ! 从命令行加载配置信息，并和配置文件合并
    #================================================================
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    #================================================================
    # # ! 设置随机种子
    #================================================================
    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    #================================================================
    # # ! 输出目录在上级目录下的output文件夹下的开始时间戳下
    #================================================================
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG:
        # bshuffle = False
        split_dir = 'test'

    #================================================================
    # # ! 加载数据加载器
    #================================================================
    # Get data loader
    # ! imsize = 64*2^(3-1) = 256
    # imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    imsize = cfg.IMAGE.SIZE
    # ! img: [3, ih, iw]
    # !   => [3, imsize * 76 / 64=304, imsize * 76 / 64=304] 
    # !   => [3, imsize=256, imsize=256]
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    dataset = TextDataset(cfg.DATA_DIR, split_dir,
                          base_size=cfg.IMAGE.SIZE,
                          transform=image_transform)
    assert dataset
    cfg.TEXT.VOCAB_SIZE = dataset.n_words
    # ! drop_last = True: 最后一个不完整的batch丢弃
    # ! shuffle = True: 数据重新洗牌打乱顺序
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

    # ! 实例化trainer
    # Define models and go to train/evaluate
    algo = trainer(output_dir, dataloader, dataset.n_words, dataset.ixtoword)

    start_t = time.time()
    if cfg.TRAIN.FLAG: # ! 开始训练
        algo.train()
    else:
        '''generate images from pre-extracted embeddings'''
        if cfg.B_VALIDATION: # ! 验证模型，在验证集上生成单个图像并保存
            algo.sampling(split_dir)  # generate images for the whole valid dataset
        # else: # ! 测试模型，从指定文件的文本生成图像
        #     gen_example(dataset.wordtoix, algo)  # generate images for customized captions
    end_t = time.time()
    print('Total time for training:', end_t - start_t)
