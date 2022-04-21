'''
Description: 用于对某些不清楚功能的函数或者方法的测试
version: 
Author: Yue Yang
Date: 2022-03-30 15:51:09
LastEditors: Yue Yang
LastEditTime: 2022-04-06 16:29:24
'''
# # ! 测试GELU是否能够直接用于三维的张量
# # ! 可以，没有维度的限制
# import torch
# import torch.nn as nn
# m = nn.GELU()
# input = torch.randn([3,5,5])
# print(input)
# output = m(input)
# print(output)


# # ! 测试 nn.Embedding 函数 padding_idx 参数的功能
# # ! 值为 padding_idx 的输入全部为0，且无梯度信息
# import torch 
# import torch.nn as nn
# from torch.autograd import Variable

# input = torch.LongTensor([1,2,3,0]).view(1,-1).transpose(0,1)
# emb = nn.Embedding(4, 5, padding_idx=0)
# output = emb(input)
# print(input)
# print(output)


# # ! 测试 Image.open 函数读入的图像是 [0,1] 还是 [0,255]
# # ! 测试失败，查阅后应该是 [0,255] ，数据集加载时的 norm 会将图像变成 [-1,1] 的 tensor
# from PIL import Image
# img_path = "/home/liby/Desktop/coco/000000030785.jpg"
# img = Image.open(img_path).convert('RGB')
# print(img)