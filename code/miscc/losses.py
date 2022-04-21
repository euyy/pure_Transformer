'''
Description: 
version: 
Author: Yue Yang
Date: 2022-04-07 14:29:19
LastEditors: Yue Yang
LastEditTime: 2022-04-21 12:33:50
'''
import torch
import torch.nn as nn
import numpy as np
from miscc.config import cfg 

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """
    Returns cosine similarity between x1 and x2, computed along dim.
    return: x1*x2 / ( |x1| * |x2| )
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def words_loss(img_features, words_emb, labels,
               cap_lens, class_ids, batch_size):
    '''
    description: 传入图像特征 [bs, emb, w, h] 和单词特征 [bs, emb, cap_len]
                返回两个loss
    param {*} img_features: tensor(14,256,17,17)
    param {*} words_emb: tensor(14,256,12)
    param {*} labels: tensor(14)
    param {*} cap_lens: tensor(14)
    param {*} class_ids: array(14)
    param {*} batch_size: 14
    return {*} loss0: tensor(1)
    return {*} loss1: tensor(1)
    return {*} att_maps: tensor(bs, sent_len, 17, 17)
    '''
    
    """
        words_emb(query): batch x nef x seq_len
        img_features(context): batch x nef x 17 x 17
    """

    masks = []
    att_maps = []
    similarities = []
    # cap_lens = cap_lens.data.tolist() # ! list[14]
    for i in range(batch_size):
        if class_ids is not None:
            mask = (class_ids == class_ids[i]).astype(np.uint8) # ! array(14)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        # Get the i-th text description
        words_num = cap_lens[i]
        # -> 1 x nef x words_num
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous() # ! tensor(1,256,cap_len)
        # -> batch_size x nef x words_num
        word = word.repeat(batch_size, 1, 1) # ! tensor(14,256,cap_len)
        # batch x nef x 17*17
        context = img_features.view(batch_size,-1,16,16)  # ! tensor(14,256,17,17)
        """
            word(query): batch x nef x words_num
            context: batch x nef x 17 x 17
            weiContext: batch x nef x words_num
            attn: batch x words_num x 17 x 17
        """
        '''
        weiContext: tensor(14,256,cap_len), 图像引导后的文本特征
        attn: tensor(14,cap_len,17,17)
        第i个样本的单词特征向量和所有样本的图像特征进行注意力操作
        不同的样本句子长度cap_len可能不一样
        '''
        
        weiContext, attn = func_attention(word, context, cfg.TRAIN.SMOOTH.GAMMA1)
        
        att_maps.append(attn[i].unsqueeze(0).contiguous())
        # --> batch_size x words_num x nef
        word = word.transpose(1, 2).contiguous() # ! tensor(14,cap_len,256),第i个样本单词特征复制成bs个样本
        weiContext = weiContext.transpose(1, 2).contiguous() # ! tensor(14,cap_len,256),第i个样本单词在所有样本图像特征的注意力引导下,生成的新的单词特征
        # --> batch_size*words_num x nef
        word = word.view(batch_size * words_num, -1) # ! tensor(14*cap_len,256)
        weiContext = weiContext.view(batch_size * words_num, -1) # ! tensor(14*cap_len,256)
        #
        # -->batch_size*words_num
        row_sim = cosine_similarity(word, weiContext) # ! tensor(14*cap_len) ,计算了新生成的单词特征与原有的单词特征之间的相似性
        # --> batch_size x words_num
        row_sim = row_sim.view(batch_size, words_num) # ! tensor(14, cap_len) 

        # Eq. (10)
        row_sim.mul_(cfg.TRAIN.SMOOTH.GAMMA2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True) # ! tensor(14，1) 
        row_sim = torch.log(row_sim) # ! tensor(14，1) 

        # --> 1 x batch_size
        # similarities(i, j): the similarity between the i-th image and the j-th text description
        similarities.append(row_sim)

    # batch_size x batch_size
    similarities = torch.cat(similarities, 1) # ! tensor(14,14)
    if class_ids is not None:
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.ByteTensor(masks)
        if cfg.CUDA:
            masks = masks.cuda() # ! tensor(14,14)

    similarities = similarities * cfg.TRAIN.SMOOTH.GAMMA3
    if class_ids is not None:
        similarities.data.masked_fill_(masks == 1, -float('inf'))
    similarities1 = similarities.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(similarities, labels) # ! tensor(1),logSoftmax + nllloss, nllloss是根据labels选择每个样本的值,取负值,求和,再求均值
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1, att_maps


def func_attention(query, context, gamma1): 
    '''
    description: 单词向量在图像特征的引导下,生成新的单词特征向量
        14个样本,每个样本最多12个单词,每个单词是256维的向量
        图像特征通道数256,每个通道是17*17大小
        返回在图像特征的引导下生成的新的单词特征向量
        返回: context * contextT * query 
            (256,17*17) * (17*17,256) * (256,cap_len) => (256,cap_len)
    param {*} query(word): tensor(14,256,12)
    param {*} context(img_feat): tensor(14,256,17,17)
    param {*} gamma1 4.0,超参,用于放缩attn
    return {*} weightedContext: tensor(14, 256, 12),
    return {*} attn: tensor(14,12,17,17), 每个单词对每个像素点的attn值
    '''
    """
    query: batch x ndf x queryL
    context: batch x ndf x ih x iw (sourceL=ihxiw)
    mask: batch_size x sourceL
    """
    '''
        14个样本,每个样本最多12个单词,每个单词是256维的向量
        图像特征通道数256,每个通道是17*17大小
        返回值加权的内容,就是单词在像素点的256维向量的引导下,生成新的单词的向量
        query(word): tensor(14,256,12)
        context(img_feat): tensor(14,256,17,17),单词和文本
    '''
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

    # --> batch x sourceL x ndf
    context = context.view(batch_size, -1, sourceL) # ! tensor(14,256,17*17=289)
    contextT = torch.transpose(context, 1, 2).contiguous() # ! tensor(14,289,256)

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    attn = torch.bmm(contextT, query) # Eq. (7) in AttnGAN paper # ! tensor(14,289,12)
    # --> batch*sourceL x queryL
    attn = attn.view(batch_size*sourceL, queryL) # ! tensor(14*289,12)
    attn = nn.Softmax(dim=1)(attn)  # Eq. (8)

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL) # ! tensor(14,289,12)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous() # ! tensor(14,12,289)
    attn = attn.view(batch_size*queryL, sourceL) # ! tensor(14*12,289)
    #  Eq. (9)
    attn = attn * gamma1
    attn = nn.Softmax(dim=1)(attn)
    attn = attn.view(batch_size, queryL, sourceL) # ! tensor(14,12,289)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous() # ! tensor(14,289,12)

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT) # ! tensor(14,256,12)

    return weightedContext, attn.view(batch_size, -1, ih, iw) # ! tensor(14, 256,12), tensor(14,12,17,17)


def discriminator_loss(discriminator, words_emb, fake_imgs,
                        real_imgs, real_labels, fake_labels):
    real_uncond_result, fake_uncond_result, real_cond_result, fake_cond_result, wrong_cond_result = \
        discriminator(words_emb, fake_imgs, real_imgs)

    # print(real_uncond_result)
    # print(fake_uncond_result)
    # print(real_cond_result)
    # print(fake_cond_result)
    # print(wrong_cond_result)
    
    # real_uncond_result[real_uncond_result<0.0] = 0.0
    # real_uncond_result[real_uncond_result>1.0] = 1.0

    # fake_uncond_result[fake_uncond_result<0.0] = 0.0
    # fake_uncond_result[fake_uncond_result>1.0] = 1.0

    # real_cond_result[real_cond_result<0.0] = 0.0
    # real_cond_result[real_cond_result>1.0] = 1.0

    # fake_cond_result[fake_cond_result<0.0] = 0.0
    # fake_cond_result[fake_cond_result>1.0] = 1.0

    # wrong_cond_result[wrong_cond_result<0.0] = 0.0
    # wrong_cond_result[wrong_cond_result>1.0] = 1.0

    # print(real_uncond_result)
    # print(fake_uncond_result)
    # print(real_cond_result)
    # print(fake_cond_result)
    # print(wrong_cond_result)

    real_uncond_errD = nn.BCELoss()(real_uncond_result.squeeze(1), real_labels)
    
    fake_uncond_errD = nn.BCELoss()(fake_uncond_result.squeeze(1), fake_labels)

    real_cond_errD = nn.BCELoss()(real_cond_result.squeeze(1), real_labels)
    fake_cond_errD = nn.BCELoss()(fake_cond_result.squeeze(1), fake_labels)
    
    # wrong_cond_errD = nn.BCELoss()(wrong_cond_result.squeeze(1), fake_labels[1:])
    

    errD = ((real_uncond_errD + fake_uncond_errD) / 2. +
                (real_cond_errD + fake_cond_errD ) / 2.)
    return errD

def generator_loss(discriminator, img_encoder, imgs, fake_imgs, real_labels, words_emb, match_labels, cap_lens, class_ids):
    batch_size = real_labels.size(0)
    logs = ''
    errG_total = 0
    real_uncond_result, fake_uncond_result, real_cond_result, fake_cond_result, wrong_cond_result = \
        discriminator(imgs, fake_imgs, words_emb)
    
    cond_errG = nn.BCELoss()(fake_cond_result, real_labels)
    uncond_errG = nn.BCELoss()(fake_uncond_result, real_labels)
    errG_total = cond_errG + uncond_errG
    logs+= 'g_loss: %.2f ' % (errG_total.item())

    fake_output, real_output, txt_output = \
        img_encoder(fake_imgs, imgs, words_emb)

    w_loss0, w_loss1, _ = words_loss(real_output.transpose(1,2).contiguous(), txt_output.transpose(1,2).contiguous(), match_labels, cap_lens, class_ids, batch_size)
    real_errG = (w_loss0 + w_loss1) * cfg.TRAIN.SMOOTH.LAMBDA

    w_loss0, w_loss1, attn = words_loss(fake_output.transpose(1,2).contiguous(), txt_output.transpose(1,2).contiguous(), match_labels, cap_lens, class_ids, batch_size)
    fake_errG = (w_loss0 + w_loss1) * cfg.TRAIN.SMOOTH.LAMBDA


    errG_total += (real_errG + fake_errG)
    logs += 'real_loss: %.2f fake_loss: %.2f ' % (real_errG.item(), fake_errG.item())
    
    # attn =torch.cat(attn,0)
    return errG_total, logs, attn