'''
Description: 
version: 
Author: Yue Yang
Date: 2022-03-30 12:14:02
LastEditors: Yue Yang
LastEditTime: 2022-04-16 17:20:50
'''
from tkinter import W
from turtle import up
import torch
import torch.nn as nn 
from miscc.config import cfg 
from transformer.Models import Encoder, PositionalEncoding, words_pooling
from transformer.Layers import DecoderLayer, EncoderLayer
from transformer.Models import get_target_mask



def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)


class TEXT_ENCODER(nn.Module):
    def __init__(self,):
        super(TEXT_ENCODER, self).__init__()
        self.encoder = Encoder(n_src_vocab=cfg.TEXT.VOCAB_SIZE,
                                d_word_vec=cfg.TEXT.EMBEDDING_DIM,
                                n_layers=6,
                                n_head=8,
                                d_k=64,
                                d_v=64,
                                d_model=256,
                                d_inner=512,
                                pad_idx=0,
                                n_position=200
                                )
    
    def forward(self, sent_seq):
        '''
        description: 文本编码器，将文本序列编码成向量
        param {*} self
        param {*} sent_seq: [bs, seq_lens]
        return {*} [bs, max_len, d_model]
        '''
        words_emb = self.encoder(sent_seq)
        return words_emb


class IMG_FEAT_EXTRATOR(nn.Module):
    '''
    description: 从图像中提取特征，并返回图像特征
    param {*} 
    return {*} [bs, 16, ndf*8]
    '''
    def __init__(self, ndf):
        super(IMG_FEAT_EXTRATOR, self).__init__()
        # ndf = cfg.IMAGE.FEAT_EMB_DIM
        self.ndf = ndf
        self.encode_img = nn.Sequential(
            # --> state size. ndf x in_size/2 x in_size/2
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.GELU(),
            # --> state size 2ndf x x in_size/4 x in_size/4
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.GroupNorm(2, ndf * 2),
            nn.GELU(),
            # --> state size 4ndf x in_size/8 x in_size/8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.GroupNorm(4, ndf * 4),
            nn.GELU(),
            # --> state size 8ndf x in_size/16 x in_size/16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.GroupNorm(8, ndf * 8),
            nn.GELU()
        )
        
    def forward(self, imgs):
        '''
        description: 
        param {*} self
        param {*} imgs: [bs, 3, 64, 64]
        return {*} [bs, 16, ndf*8]
        '''
        # ! [bs, 3, 64, 64] => [bs, ndf*8, 4, 4]
        img_feat = self.encode_img(imgs)
        # ! => [bs, ndf*8, 16] => [bs, 16, ndf*8]
        img_feats = img_feat.view(img_feat.size(0), self.ndf*8,-1).transpose(1,2).contiguous()
        # ! => [bs, 16, ndf*8]
        # img_feat = img_feat.transpose(1,2)
        return img_feats


class FEAT_EXTRATOR(nn.Module):
    '''
    description: 从图像中提取特征，并返回图像特征
    param {*} 
    return {*} [bs, 16*16, ndf*8]
    '''
    def __init__(self, ndf):
        super(FEAT_EXTRATOR, self).__init__()
        # ndf = cfg.IMAGE.FEAT_EMB_DIM
        self.ndf = ndf
        self.encode_img = nn.Sequential(
            # --> state size. ndf x in_size/2 x in_size/2
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.GELU(),
            # --> state size 2ndf x x in_size/4 x in_size/4
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.GroupNorm(2, ndf * 2),
            nn.GELU(),
            # --> state size 4ndf x in_size/4 x in_size/4
            nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=False),
            nn.GroupNorm(4, ndf * 4),
            nn.GELU(),
            # --> state size 8ndf x in_size/4 x in_size/4
            nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=False),
            nn.GroupNorm(8, ndf * 8),
            nn.GELU()
        )
        
    def forward(self, imgs):
        '''
        description: 
        param {*} self
        param {*} imgs: [bs, 3, 64, 64]
        return {*} [bs, 16*16, ndf*8]
        '''
        # ! [bs, 3, 64, 64] => [bs, ndf*8, 16, 16]
        img_feat = self.encode_img(imgs)
        # ! => [bs, ndf*8, 16*16] => [bs, 16*16, ndf*8]
        img_feats = img_feat.view(img_feat.size(0), self.ndf*8,-1).transpose(1,2).contiguous()
        # ! => [bs, 16*16, ndf*8]
        # img_feat = img_feat.transpose(1,2)
        return img_feats


class IMAGE_DECODER(nn.Module):
    def __init__(self, dropout=0.1):
        super(IMAGE_DECODER, self).__init__()
        img_feat_len = 17
        self.ndf = cfg.IMAGE.FEAT_EMB_DIM
        d_model = self.ndf*8
        self.img_extrator = IMG_FEAT_EXTRATOR(self.ndf)
        self.position_enc = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        # ! [bs, 4*4+1, ndf*8]
        self.stage1 = StageBlock(d_model=self.ndf*8, 
                                    d_inner=512,
                                    n_head=8,
                                    d_k=64, 
                                    d_v=64,
                                    n_layers=4)
        # ! [bs, 16, ndf*8] => [bs, ndf//2, 64, 64]
        self.stage2 = INIT_STAGE_G()
        # ! [bs, ndf//2, 64, 64]
        self.stage3 = NEXT_STAGE_G(self.ndf//2,cfg.TEXT.EMBEDDING_DIM)
        
        self.img_generator = IMAGE_GENERATOR(self.ndf//2)    

    def pixel_upsample(self, x, H, W, scale=4):
        '''
        description: [bs, C, H, W] => [bs, C/scale^2, H*scale, W*scale], 通道数缩小 scale*scale 倍，图像长宽各放大 scale 倍
        param {*} x: [bs, H*W, C]
        param {*} H
        param {*} W
        return {*} [bs, H*scale*W*scale, C/scale/scale]
        '''
        
        B, N, C = x.size()
        assert N == H*W
        x = x.permute(0, 2, 1)
        x = x.view(-1, C, H, W)
        x = nn.PixelShuffle(scale)(x)
        B, C, H, W = x.size()
        x = x.view(-1, C, H*W)
        x = x.permute(0,2,1)
        return x

    def forward(self, imgs, word_emb, src_mask):
        '''
        description: 
        param {*} self
        param {*} imgs [bs, 16, ndf*8]
        param {*} word_emb [bs, cap_lens, txt_emb]
        param {*} src_mask [bs, cap_lens]
        return {*}
        '''

        ################################
        # # ! 从真实图像提取特征
        ################################
        # ! [bs, 3, 64, 64] => [bs, 16, ndf*8]
        img_feat = self.img_extrator(imgs)
        bs, _, feat_dim = img_feat.size()

        ################################
        # # ! 增加start位置，并进行位置编码
        ################################
        label_code0 = torch.zeros([bs,1,feat_dim]) # ! [bs, 1, ndf*8]
        if cfg.CUDA:
            label_code0 = label_code0.cuda()
        img_feat = torch.cat((label_code0,img_feat),1)# ! => [bs, 17, ndf*8]
        # ! [bs, 17, ndf*8]
        img_feat_input = self.layer_norm(self.dropout(self.position_enc(img_feat)))
        # img_feat_input = self.layer_norm(img_feat_input)

        ################################
        # # ! 第一阶段的Transformer decoder
        ################################
        # ! [bs, 17, ndf*8]
        target_mask = get_target_mask(img_feat_input.size(1),img_feat_input.device)
        stage1_code = self.stage1(img_feat_input, word_emb,target_mask, src_mask)
        # stage1_code = stage1_code[0]

        ################################
        # # ! 第二阶段 
        # # ! [bs, 16, ndf*8] => [bs, ndf//2, 64, 64]
        ################################
        
        stage2_code_in = stage1_code[0][:,:-1,:] # ! [bs, 16, ndf*8]
        stage2_code_out = self.stage2(stage2_code_in) # ! [bs, ndf//2, 64, 64]
        
        ################################
        # # ! 第三阶段 特征融合, 融合 3 次
        # # ! [bs, ndf//2, 64, 64]
        ################################
        stage3_code,attn = self.stage3(stage2_code_out, word_emb.transpose(1,2).contiguous(), src_mask)
        # stage3_code,_ = self.stage3(stage3_code, word_emb.transpose(1,2), src_mask)
        # stage3_code,attn = self.stage3(stage3_code, word_emb.transpose(1,2), src_mask)

        ################################
        # # ! 第三阶段 特征融合, 融合 3 次
        ################################
        fake_imgs = self.img_generator(stage3_code)

        return fake_imgs, attn



        # ################################
        # # # ! 1 => 2 的过渡阶段，进行维度变化
        # ################################
        # # ! => [bs, 16*16, ndf/2]
        # upsapmle1 = stage1_code[:,:-1,:]  # ! [bs, 16, ndf*8]
        # upsample1_code = self.pixel_upsample(upsapmle1,4,4) # ! [bs, 16*16, ndf/2]
        # label_code1 = stage1_code[:,-1,:] # ! [bs, 1, ndf*8]
        # # label_code1 = torch.zeros([upsapmle1.size(0),1,upsapmle1.size(-1)])
        # label_code1 = self.linear1(label_code1.squeeze(1)).unsqueeze(1) # ! [bs, 1, ndf*8] => # ! [bs, 1, ndf/2]
        # upsample1_code = torch.cat([label_code1,upsample1_code],1) # ! => [bs, 16*16+1, ndf/2]
        
        # ################################
        # # # ! 第二阶段的Transformer decoder
        # ################################
        # # ! => [bs, 16*16+1, ndf/2]
        # target_mask = get_target_mask(upsample1_code.size(1),upsample1_code.device)
        # stage2_code = self.stage2(upsample1_code,word_emb, target_mask, src_mask)
        # stage2_code = stage2_code[0]

        # ################################
        # # # ! 2 => 3 的过渡阶段，进行维度变化
        # ################################
        # # ! => [bs, 64*64+1, ndf/32]
        # upsample2 = stage2_code[:,:-1,:] # ! [bs,16*16,ndf/2]
        # upsample2_code = self.pixel_upsample(upsample2,16,16) # ! [bs,64*64,ndf/2]
        # label_code2 = stage2_code[:,-1,:] # ! [bs, 1, ndf/2]
        # label_code2 = self.linear2(label_code2.squeeze(1)).unsqueeze(1) # ! [bs, 1, ndf/2] => [bs, 1, ndf/32]
        # # label_code2 = torch.zeros([upsample2_code.size(0),1,upsample2_code.size(-1)])
        # upsample2_code = torch.cat([label_code2,upsample2_code],1)  # ! => [bs, 64*64+1, ndf/32]

        # ################################
        # # # ! 第三阶段的Transformer decoder
        # ################################
        # # ! [bs, 64*64+1, ndf/32]
        # target_mask = get_target_mask(upsample2_code.size(1),upsample2_code.device)
        # stage3_code,_, attn_list = self.stage3(upsample2_code,word_emb, target_mask, src_mask, return_attns=True)
        
        # ################################
        # # # ! 输出进行转置以及变形
        # ################################
        # stage3_code = stage3_code.transpose().view(-1,self.ndf/32,64,64)
        # fake_imgs = self.img_generator(stage3_code)
        # # ! 输出的值的范围在 [-1,1] 之间
        # # ! [bs, 3, 64, 64]
        # return fake_imgs, attn_list[-1]


class IMAGE_GENERATOR(nn.Module):
    def __init__(self, ndf):
        super(IMAGE_GENERATOR, self).__init__()
        # ndf = cfg.IMAGE.FEAT_EMB_DIM
        self.img = nn.Sequential(
            conv3x3(ndf, 3),
            nn.Tanh()
        )

    def forward(self, img_feat):
        # ! [bs, ndf/32, 64, 64] => [bs, 3, 64, 64]
        out_img = self.img(img_feat)
        return out_img

class StageBlock(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, n_layers, dropout=0.1):
        super(StageBlock, self).__init__()
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
            
    def forward(self, trg_seq, enc_output, trg_mask, src_mask, return_attns=False):
        dec_slf_attn_list, dec_enc_attn_list = [], []
        dec_output = trg_seq
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []
            if return_attns:
                return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class DISCRIMINATOR(nn.Module):
    def __init__(self, d_model=256, dropout=0.1):
        super(DISCRIMINATOR, self).__init__()
        img_feat_len = 1+16*16*2+cfg.TEXT.WORDS_NUM 
        self.ndf = cfg.IMAGE.FEAT_EMB_DIM
        assert d_model == self.ndf*8
        n_layers=10
        self.img_extrator = FEAT_EXTRATOR(self.ndf)
        self.position_enc = PositionalEncoding(self.ndf * 8)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.uncond = nn.ModuleList([
            EncoderLayer(d_model, d_inner=512, n_head=8, d_k=64, d_v=64, dropout=dropout)
            for _ in range(n_layers)])
        self.cond = nn.ModuleList([
            EncoderLayer(d_model, d_inner=512, n_head=8, d_k=64, d_v=64, dropout=dropout)
            for _ in range(n_layers)])
        # self.cond = nn.Sequential(*self.cond_layers)
        self.img_dis = nn.Sequential(
            nn.Linear(self.ndf*8,1),
            nn.Sigmoid()
        )

    def forward(self, real_imgs, fake_imgs, words_emb):
        '''
        description: 
        param {*} self
        param {*} real_imgs: [bs, 3, 64, 64]
        param {*} fake_imgs: [bs, 3, 64, 64]
        param {*} words: [bs, max_len, d_model]
        return {*}  real_uncond_result: [bs,1] in (0,1)
                    fake_uncond_result: [bs,1] in (0,1)
                    cond_result: [bs,1] in (0,1)
                    real_output: [bs, 16*16, ndf*8]
                    fake_output: [bs, 16*16, ndf*8]
                    txt_output: [bs, cap_lens, ndf*8]
        # '''
        real_imgs_feat = self.img_extrator(real_imgs.detach()).transpose(1,2).contiguous() # ! [bs,ndf*8,16*16]  
        fake_imgs_feat = self.img_extrator(fake_imgs).transpose(1,2).contiguous() # ! [bs,ndf*8,16*16]
        label_code = torch.zeros([real_imgs_feat.size(0),1,real_imgs_feat.size(-1)]) # ! [bs,1,ndf*8]
        if cfg.CUDA:
            label_code = label_code.cuda()
        # # sent_emb = words_pooling(words,words_mask)
        real_uncond_input = torch.cat((label_code,real_imgs_feat),1) # ! [bs,1+16*16,ndf*8]
        fake_uncond_input = torch.cat((label_code,fake_imgs_feat),1) # ! [bs,1+16*16,ndf*8]
        real_uncond_input = self.layer_norm(self.dropout(self.position_enc(real_uncond_input))) # ! [bs,1+16*16,ndf*8]
        fake_uncond_input = self.layer_norm(self.dropout(self.position_enc(fake_uncond_input))) # ! [bs,1+16*16,ndf*8]
        
        real_uncond_output = real_uncond_input
        fake_uncond_output = fake_uncond_input
        # real_uncond_output,_ = self.uncond[0](real_uncond_input)
        # fake_uncond_output,_ = self.uncond[0](fake_uncond_input)
        for net in self.uncond:
            real_uncond_output,_ = net(real_uncond_output)
            fake_uncond_output,_ = net(fake_uncond_output)
        # real_uncond_output = self.cond(real_uncond_input)
        # fake_uncond_output = self.cond(fake_uncond_input)

        real_cond_input = torch.cat((label_code,real_imgs_feat,words_emb),1) # ! [bs,1+16*16+cap_len,ndf*8]
        fake_cond_input = torch.cat((label_code,fake_imgs_feat,words_emb),1) # ! [bs,1+16*16+cap_len,ndf*8]
        real_cond_input = self.layer_norm(self.dropout(self.position_enc(real_cond_input))) # ! [bs,1+16*16+cap_len,ndf*8]
        fake_cond_input = self.layer_norm(self.dropout(self.position_enc(fake_cond_input))) # ! [bs,1+16*16+cap_len,ndf*8]

        real_cond_output = real_cond_input
        fake_cond_output= fake_cond_input
        for net in self.cond:
            real_cond_output,_ = net(real_cond_output)
            fake_cond_output,_ = net(fake_cond_output)
        # real_cond_output = self.cond(real_cond_input)
        # fake_cond_output = self.cond(fake_cond_input)

        wrong_cond_input = torch.cat((label_code[1:],real_imgs_feat[:-1],words_emb[1:]),1) # ! [bs-1,1+16*16+cap_len,ndf*8]
        wrong_cond_input = self.layer_norm(self.dropout(self.position_enc(wrong_cond_input))) # ! [bs,1+16*16+cap_len,ndf*8]
        # wrong_cond_output = self.cond(wrong_cond_input)
        wrong_cond_output = wrong_cond_input
        for net in self.cond:
            wrong_cond_output,_ = net(wrong_cond_output)

        # ! [bs, 1] in (0,1)
        real_uncond_result = self.img_dis(real_uncond_output[:,0,:].squeeze(1))
        fake_uncond_result = self.img_dis(fake_uncond_output[:,0,:].squeeze(1))
        real_cond_result = self.img_dis(real_cond_output[:,0,:].squeeze(1))
        fake_cond_result = self.img_dis(fake_cond_output[:,0,:].squeeze(1))
        wrong_cond_result = self.img_dis(wrong_cond_output[:,0,:].squeeze(1))
        # print(real_uncond_result)
        # print(fake_uncond_result)
        # print(real_cond_result)
        # print(fake_cond_result)
        # print(wrong_cond_result)
        
        return real_uncond_result, fake_uncond_result, real_cond_result, fake_cond_result, wrong_cond_result
        # input_code = torch.cat([label_code,real_imgs_feat,fake_imgs_feat,words_emb]) #![bs,1+16*16*2+max_len, d_model]

        # enc_input = self.layer_norm(self.dropout(self.position_enc(input_code)))
        # enc_output = self.encoder(enc_input)
        # uncond_output, real_output, fake_output, txt_output = enc_output[:,0,:], enc_output[:,1:1+16,:], enc_output[:,1+16:1+32,:], enc_output[:,1+32:,:]
        # cond_result = self.img_dis(uncond_output.squeeze(1))
        # real_uncond_result = self.img_dis(real_img_output[:,0,:].squeeze(1))
        # fake_uncond_result = self.img_dis(fake_img_output[:,0,:].squeeze(1))
        # # ! [bs,1] in (0,1), [bs,1] in (0,1), [bs,1] in (0,1), [bs, 16*16, ndf*8], [bs, 16*16, ndf*8], [bs, cap_lens, ndf*8]
        # return real_uncond_result, fake_uncond_result, cond_result, real_output, fake_output, txt_output

class IMAGE_ENCODER(nn.Module):
    def __init__(self, d_model=256, dropout=0.1):
        super(IMAGE_ENCODER, self).__init__()
        img_feat_len = 1+16*16*2+cfg.TEXT.WORDS_NUM 
        self.ndf = cfg.IMAGE.FEAT_EMB_DIM
        assert d_model == self.ndf*8
        n_layers = 2
        self.img_extrator = FEAT_EXTRATOR(self.ndf)
        self.position_enc = PositionalEncoding(self.ndf * 8)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.enc = nn.ModuleList([
            EncoderLayer(d_model, d_inner=512, n_head=8, d_k=64, d_v=64, dropout=dropout)
            for _ in range(n_layers)])

        self.img_dis = nn.Sequential(
            nn.Linear(self.ndf*8,1),
            nn.Sigmoid()
        )

    def forward(self, real_imgs, fake_imgs, words_emb):
        '''
        description: 
        param {*} self
        param {*} real_imgs: [bs, 3, 64, 64]
        param {*} fake_imgs: [bs, 3, 64, 64]
        param {*} words: [bs, max_len, d_model]
        return {*}  real_uncond_result: [bs,1] in (0,1)
                    fake_uncond_result: [bs,1] in (0,1)
                    cond_result: [bs,1] in (0,1)
                    real_output: [bs, 16*16, ndf*8]
                    fake_output: [bs, 16*16, ndf*8]
                    txt_output: [bs, cap_lens, ndf*8]
        # '''

        real_imgs_feat = self.img_extrator(real_imgs).transpose(1,2).contiguous() # ! [bs,ndf*8,16*16] 
        fake_imgs_feat = self.img_extrator(fake_imgs).transpose(1,2).contiguous() # ! [bs,ndf*8,16*16]

        enc_input = torch.cat((fake_imgs_feat, real_imgs_feat, words_emb),1) # ! [bs,16*16*2+cap_len,ndf*8]
        enc_input = self.layer_norm(self.dropout(self.position_enc(enc_input))) # ! [bs,16*16*2+cap_len,ndf*8]
        
        enc_output = enc_input
        for net in self.enc:
            enc_output,_ = net(enc_output)

        fake_output, real_output, txt_output = enc_output[:,:16*16,:], enc_output[:,16*16:16*16*2,:], enc_output[:,16*16*2:,:]
        return fake_output, real_output, txt_output


class NEXT_STAGE_G(nn.Module):
    def __init__(self, ngf, nef):
        '''
        description: 
        param {*} self
        param {*} ngf: 图像的维度, ndf//2
        param {*} nef: 文本的维度
        return {*}
        '''
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.ef_dim = nef
        self.num_residual = cfg.MODEL.R_NUM
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(cfg.MODEL.R_NUM):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        self.att = GlobalAttentionGeneral(ngf, self.ef_dim)
        self.residual = self._make_layer(ResBlock, ngf * 2)
        self.upsample = upBlock(ngf * 2, ngf)
        self.block = nn.Sequential(
            conv3x3(ngf * 2, ngf*2),
            nn.BatchNorm2d(ngf*2),
            GLU())

    def forward(self, h_code, word_embs, mask):
        '''
        description: 
        param {*} self
        param {*} h_code: img_feat, [bs, ndf//2, 64, 64]
        param {*} word_embs: [bs, emb_dim,cap_len]
        param {*} mask: [bs, cap_len]
        return {*} out_code: [bs, ndf//2, 64, 64]
                    att: [bs, cap_len, ih x iw]
        '''
        """
            h_code1(query):  batch x idf x ih x iw (queryL=ihxiw)
            word_embs(context): batch x cdf x sourceL (sourceL=seq_len)
            c_code1: batch x idf x queryL
            att1: batch x sourceL x queryL
        """
        self.att.applyMask(mask)
        c_code, att = self.att(h_code, word_embs)
        h_c_code = torch.cat((h_code, c_code), 1)
        out_code = self.residual(h_c_code)
        out_code = self.block(out_code)
        # # state size ngf/2 x 2in_size x 2in_size
        # out_code = self.upsample(out_code)

        return out_code, att


class GlobalAttentionGeneral(nn.Module):
    def __init__(self, idf, cdf):
        super(GlobalAttentionGeneral, self).__init__()
        self.conv_context = conv1x1(cdf, idf)
        self.sm = nn.Softmax(dim=1)
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask.squeeze(1)  # batch x sourceL

    def forward(self, input, context):
        """
            input: batch x idf x ih x iw (queryL=ihxiw)
            context: batch x cdf x sourceL
        """
        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context.size(0), context.size(2)

        # --> batch x queryL x idf
        target = input.view(batch_size, -1, queryL)
        targetT = torch.transpose(target, 1, 2).contiguous()
        # batch x cdf x sourceL --> batch x cdf x sourceL x 1
        sourceT = context.unsqueeze(3)
        # --> batch x idf x sourceL
        sourceT = self.conv_context(sourceT).squeeze(3)

        # Get attention
        # (batch x queryL x idf)(batch x idf x sourceL)
        # -->batch x queryL x sourceL
        attn = torch.bmm(targetT, sourceT)
        # --> batch*queryL x sourceL
        attn = attn.view(batch_size*queryL, sourceL)
        if self.mask is not None:
            # batch_size x sourceL --> batch_size*queryL x sourceL
            mask = self.mask.repeat(queryL, 1)
            attn.data.masked_fill_(mask.data==False, -float('inf'))
        attn = self.sm(attn)  # Eq. (2)
        # --> batch x queryL x sourceL
        attn = attn.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        attn = torch.transpose(attn, 1, 2).contiguous()

        # (batch x idf x sourceL)(batch x sourceL x queryL)
        # --> batch x idf x queryL
        weightedContext = torch.bmm(sourceT, attn)
        weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        attn = attn.view(batch_size, -1, ih, iw)

        return weightedContext, attn

def conv1x1(in_planes, out_planes):
    '''
    name: 
    test: test font
    msg: 
    param {*}
    return {*}
    '''    
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=False)


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num))

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, size=None):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.size = size

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, size=self.size)
        return x

# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        Interpolate(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU())
    return block


class INIT_STAGE_G(nn.Module):
    def __init__(self):
        super(INIT_STAGE_G, self).__init__()
        
        self.ndf = cfg.IMAGE.FEAT_EMB_DIM
        self.gf_dim = self.ndf*8

        self.define_module()

    def define_module(self):
        ngf = self.gf_dim
        
        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)

    def forward(self, img_feat):
        '''
        description: 
        param {*} self
        param {*} img_feat: [bs, 16, ndf*8]
        return {*} [bs, ndf//2, 64, 64]
        '''
        """
        :param z_code: batch x cfg.GAN.Z_DIM
        :param c_code: batch x cfg.TEXT.EMBEDDING_DIM
        :return: batch x ngf/16 x 64 x 64
        """
        
        # state size ngf x 4 x 4
        # ! [bs, 16, ndf*8] => [bs, ndf*8, 16] => [bs, ndf*8, 4, 4]
        out_code = img_feat.transpose(1,2).contiguous().view(-1,self.ndf*8,4,4) 
        # state size ngf/3 x 8 x 8
        # ! => [bs, ndf*4, 8, 8]
        out_code = self.upsample1(out_code)
        # state size ngf/4 x 16 x 16
        # ! => [bs, ndf*2, 16, 16]
        out_code = self.upsample2(out_code)
        # state size ngf/8 x 32 x 32
        # ! => [bs, ndf, 32， 32]
        out_code32 = self.upsample3(out_code)
        # state size ngf/16 x 64 x 64
        # ! => [bs, ndf//2, 64, 64]
        out_code64 = self.upsample4(out_code32)

        return out_code64
