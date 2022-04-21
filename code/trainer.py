'''
Description: 
version: 
Author: Yue Yang
Date: 2022-03-30 09:12:50
LastEditors: Yue Yang
LastEditTime: 2022-04-21 13:00:32
'''
from cgitb import text
from dis import dis
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import itertools
from miscc.config import cfg
from miscc.utils import mkdir_p, weights_init, build_super_images, copy_G_params, load_params
from miscc.losses import words_loss, discriminator_loss, generator_loss
import os
import time
from datasets import prepare_data

from model import TEXT_ENCODER, IMAGE_DECODER, DISCRIMINATOR, IMAGE_ENCODER
# from transformer.Models import get_pad_mask
def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)

class condGANTrainer(nn.Module):
    def __init__(self, output_dir, data_loader, n_words, ixtoword):
        super(condGANTrainer, self).__init__()
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)

        torch.cuda.set_device(cfg.GPU_ID)
        # ! cudnn.benchmark = True: 使 cuDNN 对多个卷积算法进行基准测试并选择最快的。
        cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.n_words = n_words
        self.ixtoword = ixtoword
        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)

    def build_models(self):
        img_ndf = cfg.IMAGE.FEAT_EMB_DIM
        text_encoder = TEXT_ENCODER()
        img_decoder = IMAGE_DECODER()
        img_encoder = IMAGE_ENCODER()
        discriminator = DISCRIMINATOR()
        # ! 加载预训练模型
        epoch = 0
        if(cfg.MODEL.TEXT_ENCODER):
            model_path = cfg.MODEL.TEXT_ENCODER
            state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.MODEL.TEXT_ENCODER)

            istart = cfg.MODEL.TEXT_ENCODER.rfind('_') + 1
            iend = cfg.MODEL.TEXT_ENCODER.rfind('.')
            epoch = cfg.MODEL.TEXT_ENCODER[istart:iend]
            epoch = int(epoch) + 1

            model_path = cfg.MODEL.IMAGE_DECODER
            state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
            img_decoder.load_state_dict(state_dict)
            print('Load image decoder from:', cfg.MODEL.IMAGE_DECODER)

            model_path = cfg.MODEL.IMAGE_ENCODER
            state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
            img_encoder.load_state_dict(state_dict)
            print('Load image encoder from:', cfg.MODEL.IMAGE_ENCODER)

            model_path = cfg.MODEL.DISCRIMINATOR
            state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
            discriminator.load_state_dict(state_dict)
            print('Load discriminator from:', cfg.MODEL.DISCRIMINATOR)
        else: # ! 初始化模型参数
            text_encoder.apply(weights_init)
            img_decoder.apply(weights_init)
            img_encoder.apply(weights_init)
            discriminator.apply(weights_init)
        
        if cfg.CUDA:
            text_encoder = text_encoder.cuda()
            img_decoder = img_decoder.cuda()
            img_encoder = img_encoder.cuda()
            discriminator = discriminator.cuda()

        return [text_encoder, img_decoder, img_encoder, discriminator, epoch]
        
    def define_optimizer(self, text_encoder, img_decoder, img_encoder, discriminator):
        optimizerG = optim.Adam(itertools.chain(text_encoder.parameters(), img_decoder.parameters(), img_encoder.parameters()),
        lr=cfg.TRAIN.GENERATOR_LR,
        betas=(0.5,0.999))
        
        optimizerD = optim.Adam(discriminator.parameters(),
        lr=cfg.TRAIN.DISCRIMINATOR_LR,
        betas=(0.5,0.999))
        return [optimizerG, optimizerD]

    def prepare_labels(self):
        batch_size = self.batch_size
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        match_labels = Variable(torch.LongTensor(range(batch_size)))
        if cfg.CUDA:
            real_labels = real_labels.cuda()
            fake_labels = fake_labels.cuda()
            match_labels = match_labels.cuda()

        return real_labels, fake_labels, match_labels

    def save_model(self, text_encoder, avg_param_text_encoder, img_decoder, avg_param_img_decoder, img_encoder, avg_param_img_encoder, discriminator, epoch):
        backup_text_encoder = copy_G_params(text_encoder)
        load_params(text_encoder, avg_param_text_encoder)
        torch.save(text_encoder.state_dict(),
            '%s/text_encoder_epoch_%d.pth' % (self.model_dir, epoch))
        load_params(text_encoder, backup_text_encoder)

        backup_img_decoder = copy_G_params(img_decoder)        
        load_params(img_decoder, avg_param_img_decoder)
        torch.save(img_decoder.state_dict(),
            '%s/img_decoder_epoch_%d.pth' % (self.model_dir, epoch))
        load_params(img_decoder, backup_img_decoder)

        backup_img_encoder = copy_G_params(img_encoder)        
        load_params(img_encoder, avg_param_img_encoder)
        torch.save(img_encoder.state_dict(),
            '%s/img_encoder_epoch_%d.pth' % (self.model_dir, epoch))
        load_params(img_encoder, backup_img_encoder)
        
        torch.save(discriminator.state_dict(),
            '%s/discriminator_epoch_%d.pth' % (self.model_dir, epoch))
            
        print('Save G/D models.')

    def save_img_results(self, captions, cap_lens, class_ids, imgs, text_encoder, img_encoder, img_decoder, discriminator, gen_iterations, real_labels, match_labels, name='current'):
        words_embs = text_encoder(captions)
        words_emb = words_embs[0]

        # mask = (captions == 0)
        # num_words = words_emb.size(2)
        # if mask.size(1) > num_words:
        #     mask = mask[:,:num_words]
        mask = get_pad_mask(captions,pad_idx=0)
        fake_imgs, attn_map = img_decoder(imgs,words_emb, mask)
        att_sze = attn_map.size(2)
        img_set,_ = build_super_images(fake_imgs.detach().cpu(), captions,self.ixtoword, attn_map, att_sze)
        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/G_%s_%d.png'\
                % (self.image_dir, name, gen_iterations)
            im.save(fullpath)
        real_uncond_result, fake_uncond_result, real_cond_result, fake_cond_result, wrong_cond_result \
            = discriminator(fake_imgs, imgs, words_emb)
        _, _, att_map = generator_loss(discriminator, img_encoder, imgs, fake_imgs, real_labels, words_emb, match_labels, cap_lens, class_ids)
        # _, _, att_map = words_loss(fake_output.detach().cpu(),txt_output.detach().cpu(),None, cap_lens,None,self.batch_size)
        att_sze = 16
        img_set,_ = build_super_images(fake_imgs.detach().cpu(), captions,self.ixtoword, att_map, att_sze)
        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/D_%s_%d.png'\
                % (self.image_dir, name, gen_iterations)
            im.save(fullpath)

    def set_requires_grad_value(self, models_list, brequires):
        for i in range(len(models_list)):
            for p in models_list[i].parameters():
                p.requires_grad = brequires

    def train(self):
        text_encoder, img_decoder, img_encoder, discriminator, start_epoch = self.build_models()

        avg_param_text_encoder = copy_G_params(text_encoder)
        avg_param_img_decoder = copy_G_params(img_decoder)
        avg_param_img_encoder = copy_G_params(img_encoder)

        optimizerG, optimizerD = self.define_optimizer(text_encoder, img_decoder, img_encoder, discriminator)
        # ! labels:tensor(bs)
        # ! real_labels:全1     fake_labels:全0     match_labels:0到bs-1
        real_labels, fake_labels, match_labels = self.prepare_labels()

        gen_iterations = 0
        for epoch in range(start_epoch, self.max_epoch+1):
            start_t = time.time()

            data_iter = iter(self.data_loader)
            step = 0
            while step < self.num_batches:
                self.set_requires_grad_value([discriminator], True)
                self.set_requires_grad_value([text_encoder, img_encoder, img_decoder], False)
                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                data = data_iter.next()
                imgs, captions, cap_lens, class_ids, keys = prepare_data(data)
                words_emb = text_encoder(captions)
                words_embs = words_emb[0]
                # mask = (captions == 0)
                # num_words = words_emb.size(2)
                # if mask.size(1) > num_words:
                #     mask = mask[:,:num_words]
                mask = get_pad_mask(captions,pad_idx=0)
                #######################################################
                # (2) Generate fake images
                ######################################################
                fake_imgs,_ = img_decoder(imgs, words_embs, mask)
                
                #######################################################
                # (3) Update D network
                ######################################################
                errD = 0
                D_logs = ''
                discriminator.zero_grad()
                errD = discriminator_loss(discriminator, imgs, fake_imgs, words_embs, real_labels, fake_labels)
                errD.backward()
                optimizerD.step()
                D_logs += 'errD: %.2f ' % (errD.item())

                #######################################################
                # (4) Update G network: maximize log(D(G(z)))
                ######################################################
                step += 1
                gen_iterations += 1

                self.set_requires_grad_value([discriminator], False)
                self.set_requires_grad_value([text_encoder, img_encoder, img_decoder], True)
            
                text_encoder.zero_grad()
                img_decoder.zero_grad()
                img_encoder.zero_grad()

                errG_total, G_logs,_ = \
                    generator_loss(discriminator, img_encoder,imgs, fake_imgs, real_labels, words_embs, match_labels, cap_lens, class_ids)
                # backward and update parameters
                errG_total.backward(retain_graph=True)
                optimizerG.step()
                
                for p, avg_p in zip(text_encoder.parameters(), avg_param_text_encoder):
                    avg_p.mul_(0.999).add_(0.001, p.data)
                for p, avg_p in zip(img_decoder.parameters(), avg_param_img_decoder):
                    avg_p.mul_(0.999).add_(0.001, p.data)   
                for p, avg_p in zip(img_encoder.parameters(), avg_param_img_encoder):
                    avg_p.mul_(0.999).add_(0.001, p.data)  

                if gen_iterations % 100 == 0:
                    print(D_logs + '\n' + G_logs)
                
                # save images
                if gen_iterations % 1000 == 0:
                    backup_para_text_encoder = copy_G_params(text_encoder)
                    load_params(text_encoder, avg_param_text_encoder)
                    backup_para_img_decoder = copy_G_params(img_decoder)
                    load_params(img_decoder, avg_param_img_decoder)

                    # captions, cap_lens, class_ids, imgs, text_encoder, img_encoder, img_decoder, discriminator, gen_iterations, real_labels, match_labels,
                    self.save_img_results(captions, cap_lens, class_ids, imgs, text_encoder, img_encoder, img_decoder, discriminator, epoch, real_labels, match_labels, name='average')

                    load_params(text_encoder, backup_para_text_encoder)
                    load_params(img_decoder, backup_para_img_decoder)
            end_t = time.time()

            print('''[%d/%d][%d]
                  Loss_D: %.2f Loss_G: %.2f Time: %.2fs'''
                  % (epoch, self.max_epoch, self.num_batches,
                     errD.item(), errG_total.item(),
                     end_t - start_t))
            
            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:  # and epoch != 0:
                # text_encoder, avg_param_text_encoder, img_decoder, avg_param_img_decoder, discriminator, epoch
                self.save_model(text_encoder, avg_param_text_encoder, img_decoder, avg_param_img_decoder, img_encoder, avg_param_img_encoder, discriminator, epoch)
            
        self.save_model(text_encoder, avg_param_text_encoder, img_decoder, avg_param_img_decoder, img_encoder, avg_param_img_encoder, discriminator, self.max_epoch)

