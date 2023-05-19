# Copyright (C) 2018 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the MIT License. To view a copy of this license,
# visit https://opensource.org/licenses/MIT.

"""AttGAN, generator, and discriminator."""

import torch
import torch.nn as nn
from nn import LinearBlock, Conv2dBlock, ConvTranspose2dBlock
from torchsummary import summary
import numpy as np
import torchsnooper
from torchvision import models
from loss import StyleLoss, PerceptualLoss
# This architecture is for images of 128x128
# In the original AttGAN, slim.conv2d uses padding 'same'
MAX_DIM = 64 * 16  # 1024
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, enc_dim=64, enc_layers=5, enc_norm_fn='batchnorm', enc_acti_fn='lrelu',
                 dec_dim=64, dec_layers=5, dec_norm_fn='batchnorm', dec_acti_fn='relu',
                 n_attrs=4, shortcut_layers=1, inject_layers=0, img_size=128):
        super(Generator, self).__init__()
        self.shortcut_layers = min(shortcut_layers, dec_layers - 1)
        self.inject_layers = min(inject_layers, dec_layers - 1)
        self.f_size = img_size // 2**enc_layers  # f_size = 4 for 128x128
        
        layers = []
        n_in = 3
        for i in range(enc_layers):
            n_out = min(enc_dim * 2**i, MAX_DIM)
            layers += [Conv2dBlock(
                n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=enc_norm_fn, acti_fn=enc_acti_fn
            )]
            n_in = n_out
        self.enc_layers = nn.ModuleList(layers)
        layers = []
        n_in = 448 #n_attrs  # 1024 + 13
        for i in range(dec_layers):
            if i < dec_layers - 1:
                n_out = min(dec_dim * 2**(dec_layers-i-1), MAX_DIM)
                layers += [ConvTranspose2dBlock(
                    n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=dec_norm_fn, acti_fn=dec_acti_fn
                )]
                n_in = n_out
                n_in = n_in + n_in//2 if self.shortcut_layers > i else n_in
                #n_in = n_in + n_attrs if self.inject_layers > i else n_in
            else:
                layers += [ConvTranspose2dBlock(
                    n_in, 3, (4, 4), stride=2, padding=1, norm_fn='none', acti_fn='tanh'
                )]
        self.dec_layers = nn.ModuleList(layers)
        self.uplayer = torch.nn.UpsamplingBilinear2d(scale_factor=16)
        self.uplayer8 = torch.nn.UpsamplingBilinear2d(scale_factor=8)
        self.uplayer4 = torch.nn.UpsamplingBilinear2d(scale_factor=4)
        self.uplayer1 = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        self.uplayer2 = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_mask = Conv2dBlock(
                64, 1, (3, 3), stride=1, padding=1, norm_fn='none', acti_fn='sigmoid'
            )
        self.conv_mask1 = Conv2dBlock(
                256+1, 1, (3, 3), stride=1, padding=1, norm_fn='none', acti_fn='sigmoid'
            )
        self.conv_mask2 = Conv2dBlock(
                128+1, 1, (3, 3), stride=1, padding=1, norm_fn='none', acti_fn='sigmoid'
            )
        self.conv_mask3 = Conv2dBlock(
                64+1, 1, (3, 3), stride=1, padding=1, norm_fn='none', acti_fn='sigmoid'
            )

    def encode(self, x):
        z = x
        zs = []
        for layer in self.enc_layers:
            z = layer(z)
            zs.append(z)
        z1, z2  = torch.split(z, 512-64, dim=1)
        m_z = self.conv_mask(z2)
        mas = self.uplayer(m_z)
        return zs, m_z, mas

    def classify(self, zs, a):
        box = []
        for i in range(a.size(1)):
            num_chs = zs.size(1)
            per_chs = float(num_chs)
            start = int(np.rint(per_chs * i))
            end = int(np.rint(per_chs * (i + 1)))
            temp = zs.narrow(1, start, end-start)
            av = a.view(a.size(0), -1, 1, 1)
            ai =av[:,i,:,:]
            ai = torch.unsqueeze(ai, 1)
            tar_i =ai.repeat(1, 1,  zs.size(2), zs.size(2))
            box.append(torch.mul(tar_i, zs))
            break
        re = torch.cat(box, dim=1)	
        return re

    #@torchsnooper.snoop()
    def decode(self, zs, m_z, att):
        ms = []
        ms1 = []
        z1, _  = torch.split(zs[-1], 512-64, dim=1)
        z = z1*(1-m_z)
        ms1.append(self.uplayer(m_z))

        m_z1 = self.uplayer1(m_z)
        z_temp = torch.cat([zs[-2], m_z1], dim=1)
        m_z1 = self.conv_mask1(z_temp)
        m_z1 = self.classify(m_z1, att)
        ms.append(m_z1)
        ms1.append(self.uplayer8(m_z1))

        m_z2 = self.uplayer2(m_z1)
        z_temp = torch.cat([zs[-3], m_z2], dim=1)
        m_z2 = self.conv_mask2(z_temp)
        m_z2 = self.classify(m_z2, att)
        ms.append(m_z2)
        ms1.append(self.uplayer4(m_z2))

        m_z3 = self.uplayer2(m_z2)
        z_temp = torch.cat([zs[-4], m_z3], dim=1)
        m_z3 = self.conv_mask3(z_temp)
        m_z3 = self.classify(m_z3, att)
        ms.append(m_z3)
        ms1.append(self.uplayer1(m_z3))

        m_z4 = self.uplayer2(m_z3)
        m_z4 = self.uplayer(m_z)  #self.classify(m_z4, att)
        #z_temp = torch.cat([zs[-5], m_z4], dim=1)
        #m_z4 = self.conv_mask4(z_temp)
        ms.append(m_z4)
        ms1.append(m_z4)
        #a_tile = a.view(a.size(0), -1, 1, 1).repeat(1, 1, self.f_size, self.f_size)
        #z = zs[-1]#torch.cat([zs[-1], a_tile], dim=1)			   
            
        for i, layer in enumerate(self.dec_layers):
            z = layer(z)
            if self.shortcut_layers > i:  # Concat 1024 with 512
                z_att = z*ms[i]
                z_rest = zs[-i-2]*(1-ms[i])
                z = torch.cat([z, z_rest], dim=1)
        return z, ms, ms1, m_z4

    #@torchsnooper.snoop()
    def decode1(self, zs, zs1, m_z, r_z, ms):
        
        z1, _  = torch.split(zs[-1], 512-64, dim=1)
        z11, _  = torch.split(zs1[-1], 512-64, dim=1)
        z = z1*(1- m_z-r_z) + z11*(r_z+m_z)
            
        for i, layer in enumerate(self.dec_layers):
            z = layer(z)
            if self.shortcut_layers > i:  # Concat 1024 with 512
                z_att = z*ms[i]
                z_rest = zs[-i-2]*(1-ms[i])+ zs1[-i-2]*ms[i]
                z = torch.cat([z, z_rest], dim=1)
        return z

    def decode2(self, zs1, zs, r_z, m_z, att):
        ms = []
        
        z1, _  = torch.split(zs[-1], 512-64, dim=1)
        z11, _  = torch.split(zs1[-1], 512-64, dim=1)
        z = z11*(1-r_z) + z1*r_z

        m_z1 = self.uplayer1(r_z)
        z_temp = torch.cat([zs[-2], m_z1], dim=1)
        m_z1 = self.conv_mask1(z_temp)
        ms.append(m_z1)

        m_z2 = self.uplayer2(m_z1)
        z_temp = torch.cat([zs[-3], m_z2], dim=1)
        m_z2 = self.conv_mask2(z_temp)
        ms.append(m_z2)

        m_z3 = self.uplayer2(m_z2)
        z_temp = torch.cat([zs[-4], m_z3], dim=1)
        m_z3 = self.conv_mask3(z_temp)
        ms.append(m_z3)

        m_z4 = self.uplayer2(m_z3)
        #z_temp = torch.cat([zs[-5], m_z4], dim=1)
        #m_z4 = self.conv_mask4(z_temp)
        ms.append(m_z4)			   
            
        for i, layer in enumerate(self.dec_layers):
            z = layer(z)
            if self.shortcut_layers > i:  # Concat 1024 with 512
                z_att = z*ms[i]
                z_rest = zs[-i-2]*(1-ms[i])+ zs1[-i-2]*ms[i]
                z = torch.cat([z, z_rest], dim=1)
        return z
    
    def forward(self, x, a=None, b=None, c=None, d=None, mode='enc'):
        if mode == 'enc':
            return self.encode(x)
        if mode == 'dec':
            #assert a is not None, 'No given attribute.'
            z = self.decode(x, a, b)
            return z
        if mode =='dec1':
            z = self.decode1(x, a, b, c, d)
            return z
        raise Exception('Unrecognized mode: ' + mode)

class Discriminators(nn.Module):
    # No instancenorm in fcs in source code, which is different from paper.
    def __init__(self, dim=64, norm_fn='instancenorm', acti_fn='lrelu',
                 fc_dim=1024, fc_norm_fn='none', fc_acti_fn='lrelu', n_layers=5, img_size=128):
        super(Discriminators, self).__init__()
        self.f_size = img_size // 2**n_layers
        
        layers = []
        n_in = 3
        for i in range(n_layers):
            n_out = min(dim * 2**i, MAX_DIM)
            layers += [Conv2dBlock(
                n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
            )]
            n_in = n_out
        self.conv = nn.Sequential(*layers)
        self.fc_adv = nn.Sequential(
            LinearBlock(1024 * self.f_size * self.f_size, fc_dim, fc_norm_fn, fc_acti_fn),
            LinearBlock(fc_dim, 1, 'none', 'none')
        )
       
        self.fc_cls = nn.Sequential(
            LinearBlock(1024 * self.f_size * self.f_size, fc_dim, fc_norm_fn, fc_acti_fn),
            LinearBlock(fc_dim, 1, 'none', 'none')
        )

        self.fc_cls8 = nn.Sequential(
            LinearBlock(1024 * self.f_size * self.f_size, fc_dim, fc_norm_fn, fc_acti_fn),
            LinearBlock(fc_dim, 6, 'none', 'none')
        )
        
    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_adv(h), self.fc_cls(h), self.fc_cls8(h)

class Discriminators1(nn.Module):
    # No instancenorm in fcs in source code, which is different from paper.
    def __init__(self, dim=64, norm_fn='instancenorm', acti_fn='lrelu',
                 fc_dim=1024, fc_norm_fn='none', fc_acti_fn='lrelu', n_layers=5, img_size=128):
        super(Discriminators1, self).__init__()
        self.f_size = img_size // 2**n_layers
        
        layers = []
        n_in = 3
        for i in range(n_layers):
            n_out = min(dim * 2**i, MAX_DIM)
            layers += [Conv2dBlock(
                n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
            )]
            n_in = n_out
        self.conv = nn.Sequential(*layers)
        self.fc_adv = nn.Sequential(
            LinearBlock(1024 * self.f_size * self.f_size, fc_dim, fc_norm_fn, fc_acti_fn),
            LinearBlock(fc_dim, 1, 'none', 'none')
        )

        self.fc_cls = nn.Sequential(
            LinearBlock(1024 * self.f_size * self.f_size, fc_dim, fc_norm_fn, fc_acti_fn),
            LinearBlock(fc_dim, 6, 'none', 'none')
        )
        
    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_adv(h), self.fc_cls(h)


import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim


# multilabel_soft_margin_loss = sigmoid + binary_cross_entropy

class AttGAN():
    def __init__(self, args):
        self.mode = args.mode
        self.gpu = args.gpu
        self.multi_gpu = args.multi_gpu if 'multi_gpu' in args else False
        self.lambda_1 = args.lambda_1
        self.lambda_2 = args.lambda_2
        self.lambda_3 = args.lambda_3
        self.lambda_gp = args.lambda_gp
        
        self.G = Generator(
            args.enc_dim, args.enc_layers, args.enc_norm, args.enc_acti,
            args.dec_dim, args.dec_layers, args.dec_norm, args.dec_acti,
            args.n_attrs, args.shortcut_layers, args.inject_layers, args.img_size
        )
        self.G.train()
        if self.gpu: self.G.cuda()
        summary(self.G, [(3, args.img_size, args.img_size), (args.n_attrs, 1, 1)], batch_size=4, device='cuda' if args.gpu else 'cpu')
        
        self.D = Discriminators(
            args.dis_dim, args.dis_norm, args.dis_acti,
            args.dis_fc_dim, args.dis_fc_norm, args.dis_fc_acti, args.dis_layers, args.img_size
        )
        self.D.train()
        if self.gpu: self.D.cuda()
        summary(self.D, [(3, args.img_size, args.img_size)], batch_size=4, device='cuda' if args.gpu else 'cpu')

        self.D1 = Discriminators1(
            args.dis_dim, args.dis_norm, args.dis_acti,
            args.dis_fc_dim, args.dis_fc_norm, args.dis_fc_acti, args.dis_layers, args.img_size
        )
        self.D1.train()
        if self.gpu: self.D1.cuda()
        summary(self.D1, [(3, args.img_size, args.img_size)], batch_size=4, device='cuda' if args.gpu else 'cpu')
        
        if self.multi_gpu:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)
            self.D1 = nn.DataParallel(self.D1)
        
        self.optim_G = optim.Adam(self.G.parameters(), lr=args.lr, betas=args.betas)
        self.optim_D = optim.Adam(self.D.parameters(), lr=args.lr, betas=args.betas)
        self.optim_D1 = optim.Adam(self.D1.parameters(), lr=args.lr, betas=args.betas)
    
    def set_lr(self, lr):
        for g in self.optim_G.param_groups:
            g['lr'] = lr
        for g in self.optim_D.param_groups:
            g['lr'] = lr
        for g in self.optim_D1.param_groups:
            g['lr'] = lr
        
    #@torchsnooper.snoop()
    def classify(self, zs, a):
        box = []
        for i in range(a.size(1)):
            num_chs = zs.size(1)
            per_chs = float(num_chs)
            start = int(np.rint(per_chs * i))
            end = int(np.rint(per_chs * (i + 1)))
            temp = zs.narrow(1, start, end-start)
            av = a.view(a.size(0), -1, 1, 1)
            ai =av[:,i,:,:]
            ai = torch.unsqueeze(ai, 1)
            tar_i =ai.repeat(1, 1,  zs.size(2), zs.size(2))
            box.append(torch.mul(tar_i, zs))
            break
        re = torch.cat(box, dim=1)	
        return re

    def diffatt(self, re_a, re_b, att_a, att_b, index):
        num_chs = re_a.size(1)
        per_chs = float(num_chs) / 4
        box = []
        for i in range(att_a.size(1)):
            start = int(np.rint(per_chs * i))
            end = int(np.rint(per_chs * (i + 1)))
            temp_a = re_a.narrow(1, start, end-start)
            temp_b = re_b.narrow(1, start, end-start)
            if i != index and i != 1 :
                #box.append(torch.zeros_like(temp_b))
                box.append(temp_a)
            else:
                box.append(temp_b)
        z = torch.cat(box, dim=1)		
        return z

    def trainG(self, img_a, img_b, att_a, att_b):
        for p in self.D.parameters():
            p.requires_grad = False
        
        zs_a, z_ma, ma128 = self.G(img_a, mode='enc')
        zs_b, z_mb, mb128 = self.G(img_b, mode='enc')
        att_b_1 = att_b[:,0].view(att_b.size(0), -1)
        att_a_1 = att_a[:,0].view(att_a.size(0), -1)
        gen2_b = self.classify(z_mb, att_b_1)
        gen2_a = self.classify(z_ma, att_a_1)
        ma128 =  self.classify(ma128, att_a_1)
        mb128 =  self.classify(mb128, att_b_1)

        img_recon_b, msb, msb1, zm_b128  = self.G(zs_b, gen2_b, att_b_1, mode='dec')
        img_recon_a, msa, msa1, zm_a128 = self.G(zs_a, gen2_a, att_a_1, mode='dec')
        
        img_fake_a = self.G(zs_a, zs_b, gen2_a, gen2_b, msb, mode='dec1')
        img_fake_b = self.G(zs_b, zs_a, gen2_b, gen2_a, msb, mode='dec1')

        #F.interpolate(msb[-], size = (128,128), mode = 'bilinear')
        img_fake_a = img_a*(1-zm_b128) + img_fake_a*zm_b128
        img_fake_b = img_b*(1-zm_b128) + img_fake_b*zm_b128
        img_fake_a8 = img_a*(1-msb1[1]) + img_fake_a*msb1[1]
        img_fake_b8 = img_b*(1-msb1[1]) + img_fake_b*msb1[1]
        img_fake_a4 = img_a*(1-msb1[2]) + img_fake_a*msb1[2]
        img_fake_b4 = img_b*(1-msb1[2]) + img_fake_b*msb1[2]
        img_fake_a2 = img_a*(1-msb1[3]) + img_fake_a*msb1[3]
        img_fake_b2 = img_b*(1-msb1[3]) + img_fake_b*msb1[3]
        
        d_a_fake, dc_a_fake, dc8_a_fake = self.D(img_fake_a)
        d_b_fake, dc_b_fake, dc8_b_fake = self.D(img_fake_b)

        d_a_fake8, dc_a_fake8, dc8_a_fake8 = self.D(img_fake_a8)
        d_b_fake8, dc_b_fake8, dc8_b_fake8 = self.D(img_fake_b8)
        d_a_fake4, dc_a_fake4, dc8_a_fake4 = self.D(img_fake_a4)
        d_b_fake4, dc_b_fake4, dc8_b_fake4 = self.D(img_fake_b4)
        d_a_fake2, dc_a_fake2, dc8_a_fake2 = self.D(img_fake_a2)
        d_b_fake2, dc_b_fake2, dc8_b_fake2 = self.D(img_fake_b2)
        
        if self.mode == 'wgan':
            gf_loss = -d_a_fake.mean()-d_b_fake.mean()-d_a_fake8.mean()-d_b_fake8.mean()-d_a_fake4.mean()-d_b_fake4.mean()-d_a_fake2.mean()-d_b_fake2.mean()

        
        gr_loss = F.l1_loss(img_recon_a, img_a) + F.l1_loss(img_recon_b, img_b)

        att_b_f = att_b.clone()
        att_b_f[:, 0] = att_a[:, 0]
        att_a_f = att_a.clone()
        att_a_f[:, 0] = att_b[:, 0]
        gc_loss = F.binary_cross_entropy_with_logits(dc_a_fake, att_b_1) + F.binary_cross_entropy_with_logits(dc_b_fake, att_a_1) + F.binary_cross_entropy_with_logits(dc_a_fake8, att_b_1) + F.binary_cross_entropy_with_logits(dc_b_fake8, att_a_1)+ F.binary_cross_entropy_with_logits(dc_a_fake4, att_b_1) + F.binary_cross_entropy_with_logits(dc_b_fake4, att_a_1) + F.binary_cross_entropy_with_logits(dc_a_fake2, att_b_1) + F.binary_cross_entropy_with_logits(dc_b_fake2, att_a_1)

        gc_loss8 = F.binary_cross_entropy_with_logits(dc8_a_fake, att_a_f) + F.binary_cross_entropy_with_logits(dc8_b_fake, att_b_f) + F.binary_cross_entropy_with_logits(dc8_a_fake8, att_a_f) + F.binary_cross_entropy_with_logits(dc8_b_fake8, att_b_f)+ F.binary_cross_entropy_with_logits(dc8_a_fake4, att_a_f) + F.binary_cross_entropy_with_logits(dc8_b_fake4, att_b_f)+ F.binary_cross_entropy_with_logits(dc8_a_fake2, att_a_f) + F.binary_cross_entropy_with_logits(dc8_b_fake2, att_b_f)

        g_loss = gf_loss + 100 * gr_loss + 10*gc_loss + 5*gc_loss8
        
        self.optim_G.zero_grad()
        g_loss.backward()
        self.optim_G.step()
        
        errG = {
            'g_loss': g_loss.item(), 'gf_loss': gf_loss.item(),
            'gr_loss': gr_loss.item()
            #'vgg_loss': vgg_loss.item()
        }
        return errG

    #@torchsnooper.snoop()
    def trainD(self, img_a, img_b, att_a, att_b):
        for p in self.D.parameters():
            p.requires_grad = True
        zs_a, z_ma, ma128 = self.G(img_a, mode='enc')
        zs_b, z_mb, mb128 = self.G(img_b, mode='enc')
        att_b_1 = att_b[:,0].view(att_b.size(0), -1)
        att_a_1 = att_a[:,0].view(att_a.size(0), -1)
        gen2_b = self.classify(z_mb, att_b_1)
        gen2_a = self.classify(z_ma, att_a_1)
        ma128 =  self.classify(ma128, att_a_1)
        mb128 =  self.classify(mb128, att_b_1)

        img_recon_b, msb, msb1, zm_b128  = self.G(zs_b, gen2_b, att_b_1, mode='dec')
        img_recon_a, msa, msa1, zm_a128 = self.G(zs_a, gen2_a, att_a_1, mode='dec')
        
        img_fake_a = self.G(zs_a, zs_b, gen2_a, gen2_b, msb, mode='dec1')
        img_fake_b = self.G(zs_b, zs_a, gen2_b, gen2_a, msb, mode='dec1')

       

        img_fake_a = img_a*(1-zm_b128) + img_fake_a*zm_b128
        img_fake_b = img_b*(1-zm_b128) + img_fake_b*zm_b128

        img_fake_a8 = img_a*(1-msb1[1]) + img_fake_a*msb1[1]
        img_fake_b8 = img_b*(1-msb1[1]) + img_fake_b*msb1[1]
        img_fake_a4 = img_a*(1-msb1[2]) + img_fake_a*msb1[2]
        img_fake_b4 = img_b*(1-msb1[2]) + img_fake_b*msb1[2]
        img_fake_a2 = img_a*(1-msb1[3]) + img_fake_a*msb1[3]
        img_fake_b2 = img_b*(1-msb1[3]) + img_fake_b*msb1[3]

        d_real_b, dc_real_b, dc8_real_b = self.D(img_b)
        d_fake_a , _, _= self.D(img_fake_a.detach())
        d_fake_a8 , _, _= self.D(img_fake_a8.detach())
        d_fake_a4 , _, _= self.D(img_fake_a4.detach())
        d_fake_a2 , _, _= self.D(img_fake_a2.detach())
        d_real_a, dc_real_a, dc8_real_a = self.D(img_a)
        d_fake_b, _, _ = self.D(img_fake_b.detach())
        d_fake_b8, _, _ = self.D(img_fake_b8.detach())
        d_fake_b4, _, _ = self.D(img_fake_b4.detach())
        d_fake_b2, _, _ = self.D(img_fake_b2.detach())
        
        def gradient_penalty(f, real, fake=None):
            def interpolate(a, b=None):
                if b is None:  # interpolation in DRAGAN
                    beta = torch.rand_like(a)
                    b = a + 0.5 * a.var().sqrt() * beta
                alpha = torch.rand(a.size(0), 1, 1, 1)
                alpha = alpha.cuda() if self.gpu else alpha
                inter = a + alpha * (b - a)
                return inter
            x = interpolate(real, fake).requires_grad_(True)
            pred = f(x)
            if isinstance(pred, tuple):
                pred = pred[0]
            grad = autograd.grad(
                outputs=pred, inputs=x,
                grad_outputs=torch.ones_like(pred),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            grad = grad.view(grad.size(0), -1)
            norm = grad.norm(2, dim=1)
            gp = ((norm - 1.0) ** 2).mean()
            return gp
        
        if self.mode == 'wgan':
            wd = d_real_b.mean() - d_fake_a.mean()+d_real_b.mean() - d_fake_a8.mean() +d_real_b.mean() - d_fake_a4.mean() + d_real_b.mean() - d_fake_a2.mean() + d_real_a.mean() - d_fake_b.mean() + d_real_a.mean() - d_fake_b8.mean() + d_real_a.mean() - d_fake_b4.mean() + d_real_a.mean() - d_fake_b2.mean()
            df_loss = -wd
            df_gp =  gradient_penalty(self.D, img_a, img_fake_a) + gradient_penalty(self.D, img_a, img_fake_a8) + gradient_penalty(self.D, img_a, img_fake_a4) + gradient_penalty(self.D, img_a, img_fake_a2) + gradient_penalty(self.D, img_b, img_fake_b) + gradient_penalty(self.D, img_b, img_fake_b8) + gradient_penalty(self.D, img_b, img_fake_b4) + gradient_penalty(self.D, img_b, img_fake_b2)

        dc_loss = F.binary_cross_entropy_with_logits(dc_real_b, att_b_1) + F.binary_cross_entropy_with_logits(dc_real_a, att_a_1)
        dc8_loss = F.binary_cross_entropy_with_logits(dc8_real_a, att_a) + F.binary_cross_entropy_with_logits(dc8_real_b, att_b)
        d_loss = df_loss + self.lambda_gp * df_gp + 10*dc_loss + 5*dc8_loss
        
        self.optim_D.zero_grad()
        d_loss.backward()
        self.optim_D.step()
        
        errD = {
            'd_loss': d_loss.item(), 'df_loss': df_loss.item(), 
            'df_gp': df_gp.item()
        }
        return errD

    def trainD1(self, img_a, img_b, att_a, att_b):
        for p in self.D.parameters():
            p.requires_grad = True
        zs_a, z_ma, ma128 = self.G(img_a, mode='enc')
        zs_b, z_mb, mb128 = self.G(img_b, mode='enc')
        att_b_1 = att_b[:,0].view(att_b.size(0), -1)
        att_a_1 = att_a[:,0].view(att_a.size(0), -1)
        gen2_b = self.classify(z_mb, att_b_1)
        gen2_a = self.classify(z_ma, att_a_1)
        ma128 =  self.classify(ma128, att_a_1)
        mb128 =  self.classify(mb128, att_b_1)

        
        img_fake_a = self.G(zs_a, zs_b, gen2_a, gen2_b, mode='dec1')
        img_fake_b = self.G(zs_b, zs_a, gen2_b, gen2_a, mode='dec1')
        img_fake_a = img_a*(1-mb128) + img_fake_a*mb128
        img_fake_b = img_b*(1-mb128) + img_fake_b*mb128

        d_real_a, dc_real_a = self.D1(img_a)
        d_fake_b, _ = self.D1(img_fake_b.detach())
        
        def gradient_penalty(f, real, fake=None):
            def interpolate(a, b=None):
                if b is None:  # interpolation in DRAGAN
                    beta = torch.rand_like(a)
                    b = a + 0.5 * a.var().sqrt() * beta
                alpha = torch.rand(a.size(0), 1, 1, 1)
                alpha = alpha.cuda() if self.gpu else alpha
                inter = a + alpha * (b - a)
                return inter
            x = interpolate(real, fake).requires_grad_(True)
            pred = f(x)
            if isinstance(pred, tuple):
                pred = pred[0]
            grad = autograd.grad(
                outputs=pred, inputs=x,
                grad_outputs=torch.ones_like(pred),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            grad = grad.view(grad.size(0), -1)
            norm = grad.norm(2, dim=1)
            gp = ((norm - 1.0) ** 2).mean()
            return gp
        
        if self.mode == 'wgan':
            wd = d_real_a.mean() - d_fake_b.mean()
            df_loss = -wd
            df_gp = gradient_penalty(self.D1, img_a, img_fake_b)

        dc_loss = F.binary_cross_entropy_with_logits(dc_real_a, att_a)
        d_loss = df_loss + self.lambda_gp * df_gp #+ 10*dc_loss
        
        self.optim_D1.zero_grad()
        d_loss.backward()
        self.optim_D1.step()
        
        errD = {
            'd_loss': d_loss.item(), 'df_loss': df_loss.item(), 
            'df_gp': df_gp.item()
        }
        return errD

    
    def train(self):
        self.G.train()
        self.D.train()
        self.D1.train()
    
    def eval(self):
        self.G.eval()
        self.D.eval()
        self.D1.eval()
    
    def save(self, path):
        states = {
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'D1': self.D.state_dict(),
            'optim_G': self.optim_G.state_dict(),
            'optim_D': self.optim_D.state_dict(),
            'optim_D1': self.optim_D.state_dict()
        }
        torch.save(states, path)
    
    def load(self, path):
        states = torch.load(path, map_location=lambda storage, loc: storage)
        if 'G' in states:
            self.G.load_state_dict(states['G'])
        if 'D' in states:
            self.D.load_state_dict(states['D'])
        if 'D1' in states:
            self.D1.load_state_dict(states['D1'])
        if 'optim_G' in states:
            self.optim_G.load_state_dict(states['optim_G'])
        if 'optim_D' in states:
            self.optim_D.load_state_dict(states['optim_D'])
        if 'optim_D1' in states:
            self.optim_D1.load_state_dict(states['optim_D1'])
    
    def saveG(self, path):
        states = {
            'G': self.G.state_dict()
        }
        torch.save(states, path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', dest='img_size', type=int, default=128)
    parser.add_argument('--shortcut_layers', dest='shortcut_layers', type=int, default=1)
    parser.add_argument('--inject_layers', dest='inject_layers', type=int, default=0)
    parser.add_argument('--enc_dim', dest='enc_dim', type=int, default=64)
    parser.add_argument('--dec_dim', dest='dec_dim', type=int, default=64)
    parser.add_argument('--dis_dim', dest='dis_dim', type=int, default=64)
    parser.add_argument('--dis_fc_dim', dest='dis_fc_dim', type=int, default=1024)
    parser.add_argument('--enc_layers', dest='enc_layers', type=int, default=5)
    parser.add_argument('--dec_layers', dest='dec_layers', type=int, default=5)
    parser.add_argument('--dis_layers', dest='dis_layers', type=int, default=5)
    parser.add_argument('--enc_norm', dest='enc_norm', type=str, default='batchnorm')
    parser.add_argument('--dec_norm', dest='dec_norm', type=str, default='batchnorm')
    parser.add_argument('--dis_norm', dest='dis_norm', type=str, default='instancenorm')
    parser.add_argument('--dis_fc_norm', dest='dis_fc_norm', type=str, default='none')
    parser.add_argument('--enc_acti', dest='enc_acti', type=str, default='lrelu')
    parser.add_argument('--dec_acti', dest='dec_acti', type=str, default='relu')
    parser.add_argument('--dis_acti', dest='dis_acti', type=str, default='lrelu')
    parser.add_argument('--dis_fc_acti', dest='dis_fc_acti', type=str, default='relu')
    parser.add_argument('--lambda_1', dest='lambda_1', type=float, default=100.0)
    parser.add_argument('--lambda_2', dest='lambda_2', type=float, default=10.0)
    parser.add_argument('--lambda_3', dest='lambda_3', type=float, default=1.0)
    parser.add_argument('--lambda_gp', dest='lambda_gp', type=float, default=10.0)
    parser.add_argument('--mode', dest='mode', default='wgan', choices=['wgan', 'lsgan', 'dcgan'])
    parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.5)
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.999)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    args.n_attrs = 13
    args.betas = (args.beta1, args.beta2)
    attgan = AttGAN(args)
