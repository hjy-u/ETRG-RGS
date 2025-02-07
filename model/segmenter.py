import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.clip import build_model
from .bridger import Bridger_SA_RN_depth
from .layers import FPN, TransformerDecoder, MultiTaskProjector
import torchvision
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1)) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return bce + 0.1*dice

# through adpt mhsa
class ETRG_depth(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Vision & Text Encoder
        clip_model = torch.jit.load(cfg.clip_pretrain, map_location="cpu").eval()
        self.backbone = build_model(clip_model.state_dict(), cfg.word_len).float()
        self.bridger = Bridger_SA_RN_depth(d_model=cfg.ladder_dim, nhead=cfg.nhead, fusion_stage=cfg.multi_stage, word_dim = cfg.word_dim)

        # Fix Backbone non parameter is used in backbone at all
        for param_name, param in self.backbone.named_parameters():
            #if 'positional_embedding' not in param_name:
            param.requires_grad = False
        self.input_dim = 1024  # (clip visual input dim, after attnpooling)
        self.batchnorm = cfg.batchnorm
        self.lang_fusion_type = cfg.lang_fusion_type
        self.bilinear = cfg.bilinear
        self.up_factor = 2 if self.bilinear else 1
        self._build_decoder(cfg)

        self.loss = BCEDiceLoss()

    def _build_decoder(self, cfg):
        resnet18 = list(torchvision.models.resnet18(pretrained=True).children())[:-2]
        # input 4 dims -> RGBD
        resnet18[0] = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        nn.init.kaiming_normal_(resnet18[0].weight, mode='fan_in', nonlinearity='relu')
        self.resnet18 = torch.nn.Sequential(*resnet18)
        self.zoom_in = nn.Conv2d(512, 64, kernel_size=1, stride=1, bias=False)
        nn.init.kaiming_normal_(self.zoom_in.weight, mode='fan_in', nonlinearity='relu')
        self.visual_sent_fpn = FPN(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out, language_fuser=True, decoding=False)
        self.proj = MultiTaskProjector(cfg.word_dim, cfg.vis_dim // 2, 3)
        self.decoder = TransformerDecoder(num_layers=cfg.num_layers,
                          d_model=cfg.vis_dim,
                          nhead=cfg.num_head,
                          dim_ffn=cfg.dim_ffn,
                          dropout=cfg.dropout,
                          return_intermediate=cfg.intermediate)


    def forward(self, img, depth, word, mask=None, grasp_qua_mask=None, grasp_sin_mask=None, grasp_cos_mask=None, grasp_wid_mask=None):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w
        '''
        input_shape = img.shape[-2:]
        # padding mask used in decoder
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()
        rgbd = depth.repeat(1,3,1,1)
        # rgbd = torch.cat([img, depth], dim=1)
        rgbd_feature = self.zoom_in(self.resnet18(rgbd))
        B, C, _, _ = rgbd_feature.size()
        rgbd_feature = rgbd_feature.reshape(B, C, -1).permute(2, 0, 1)

        # vis: C3 / C4 / C5
        # word: b, length, 1024
        # state: b, 1024
        im, word, state = self.bridger(img, word, self.backbone, pad_mask, rgbd_feature)

        # im[-1] *= depth
        x_sent = self.visual_sent_fpn(im, state)
        B, _, H, W = x_sent.size()

        vis, _ = self.decoder(x_sent, word, pad_mask=pad_mask)  # (x1 not need to LN, but x0 has to)
        vis = vis.reshape(B, -1, H, W)

        pred, grasp_qua_pred, grasp_sin_pred, grasp_cos_pred, grasp_wid_pred = self.proj(vis, state)

        if pred.shape[-2:] != mask.shape[-2:]:
            pred = F.interpolate(pred, size=input_shape, mode='bilinear')
            grasp_qua_pred = F.interpolate(grasp_qua_pred, size=input_shape, mode='bilinear')
            grasp_sin_pred = F.interpolate(grasp_sin_pred, size=input_shape, mode='bilinear')
            grasp_cos_pred = F.interpolate(grasp_cos_pred, size=input_shape, mode='bilinear')
            grasp_wid_pred = F.interpolate(grasp_wid_pred, size=input_shape, mode='bilinear')

        if self.training:


            # loss = self.loss(pred, mask)

            # weight = mask * 0.5 + 1

            loss = self.loss(pred, mask)
            # loss = F.binary_cross_entropy_with_logits(pred, mask)


            grasp_qua_loss = F.smooth_l1_loss(grasp_qua_pred, grasp_qua_mask)
            grasp_sin_loss = F.smooth_l1_loss(grasp_sin_pred, grasp_sin_mask)
            grasp_cos_loss = F.smooth_l1_loss(grasp_cos_pred, grasp_cos_mask)
            grasp_wid_loss = F.smooth_l1_loss(grasp_wid_pred, grasp_wid_mask)

            # @TODO adjust coef of different loss items
            total_loss = loss + grasp_qua_loss + grasp_sin_loss + grasp_cos_loss + grasp_wid_loss
            # if torch.isnan(total_loss).sum() > 0:
            #     a = 1

            loss_dict = {}
            loss_dict["m_ins"] = loss.item()
            loss_dict["m_qua"] = grasp_qua_loss.item()
            loss_dict["m_sin"] = grasp_sin_loss.item()
            loss_dict["m_cos"] = grasp_cos_loss.item()
            loss_dict["m_wid"] = grasp_wid_loss.item()

            # loss = F.binary_cross_entropy_with_logits(pred, mask, reduction="none").sum(dim=(2,3))
            # loss = torch.dot(coef.squeeze(), loss.squeeze()) / (mask.shape[0] * mask.shape[2] * mask.shape[3])

            return ((pred.detach(), grasp_qua_pred.detach(), grasp_sin_pred.detach(), grasp_cos_pred.detach(),
                     grasp_wid_pred.detach()), (mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask),
                    total_loss, loss_dict)
        else:
            return (pred.detach(), grasp_qua_pred.detach(), grasp_sin_pred.detach(), grasp_cos_pred.detach(),
                    grasp_wid_pred.detach()), (mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask)