import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import model.fusion as fusion
from torch import Tensor


def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(True))


def linear_layer(in_dim, out_dim, bias=False):
    return nn.Sequential(nn.Linear(in_dim, out_dim, bias),
                         nn.BatchNorm1d(out_dim), nn.ReLU(True))

class MultiTaskProjector(nn.Module):
    def __init__(self, word_dim=1024, in_dim=256, kernel_size=3):
        super().__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        # visual projector
        self.vis = nn.Sequential(  # os16 -> os4
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim * 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim, 3, padding=1),
            nn.Conv2d(in_dim, in_dim*5, 1))

        # textual projector
        out_dim = 1 * in_dim * kernel_size * kernel_size + 5
        self.txt = nn.Linear(word_dim, out_dim)

    def forward(self, x, word):
        '''
            x: b, 512, 26, 26
            word: b, 512
        '''
        x = self.vis(x)
        x = torch.tensor_split(x, 5, dim=1) # no tensor_split api in torch 1.7, please use it in higher version
        # x = torch.chunk(x, 5, dim=1)

        mask_x = x[0]
        grasp_qua_x = x[1]
        grasp_wid_x = x[2]
        grasp_sin_x = x[3]
        grasp_cos_x = x[4]

        B, C, H, W = mask_x.size()


        # 1, b*256, 104, 104
        mask_x = mask_x.reshape(1, B * C, H, W)
        grasp_qua_x = grasp_qua_x.reshape(1, B * C, H, W)
        grasp_sin_x = grasp_sin_x.reshape(1, B * C, H, W)
        grasp_cos_x = grasp_cos_x.reshape(1, B * C, H, W)
        grasp_wid_x = grasp_wid_x.reshape(1, B * C, H, W)


        # txt: b, (256*3*3 + 1) -> b, 256, 3, 3 / b
        word = self.txt(word)
        weight, bias = word[:, :-5], word[:, -5:]
        weight = weight.reshape(B, C, self.kernel_size, self.kernel_size)
        # Conv2d - 1, b*256, 104, 104 -> 1, b, 104, 104
        mask_out = F.conv2d(mask_x,
                       weight,
                       padding=self.kernel_size // 2,
                       groups=weight.size(0),
                       bias=bias[:,0])
        
        grasp_qua_out = F.conv2d(grasp_qua_x,
                            weight,
                            padding=self.kernel_size // 2,
                            groups=weight.size(0),
                            bias=bias[:,1])
        
        grasp_sin_out = F.conv2d(grasp_sin_x,
                            weight,
                            padding=self.kernel_size // 2,
                            groups=weight.size(0),
                            bias=bias[:,2])

        grasp_cos_out = F.conv2d(grasp_cos_x,
                            weight,
                            padding=self.kernel_size // 2,
                            groups=weight.size(0),
                            bias=bias[:,3])
        
        grasp_wid_out = F.conv2d(grasp_wid_x,
                            weight,
                            padding=self.kernel_size // 2,
                            groups=weight.size(0),
                            bias=bias[:,-1])
            
        mask_out = mask_out.transpose(0, 1)
        grasp_qua_out = grasp_qua_out.transpose(0, 1)
        grasp_sin_out = grasp_sin_out.transpose(0, 1)
        grasp_cos_out = grasp_cos_out.transpose(0, 1)
        grasp_wid_out = grasp_wid_out.transpose(0, 1)
        # b, 1, 104, 104

        return mask_out, grasp_qua_out, grasp_sin_out, grasp_cos_out, grasp_wid_out

################added###########
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class VLSAadapter(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ffn = FeedForward(d_model, dim_feedforward, dropout=dropout, act='relu')
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        spatial_shape = [tgt.shape[0], memory.shape[0]]
        #cat text with visual
        encoder_input_list = [tgt, memory] # vis + txt
        encoder_input = torch.cat(encoder_input_list, dim=0)
        tgt2 = self.self_attn(query=encoder_input, key=encoder_input, value=encoder_input)[0]
        tgt = encoder_input + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout2(tgt2)
        # tgt = self.norm2(tgt)
        vistxt = torch.split(tgt, spatial_shape, dim=0)
        vis = vistxt[0]
        txt = vistxt[1]
        return vis, txt

class VLSAadapter_depth(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ffn = FeedForward(d_model, dim_feedforward, dropout=dropout, act='relu')
        self.norm1 = nn.LayerNorm(d_model)     
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, depth,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        spatial_shape = [depth.shape[0], tgt.shape[0], memory.shape[0]]
        #cat text with visual
        encoder_input_list = [depth, tgt, memory] # vis + txt
        encoder_input = torch.cat(encoder_input_list, dim=0)
        tgt2 = self.self_attn(query=encoder_input, key=encoder_input, value=encoder_input)[0]
        tgt = encoder_input + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout2(tgt2)
        # tgt = self.norm2(tgt)
        vistxt = torch.split(tgt, spatial_shape, dim=0)
        depth = vistxt[0]
        vis = vistxt[1]
        txt = vistxt[-1]
        return vis, txt, depth

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None, act='gelu'):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU() if act =='gelu' else nn.ReLU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

class FPN(nn.Module):
    def __init__(self, in_channels=[512, 1024, 1024],
                 out_channels=[256, 512, 1024], language_fuser=True, decoding=False):
        super(FPN, self).__init__()

        self.proj_input_dim = in_channels[-1]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lang_fusion_type = 'mult'
        self.language_fuser = language_fuser

        self.conv0 = conv_layer(in_channels[2], out_channels[1], 1, 0)
        self.conv1 = conv_layer(in_channels[1], out_channels[1], 1, 0)
        self.conv2 = conv_layer(in_channels[0], out_channels[1], 1, 0)

        if language_fuser:
            self.lang_proj0 = nn.Linear(self.proj_input_dim, out_channels[1])
            self.lang_fuser0 = fusion.names[self.lang_fusion_type](input_dim=self.in_channels[1])
            self.lang_proj1 = nn.Linear(self.proj_input_dim, out_channels[1])
            self.lang_fuser1 = fusion.names[self.lang_fusion_type](input_dim=self.in_channels[1])
            self.lang_proj2 = nn.Linear(self.proj_input_dim, out_channels[1])
            self.lang_fuser2 = fusion.names[self.lang_fusion_type](input_dim=self.in_channels[1])

        self.convp4 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        self.convp3 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        self.convp2 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        self.coordconv = nn.Sequential(conv_layer(3*out_channels[1], out_channels[1], 3, 1))

        # self.initialize_parameters()

    def initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, imgs, sent_emb=None):
        x2, x3, x4 = imgs
        p4 = self.conv0(x4)
        if self.language_fuser:
            p4 = self.lang_fuser0(p4, sent_emb, x2_mask=None, x2_proj=self.lang_proj0)
        p4_up = F.interpolate(p4, scale_factor=2, mode='bilinear')

        x3 = self.conv1(x3)
        p3 = x3 + p4_up
        if self.language_fuser:
            p3 = self.lang_fuser1(p3, sent_emb, x2_mask=None, x2_proj=self.lang_proj1)
        p3_up = F.interpolate(p3, scale_factor=2, mode='bilinear')

        x2 = self.conv2(x2)
        p2 = x2 + p3_up
        if self.language_fuser:
            p2 = self.lang_fuser2(p2, sent_emb, x2_mask=None, x2_proj=self.lang_proj2)

        f4 = self.convp4(p4)
        f4 = F.interpolate(f4, scale_factor=2, mode='bilinear')

        f3 = self.convp3(p3)
        f2 = self.convp2(p2)
        f2 = F.avg_pool2d(f2, 2, 2)

        fv = torch.cat([f4, f3, f2], dim=1)
        fv = self.coordconv(fv)

        return fv

class TransformerDecoder(nn.Module):
    def __init__(self,
                 num_layers,
                 d_model,
                 nhead,
                 dim_ffn,
                 dropout,
                 return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model=d_model,
                                    nhead=nhead,
                                    dim_feedforward=dim_ffn,
                                    dropout=dropout) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)

        # self.initialize_parameters()
    ####added
    def initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def pos1d(d_model, length):
        """
        https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                              -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe.unsqueeze(1)  # n, 1, 512

    @staticmethod
    def pos2d(d_model, height, width):
        """
        https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe.reshape(-1, 1, height * width).permute(2, 1, 0)  # hw, 1, 512

    def forward(self, vis, txt, pad_mask=None, attn_mask=None):
        '''
            vis: b, 512, h, w
            txt: b, L, 512
            pad_mask: b, L
        '''
        interm = False
        B, C, H, W = vis.size()
        _, L, D = txt.size()
        # position encoding
        vis_pos = self.pos2d(C, H, W)
        txt_pos = self.pos1d(D, L)
        # reshape & permute
        vis = vis.reshape(B, C, -1).permute(2, 0, 1)
        txt = txt.permute(1, 0, 2)
        # forward
        output = vis
        intermediate = []
        for layer in self.layers:
            output = layer(output, txt, vis_pos, txt_pos, pad_mask=pad_mask, attn_mask=attn_mask)
            if interm:
                intermediate.append(output)
        # HW, b, 512 -> b, 512, HW
        output = self.norm(output).permute(1, 2, 0)
        return output, intermediate

class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=512,
                 nhead=8,
                 dim_feedforward=512,
                 dropout=0.1):
        super().__init__()
        # Normalization Layer
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        # Attention Layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model,
                                                    nhead,
                                                    dropout=dropout,
                                                    kdim=d_model,
                                                    vdim=d_model)
        # FFN
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward),
                                 nn.ReLU(True),
                                 nn.Dropout(dropout),
                                 nn.Linear(dim_feedforward, d_model))

        # LayerNorm & Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos.to(tensor.device)

    def forward(self, vis, txt, vis_pos, txt_pos, pad_mask=None, attn_mask=None):
        '''
            vis: 26*26, b, 512
            txt: L, b, 512
            vis_pos: 26*26, 1, 512
            txt_pos: L, 1, 512
            pad_mask: b, L
            from DETR
        '''
        # Self-Attention
        # vis2 = self.norm1(vis)
        q = k = self.with_pos_embed(vis, vis_pos)
        vis2 = self.self_attn(q, k, value=vis)[0]
        vis = vis + self.dropout1(vis2)
        vis = self.self_attn_norm(vis)

        # Cross-Attention
        vis2 = self.multihead_attn(query=self.with_pos_embed(vis, vis_pos),
                                   key=self.with_pos_embed(txt, txt_pos),
                                   value=txt,
                                   key_padding_mask=pad_mask)[0]
        vis = vis + self.dropout2(vis2)
        vis = self.cross_attn_norm(vis)
        # FFN
        vis2 = self.ffn(vis)
        vis = vis + self.dropout3(vis2)
        vis = self.norm1(vis)

        return vis