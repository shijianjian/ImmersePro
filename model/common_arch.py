''' Towards An End-to-End Framework for Video Inpainting
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from einops import rearrange

from model.modules.base_module import BaseNetwork
from model.modules.sparse_transformer import TemporalSparseTransformerBlock, SoftSplit, SoftComp
from model.modules.spectral_norm import spectral_norm as _spectral_norm
from model.modules.flow_loss_utils import flow_warp
from model.modules.deformconv import ModulatedDeformConv2d

from .misc import constant_init


def length_sq(x):
    return torch.sum(torch.square(x), dim=1, keepdim=True)


def fbConsistencyCheck(flow_fw, flow_bw, alpha1=0.01, alpha2=0.5):
    flow_bw_warped = flow_warp(flow_bw, flow_fw.permute(0, 2, 3, 1))  # wb(wf(x))
    flow_diff_fw = flow_fw + flow_bw_warped  # wf + wb(wf(x))

    mag_sq_fw = length_sq(flow_fw) + length_sq(flow_bw_warped)  # |wf| + |wb(wf(x))|
    occ_thresh_fw = alpha1 * mag_sq_fw + alpha2

    # fb_valid_fw = (length_sq(flow_diff_fw) < occ_thresh_fw).float()
    fb_valid_fw = (length_sq(flow_diff_fw) < occ_thresh_fw).to(flow_fw)
    return fb_valid_fw
        

class DeformableAlignment(ModulatedDeformConv2d):
    """Second-order deformable alignment module."""
    def __init__(self, *args, **kwargs):
        # self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 3)
        self.additional_channels = kwargs.pop('additional_channels', 2)

        super(DeformableAlignment, self).__init__(*args, **kwargs)

        # TODO: remove mask from torch.chunk(out, 3, dim=1)
        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * self.out_channels + self.additional_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )
        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, cond_feat, flow):
        out = self.conv_offset(cond_feat)
        # TODO: remove mask (the 3rd output)
        o1, o2, _ = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset = offset + flow.flip(1).repeat(1, offset.size(1) // 2, 1, 1)

        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, 
                                             self.stride, self.padding,
                                             self.dilation)


class BidirectionalPropagation(nn.Module):
    def __init__(self, channel, learnable=True):
        super(BidirectionalPropagation, self).__init__()
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        self.channel = channel
        self.prop_list = ['backward_1', 'forward_1']
        self.learnable = learnable

        if self.learnable:
            for i, module in enumerate(self.prop_list):
                self.deform_align[module] = DeformableAlignment(
                    channel, channel, 3, padding=1, deform_groups=16)

                self.backbone[module] = nn.Sequential(
                    nn.Conv2d(2*channel, channel, 3, 1, 1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(channel, channel, 3, 1, 1),
                )

            self.fuse = nn.Sequential(
                    nn.Conv2d(2*channel, channel, 3, 1, 1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(channel, channel, 3, 1, 1),
                ) 

    def forward(self, x, flows_forward, flows_backward, interpolation='bilinear'):
        """
        x shape : [b, t, c, h, w]
        return [b, t, c, h, w]
        """

        # For backward warping
        # pred_flows_forward for backward feature propagation
        # pred_flows_backward for forward feature propagation
        b, t, c, h, w = x.shape
        feats = {}
        feats['input'] = [x[:, i, :, :, :] for i in range(0, t)]

        prop_list = ['backward_1', 'forward_1']
        cache_list = ['input'] +  prop_list

        for p_i, module_name in enumerate(prop_list):
            feats[module_name] = []

            if 'backward' in module_name:
                frame_idx = range(0, t)
                frame_idx = frame_idx[::-1]
                flow_idx = frame_idx
                flows_for_prop = flows_forward
                flows_for_check = flows_backward
            else:
                frame_idx = range(0, t)
                flow_idx = range(-1, t - 1)
                flows_for_prop = flows_backward
                flows_for_check = flows_forward

            for i, idx in enumerate(frame_idx):
                feat_current = feats[cache_list[p_i]][idx]

                if i == 0:
                    feat_prop = feat_current
                else:
                    flow_prop = flows_for_prop[:, flow_idx[i], :, :, :]
                    flow_check = flows_for_check[:, flow_idx[i], :, :, :]
                    feat_warped = flow_warp(feat_prop, flow_prop.permute(0, 2, 3, 1), interpolation)

                    if self.learnable:
                        cond = torch.cat([feat_current, feat_warped, flow_prop,], dim=1)
                        feat_prop = self.deform_align[module_name](feat_prop, cond, flow_prop)
                    else:
                        feat_prop = feat_warped

                # refine
                if self.learnable:
                    feat = torch.cat([feat_current, feat_prop], dim=1)
                    feat_prop = feat_prop + self.backbone[module_name](feat)
                    # feat_prop = self.backbone[module_name](feat_prop)

                feats[module_name].append(feat_prop)

            # end for
            if 'backward' in module_name:
                feats[module_name] = feats[module_name][::-1]
        outputs_b = torch.stack(feats['backward_1'], dim=1).reshape(-1, c, h, w)
        outputs_f = torch.stack(feats['forward_1'], dim=1).reshape(-1, c, h, w)

        if self.learnable:
            outputs = self.fuse(torch.cat([outputs_b, outputs_f], dim=1)) + x.reshape(-1, c, h, w)
        else:
            outputs = outputs_f

        return outputs_b.view(b, -1, c, h, w), outputs_f.view(b, -1, c, h, w), \
               outputs.view(b, -1, c, h, w)


class BidirectionalPropagationChecked(nn.Module):
    def __init__(self, channel, learnable=True):
        super(BidirectionalPropagationChecked, self).__init__()
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        self.channel = channel
        self.prop_list = ['backward_1', 'forward_1']
        self.learnable = learnable

        if self.learnable:
            for i, module in enumerate(self.prop_list):
                self.deform_align[module] = DeformableAlignment(
                    channel, channel, 3, padding=1, deform_groups=16, additional_channels=3)

                self.backbone[module] = nn.Sequential(
                    nn.Conv2d(2*channel, channel, 3, 1, 1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(channel, channel, 3, 1, 1),
                )

            self.fuse = nn.Sequential(
                    nn.Conv2d(2*channel, channel, 3, 1, 1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(channel, channel, 3, 1, 1),
                ) 

    def forward(self, x, flows_forward, flows_backward, interpolation='bilinear'):
        """
        x shape : [b, t, c, h, w]
        return [b, t, c, h, w]
        """

        # For backward warping
        # pred_flows_forward for backward feature propagation
        # pred_flows_backward for forward feature propagation
        b, t, c, h, w = x.shape
        feats = {}
        feats['input'] = [x[:, i, :, :, :] for i in range(0, t)]

        prop_list = ['backward_1', 'forward_1']
        cache_list = ['input'] +  prop_list

        for p_i, module_name in enumerate(prop_list):
            feats[module_name] = []

            if 'backward' in module_name:
                frame_idx = range(0, t)
                frame_idx = frame_idx[::-1]
                flow_idx = frame_idx
                flows_for_prop = flows_forward
                flows_for_check = flows_backward
            else:
                frame_idx = range(0, t)
                flow_idx = range(-1, t - 1)
                flows_for_prop = flows_backward
                flows_for_check = flows_forward

            for i, idx in enumerate(frame_idx):
                feat_current = feats[cache_list[p_i]][idx]

                if i == 0:
                    feat_prop = feat_current
                else:
                    flow_prop = flows_for_prop[:, flow_idx[i], :, :, :]
                    flow_check = flows_for_check[:, flow_idx[i], :, :, :]
                    flow_vaild_mask = fbConsistencyCheck(flow_prop, flow_check)
                    feat_warped = flow_warp(feat_prop, flow_prop.permute(0, 2, 3, 1), interpolation)

                    if self.learnable:
                        cond = torch.cat([feat_current, feat_warped, flow_prop, flow_vaild_mask], dim=1)
                        feat_prop = self.deform_align[module_name](feat_prop, cond, flow_prop)
                    else:
                        feat_prop = feat_warped * flow_vaild_mask + (1 - flow_vaild_mask)  * feat_current

                # refine
                if self.learnable:
                    feat = torch.cat([feat_current, feat_prop], dim=1)
                    feat_prop = feat_prop + self.backbone[module_name](feat)
                    # feat_prop = self.backbone[module_name](feat_prop)

                feats[module_name].append(feat_prop)

            # end for
            if 'backward' in module_name:
                feats[module_name] = feats[module_name][::-1]
        outputs_b = torch.stack(feats['backward_1'], dim=1).reshape(-1, c, h, w)
        outputs_f = torch.stack(feats['forward_1'], dim=1).reshape(-1, c, h, w)

        if self.learnable:
            outputs = self.fuse(torch.cat([outputs_b, outputs_f], dim=1)) + x.reshape(-1, c, h, w)
        else:
            outputs = outputs_f

        return outputs_b.view(b, -1, c, h, w), outputs_f.view(b, -1, c, h, w), \
               outputs.view(b, -1, c, h, w)


class deconv(nn.Module):
    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_size=3,
                 padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel,
                              output_channel,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=padding)

    def forward(self, x):
        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        return self.conv(x)
