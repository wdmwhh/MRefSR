import mmsr.models.archs.arch_util as arch_util
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d


class DynAgg(ModulatedDeformConv2d):
    '''
    Use other features to generate offsets and masks.
    Intialized the offset with precomputed non-local offset.
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 groups=1,
                 deform_groups=1,
                 extra_offset_mask=True):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, groups, deform_groups)
        self.extra_offset_mask = extra_offset_mask
        channels_ = self.deform_groups * 3 * self.kernel_size[
            0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            channels_,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, x, pre_offset):
        '''
        Args:
            pre_offset: precomputed_offset. Size: [b, 9, h, w, 2]
        '''
        if self.extra_offset_mask:
            # x = [input, features]
            out = self.conv_offset_mask(x[1])
            x = x[0]
        else:
            out = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        # repeat pre_offset along dim1, shape: [b, 9*groups, h, w, 2]
        pre_offset = pre_offset.repeat([1, self.deform_groups, 1, 1, 1])
        # the order of offset is [y, x, y, x, ..., y, x]
        pre_offset_reorder = torch.zeros_like(offset)
        # add pre_offset on y-axis
        pre_offset_reorder[:, 0::2, :, :] = pre_offset[:, :, :, :, 1]
        # add pre_offset on x-axis
        pre_offset_reorder[:, 1::2, :, :] = pre_offset[:, :, :, :, 0]
        offset = offset + pre_offset_reorder
        # print(offset.size())
        mask = torch.sigmoid(mask)

        offset_mean = torch.mean(torch.abs(offset - pre_offset_reorder))
        if offset_mean > 100:
            logger.warning(
                'Offset mean is {}, larger than 100.'.format(offset_mean))
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                           self.stride, self.padding, self.dilation,
                           self.groups, self.deform_groups)


class ContentExtractor(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, nf=64, n_blocks=16):
        super(ContentExtractor, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.body = arch_util.make_layer(
            arch_util.ResidualBlockNoBN, n_blocks, nf=nf)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        arch_util.default_init_weights([self.conv_first], 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        feat = self.body(feat)

        return feat


class MRAPARestorationNet(nn.Module):

    def __init__(self, ngf=64, n_blocks=16, groups=8):
        super(MRAPARestorationNet, self).__init__()
        self.content_extractor = ContentExtractor(
            in_nc=3, out_nc=3, nf=ngf, n_blocks=n_blocks)
        self.dyn_agg_restore = DynamicAggregationRestoration(
            ngf, n_blocks, groups)

        arch_util.srntt_init_weights(self, init_type='normal', init_gain=0.02)
        self.re_init_dcn_offset()

    def re_init_dcn_offset(self):
        self.dyn_agg_restore.small_dyn_agg.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.small_dyn_agg.conv_offset_mask.bias.data.zero_()
        self.dyn_agg_restore.medium_dyn_agg.conv_offset_mask.weight.data.zero_(
        )
        self.dyn_agg_restore.medium_dyn_agg.conv_offset_mask.bias.data.zero_()
        self.dyn_agg_restore.large_dyn_agg.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.large_dyn_agg.conv_offset_mask.bias.data.zero_()

    def forward(self, x, pre_offset_list, img_ref_feat_list):
        """
        Args:
            x (Tensor): the input image of SRNTT.
            maps (dict[Tensor]): the swapped feature maps on relu3_1, relu2_1
                and relu1_1. depths of the maps are 256, 128 and 64
                respectively.
        """

        base = F.interpolate(x, None, 4, 'bilinear', False)
        content_feat = self.content_extractor(x)

        upscale_restore = self.dyn_agg_restore(content_feat, pre_offset_list,
                                               img_ref_feat_list)
        return upscale_restore + base


class DynamicAggregationRestoration(nn.Module):

    def __init__(self, ngf=64, n_blocks=16, groups=8):
        super(DynamicAggregationRestoration, self).__init__()

        # dynamic aggregation module for relu3_1 reference feature
        self.small_offset_conv1 = nn.Conv2d(
            ngf + 256, 256, 3, 1, 1, bias=True)  # concat for diff
        self.small_offset_conv2 = nn.Conv2d(256, 256, 3, 1, 1, bias=True)
        self.small_dyn_agg = DynAgg(
            256,
            256,
            3,
            stride=1,
            padding=1,
            dilation=1,
            deform_groups=groups,
            extra_offset_mask=True)

        # for small scale restoration
        self.head_small = MRAPAFusion(nf=ngf, ref_nf=256)
        self.body_small = arch_util.make_layer(
            arch_util.ResidualBlockNoBN, n_blocks, nf=ngf)
        self.tail_small = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2), nn.LeakyReLU(0.1, True))

        # dynamic aggregation module for relu2_1 reference feature
        self.medium_offset_conv1 = nn.Conv2d(
            ngf + 128, 128, 3, 1, 1, bias=True)
        self.medium_offset_conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=True)
        self.medium_dyn_agg = DynAgg(
            128,
            128,
            3,
            stride=1,
            padding=1,
            dilation=1,
            deform_groups=groups,
            extra_offset_mask=True)

        # for medium scale restoration
        self.head_medium = MRAPAFusion(nf=ngf, ref_nf=128)
        self.body_medium = arch_util.make_layer(
            arch_util.ResidualBlockNoBN, n_blocks, nf=ngf)
        self.tail_medium = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2), nn.LeakyReLU(0.1, True))

        # dynamic aggregation module for relu1_1 reference feature
        self.large_offset_conv1 = nn.Conv2d(ngf + 64, 64, 3, 1, 1, bias=True)
        self.large_offset_conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.large_dyn_agg = DynAgg(
            64,
            64,
            3,
            stride=1,
            padding=1,
            dilation=1,
            deform_groups=groups,
            extra_offset_mask=True)

        # for large scale
        self.head_large = MRAPAFusion(nf=ngf, ref_nf=64)
        self.body_large = arch_util.make_layer(
            arch_util.ResidualBlockNoBN, n_blocks, nf=ngf)
        self.tail_large = nn.Sequential(
            nn.Conv2d(ngf, ngf // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(ngf // 2, 3, kernel_size=3, stride=1, padding=1))

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, pre_offset_list, img_ref_feat_list):
        # dynamic aggregation for relu3_1 reference feature
        relu3_swapped_feat_list = []
        for pre_offset, img_ref_feat in zip(pre_offset_list, img_ref_feat_list):
            relu3_offset = torch.cat([x, img_ref_feat['relu3_1']], 1)
            relu3_offset = self.lrelu(self.small_offset_conv1(relu3_offset))
            relu3_offset = self.lrelu(self.small_offset_conv2(relu3_offset))
            relu3_swapped_feat = self.lrelu(
                self.small_dyn_agg([img_ref_feat['relu3_1'], relu3_offset],
                                   pre_offset['relu3_1']))
            relu3_swapped_feat_list.append(relu3_swapped_feat)
        # small scale
        h = self.head_small(x, relu3_swapped_feat_list)
        h = self.body_small(h) + x
        x = self.tail_small(h)

        # dynamic aggregation for relu2_1 reference feature
        relu2_swapped_feat_list = []
        for pre_offset, img_ref_feat in zip(pre_offset_list, img_ref_feat_list):
            relu2_offset = torch.cat([x, img_ref_feat['relu2_1']], 1)
            relu2_offset = self.lrelu(self.medium_offset_conv1(relu2_offset))
            relu2_offset = self.lrelu(self.medium_offset_conv2(relu2_offset))
            relu2_swapped_feat = self.lrelu(
                self.medium_dyn_agg([img_ref_feat['relu2_1'], relu2_offset],
                                    pre_offset['relu2_1']))
            relu2_swapped_feat_list.append(relu2_swapped_feat)
        # medium scale
        h = self.head_medium(x, relu2_swapped_feat_list)
        h = self.body_medium(h) + x
        x = self.tail_medium(h)

        # dynamic aggregation for relu1_1 reference feature
        relu1_swapped_feat_list = []
        for pre_offset, img_ref_feat in zip(pre_offset_list, img_ref_feat_list):
            relu1_offset = torch.cat([x, img_ref_feat['relu1_1']], 1)
            relu1_offset = self.lrelu(self.large_offset_conv1(relu1_offset))
            relu1_offset = self.lrelu(self.large_offset_conv2(relu1_offset))
            relu1_swapped_feat = self.lrelu(
                self.large_dyn_agg([img_ref_feat['relu1_1'], relu1_offset],
                                   pre_offset['relu1_1']))
            relu1_swapped_feat_list.append(relu1_swapped_feat)
        # large scale
        h = self.head_large(x, relu1_swapped_feat_list)
        h = self.body_large(h) + x
        x = self.tail_large(h)

        return x


class MRAPAFusion(nn.Module):
    """Temporal Spatial Attention (TSA) fusion module. It is used in EDVRNet.
    Args:
        nf (int): Number of the channels of middle features.
            Default: 64.
        ref_nf (int): Number of the channels of middle features.
            Default: 256.
    """

    def __init__(self,
                 nf=64,
                 ref_nf=256):
        super().__init__()

        # multi-ref attention (before fusion conv)
        self.patch_size = 3
        channels = ref_nf
        self.conv_emb1 = nn.Sequential(
            nn.Conv2d(nf, channels, 1),
            nn.PReLU())
        self.conv_emb2 = nn.Sequential(
            nn.Conv2d(ref_nf, channels, 
                      self.patch_size, 1, self.patch_size//2),
            nn.PReLU())
        self.conv_ass = nn.Conv2d(ref_nf, channels*2,
            self.patch_size, 1, self.patch_size//2)
        self.scale = channels**-0.5
        self.feat_fusion = nn.Conv2d(
            nf + channels*2, nf, 1)

        # spatial attention (after fusion conv)
        self.spatial_attn = nn.Conv2d(
            nf + channels*2, channels*2, 1)
        self.spatial_attn_mul1 = nn.Conv2d(
            channels*2, channels*2, 3, padding=1)
        self.spatial_attn_mul2 = nn.Conv2d(
            channels*2, channels*2, 3, padding=1)
        self.spatial_attn_add1 = nn.Conv2d(
            channels*2, channels*2, 3, padding=1)
        self.spatial_attn_add2 = nn.Conv2d(
            channels*2, channels*2, 3, padding=1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def spatial_padding(self, feats):
        _, _, h, w = feats.size()
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        feats = F.pad(feats, [0, pad_w, 0, pad_h], mode='reflect')
        return feats

    def forward(self, target, refs):
        n, _, h_input, w_input = target.size()
        t = len(refs)
        
        target = self.spatial_padding(target)
        refs = torch.stack(refs, dim=1).flatten(0, 1)
        refs = self.spatial_padding(refs)
        # multi-ref attention
        embedding_target = self.conv_emb1(target) * self.scale  # (n, c, h, w)
        embedding_target = embedding_target.permute(0, 2, 3, 1).unsqueeze(3)  # (n, h, w, 1, c)
        embedding_target = embedding_target.contiguous().flatten(0, 2)  # (n*h*w, 1, c)
        emb = self.conv_emb2(refs).unflatten(0, (n, t))  # (n, t, c, h, w)
        emb = emb.permute(0, 3, 4, 2, 1)  # (n, h, w, c, t)
        emb = emb.contiguous().flatten(0, 2)  # (n*h*w, c, t)
        ass = self.conv_ass(refs).unflatten(0, (n, t))  # (n, t, c*2, h, w)
        ass = ass.permute(0, 3, 4, 1, 2)  # (n, h, w, t, c*2)
        ass = ass.contiguous().flatten(0, 2)  # (n*h*w, t, c*2)

        corr_prob = torch.matmul(embedding_target, emb)  # (n*h*w, 1, t)
        corr_prob = F.softmax(corr_prob, dim=2)
        refs = torch.matmul(corr_prob, ass).squeeze(1)  # (n*h*w, c*2)
        refs = refs.unflatten(0, (n, *target.shape[-2:]))  # (n, h, w, c*2)
        refs = refs.permute(0, 3, 1, 2).contiguous()  #(n, c*2, h, w)

        del embedding_target, emb, ass, corr_prob

        # spatial attention
        attn = self.lrelu(self.spatial_attn(torch.cat([target, refs], dim=1)))
        attn_mul = self.spatial_attn_mul2(self.lrelu(self.spatial_attn_mul1(attn)))
        attn_add = self.spatial_attn_add2(self.lrelu(self.spatial_attn_add1(attn)))
        attn_mul = torch.sigmoid(attn_mul)

        # after initialization, * 2 makes (attn_mul * 2) to be close to 1.
        refs = refs * attn_mul * 2 + attn_add

        # fusion
        feat = self.lrelu(self.feat_fusion(torch.cat([target, refs], dim=1)))
        return feat[:, :, :h_input, :w_input]
