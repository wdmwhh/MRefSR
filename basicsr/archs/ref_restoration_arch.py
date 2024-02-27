import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from basicsr.utils.registry import ARCH_REGISTRY

from .arch_util import ResidualBlockNoBN, default_init_weights, make_layer, srntt_init_weights


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
        self.body = make_layer(
            ResidualBlockNoBN, n_blocks, num_feat=nf)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first], 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        feat = self.body(feat)

        return feat


@ARCH_REGISTRY.register()
class RestorationNet(nn.Module):

    def __init__(self, ngf=64, n_blocks=16, groups=8):
        super(RestorationNet, self).__init__()
        self.content_extractor = ContentExtractor(
            in_nc=3, out_nc=3, nf=ngf, n_blocks=n_blocks)
        self.dyn_agg_restore = DynamicAggregationRestoration(
            ngf, n_blocks, groups)

        srntt_init_weights(self, init_type='normal', init_gain=0.02)
        self.re_init_dcn_offset()

    def re_init_dcn_offset(self):
        self.dyn_agg_restore.small_dyn_agg.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.small_dyn_agg.conv_offset_mask.bias.data.zero_()
        self.dyn_agg_restore.medium_dyn_agg.conv_offset_mask.weight.data.zero_(
        )
        self.dyn_agg_restore.medium_dyn_agg.conv_offset_mask.bias.data.zero_()
        self.dyn_agg_restore.large_dyn_agg.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.large_dyn_agg.conv_offset_mask.bias.data.zero_()

    def forward(self, x, pre_offset, img_ref_feat):
        """
        Args:
            x (Tensor): the input image of SRNTT.
            maps (dict[Tensor]): the swapped feature maps on relu3_1, relu2_1
                and relu1_1. depths of the maps are 256, 128 and 64
                respectively.
        """

        base = F.interpolate(x, None, 4, 'bilinear', False)
        content_feat = self.content_extractor(x)

        upscale_restore = self.dyn_agg_restore(content_feat, pre_offset,
                                               img_ref_feat)
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
        self.head_small = nn.Sequential(
            nn.Conv2d(ngf + 256, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True))
        self.body_small = make_layer(
            ResidualBlockNoBN, n_blocks, num_feat=ngf)
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
        self.head_medium = nn.Sequential(
            nn.Conv2d(ngf + 128, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True))
        self.body_medium = make_layer(
            ResidualBlockNoBN, n_blocks, num_feat=ngf)
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
        self.head_large = nn.Sequential(
            nn.Conv2d(ngf + 64, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True))
        self.body_large = make_layer(
            ResidualBlockNoBN, n_blocks, num_feat=ngf)
        self.tail_large = nn.Sequential(
            nn.Conv2d(ngf, ngf // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(ngf // 2, 3, kernel_size=3, stride=1, padding=1))

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, pre_offset, img_ref_feat):
        # dynamic aggregation for relu3_1 reference feature
        relu3_offset = torch.cat([x, img_ref_feat['relu3_1']], 1)
        relu3_offset = self.lrelu(self.small_offset_conv1(relu3_offset))
        relu3_offset = self.lrelu(self.small_offset_conv2(relu3_offset))
        relu3_swapped_feat = self.lrelu(
            self.small_dyn_agg([img_ref_feat['relu3_1'], relu3_offset],
                               pre_offset['relu3_1']))
        # small scale
        h = torch.cat([x, relu3_swapped_feat], 1)
        h = self.head_small(h)
        h = self.body_small(h) + x
        x = self.tail_small(h)

        # dynamic aggregation for relu2_1 reference feature
        relu2_offset = torch.cat([x, img_ref_feat['relu2_1']], 1)
        relu2_offset = self.lrelu(self.medium_offset_conv1(relu2_offset))
        relu2_offset = self.lrelu(self.medium_offset_conv2(relu2_offset))
        relu2_swapped_feat = self.lrelu(
            self.medium_dyn_agg([img_ref_feat['relu2_1'], relu2_offset],
                                pre_offset['relu2_1']))
        # medium scale
        h = torch.cat([x, relu2_swapped_feat], 1)
        h = self.head_medium(h)
        h = self.body_medium(h) + x
        x = self.tail_medium(h)

        # dynamic aggregation for relu1_1 reference feature
        relu1_offset = torch.cat([x, img_ref_feat['relu1_1']], 1)
        relu1_offset = self.lrelu(self.large_offset_conv1(relu1_offset))
        relu1_offset = self.lrelu(self.large_offset_conv2(relu1_offset))
        relu1_swapped_feat = self.lrelu(
            self.large_dyn_agg([img_ref_feat['relu1_1'], relu1_offset],
                               pre_offset['relu1_1']))
        # large scale
        h = torch.cat([x, relu1_swapped_feat], 1)
        h = self.head_large(h)
        h = self.body_large(h) + x
        x = self.tail_large(h)

        return x
