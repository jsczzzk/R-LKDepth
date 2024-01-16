import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath
from update import BasicUpdateBlock
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from einops import rearrange

    
def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=True):
    if type(kernel_size) is int:
        use_large_impl = kernel_size > 5
    else:
        assert len(kernel_size) == 2 and kernel_size[0] == kernel_size[1]
        use_large_impl = kernel_size[0] > 5
    has_large_impl = 'LARGE_KERNEL_CONV_IMPL' in os.environ
    if has_large_impl and in_channels == out_channels and out_channels == groups and use_large_impl and stride == 1 and padding == kernel_size // 2 and dilation == 1:
        sys.path.append(os.environ['LARGE_KERNEL_CONV_IMPL'])
        #   Please follow the instructions https://github.com/DingXiaoH/RepLKNet-pytorch/blob/main/README.md
        #   export LARGE_KERNEL_CONV_IMPL=absolute_path_to_where_you_cloned_the_example (i.e., depthwise_conv2d_implicit_gemm.py)
        # TODO more efficient PyTorch implementations of large-kernel convolutions. Pull requests are welcomed.
        # Or you may try MegEngine. We have integrated an efficient implementation into MegEngine and it will automatically use it.
        # from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
        # return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups, bias=bias)
    else:
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups, bias=bias)

use_sync_bn = False

def enable_sync_bn():
    global use_sync_bn
    use_sync_bn = True

def get_bn(channels):
    if use_sync_bn:
        return nn.SyncBatchNorm(channels)
    else:
        return nn.BatchNorm2d(channels)

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True))
    # result.add_module('bn', get_bn(out_channels))
    # result.conv.padding = padding
    return result

def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, groups=groups, dilation=dilation)
    result.add_module('nonlinear', nn.GELU())
    return result

def fuse_bn(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std

class ReparamLargeKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups,
                 small_kernel,
                 small_kernel_merged=False):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        # We assume the conv does not change the feature map size, so padding = k//2. Otherwise, you may configure padding as you wish, and change the padding of small_conv accordingly.
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=1, groups=groups)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=small_kernel,
                                             stride=stride, padding=small_kernel//2, groups=groups, dilation=1)

    def forward(self, inputs):
        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)
        return out

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            #   add to the central part
            eq_k += nn.functional.pad(small_k, [(self.kernel_size - self.small_kernel) // 2] * 4)
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = get_conv2d(in_channels=self.lkb_origin.conv.in_channels,
                                     out_channels=self.lkb_origin.conv.out_channels,
                                     kernel_size=self.lkb_origin.conv.kernel_size, stride=self.lkb_origin.conv.stride,
                                     padding=((self.lkb_origin.conv.kernel_size)[0]//2, (self.lkb_origin.conv.kernel_size)[0]//2), dilation=self.lkb_origin.conv.dilation,
                                     groups=self.lkb_origin.conv.groups, bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')


class ConvFFN(nn.Module):

    def __init__(self, in_channels, internal_channels, out_channels, drop_path):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.preffn_bn = get_bn(in_channels)
        self.pw1 = conv_bn(in_channels=in_channels, out_channels=internal_channels, kernel_size=1, stride=1, padding=0, groups=1)
        self.pw2 = conv_bn(in_channels=internal_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=1)
        self.nonlinear = nn.GELU()

    def forward(self, x):
        # out = self.preffn_bn(x)
        out = self.pw1(x)
        out = self.nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)


class RepLKBlock(nn.Module):

    def __init__(self, in_channels, dw_channels, block_lk_size, small_kernel, drop_path, small_kernel_merged=False):
        super().__init__()
        self.pw1 = conv_bn_relu(in_channels, dw_channels, 1, 1, 0, groups=1)
        self.pw2 = conv_bn(dw_channels, in_channels, 1, 1, 0, groups=1)
        self.large_kernel = ReparamLargeKernelConv(in_channels=dw_channels, out_channels=dw_channels, kernel_size=block_lk_size,
                                                  stride=1, groups=dw_channels, small_kernel=small_kernel, small_kernel_merged=small_kernel_merged)
        self.lk_nonlinear = nn.GELU()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        print('drop path:', self.drop_path)

    def forward(self, x):
        # out = self.prelkb_bn(x)
        out = self.pw1(x)
        out = self.large_kernel(out)
        out = self.lk_nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)



class RepLKNetStage(nn.Module):

    def __init__(self, channels, num_blocks, stage_lk_size, drop_path,
                 small_kernel, dw_ratio=1, ffn_ratio=4,
                 use_checkpoint=False,      # train with torch.utils.checkpoint to save memory
                 small_kernel_merged=False,
                 norm_intermediate_features=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        blks = []
        for i in range(num_blocks):
            block_drop_path = drop_path[i] if isinstance(drop_path, list) else drop_path
            #   Assume all RepLK Blocks within a stage share the same lk_size. You may tune it on your own model.
            replk_block = RepLKBlock(in_channels=channels, dw_channels=int(channels * dw_ratio), block_lk_size=stage_lk_size,
                                     small_kernel=small_kernel, drop_path=block_drop_path, small_kernel_merged=small_kernel_merged)
            convffn_block = ConvFFN(in_channels=channels, internal_channels=int(channels * ffn_ratio), out_channels=channels,
                                    drop_path=block_drop_path)
            blks.append(replk_block)
            blks.append(convffn_block)
        self.blocks = nn.ModuleList(blks)


    def forward(self, x):
        output = []
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)   # Save training memory
                if blk.__class__.__name__ == 'ConvFFN':
                    output.append(x)
            else:
                x = blk(x)
                if blk.__class__.__name__ == 'ConvFFN':
                    output.append(x)
        return output



class SepConvGRU(nn.Module):
    def __init__(self):
        super(SepConvGRU, self).__init__()
        hidden_dim = 128
        catt = 128 *2

        self.convz1 = nn.Conv2d(catt, hidden_dim, (1, 3), padding=(0, 1))
        self.convr1 = nn.Conv2d(catt, hidden_dim, (1, 3), padding=(0, 1))
        self.convq1 = nn.Conv2d(catt, hidden_dim, (1, 3), padding=(0, 1))

        self.convz2 = nn.Conv2d(catt, hidden_dim, (3, 1), padding=(1, 0))
        self.convr2 = nn.Conv2d(catt, hidden_dim, (3, 1), padding=(1, 0))
        self.convq2 = nn.Conv2d(catt, hidden_dim, (3, 1), padding=(1, 0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h

class R_LKDepth(nn.Module):
    def __init__(self):
        super(R_LKDepth, self).__init__()

        
        self.pro =  torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512, out_channels=768, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(768)
            )
        self.pro2 =  torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128*3, out_channels=128*3, kernel_size=1, stride=1, padding=0, bias=True),
            )
        
        self.x1 =  torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=128*2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GELU(),
            torch.nn.Conv2d(in_channels=128*2, out_channels=128, kernel_size=1, stride=1, padding=0, bias=True),
            )
        
        self.x2 =  torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=128*2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GELU(),
            torch.nn.Conv2d(in_channels=128*2, out_channels=128, kernel_size=1, stride=1, padding=0, bias=True),
            )
        

        self.x3 =  torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=128*2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GELU(),
            torch.nn.Conv2d(in_channels=128*2, out_channels=128, kernel_size=1, stride=1, padding=0, bias=True),
            )
        
        self.trans = nn.ModuleList([self.x1, self.x2, self.x3])
        
       
            

        self.sigmoid = nn.Sigmoid()
        self.gruc = SepConvGRU()
        self.update_block = BasicUpdateBlock()

    def upsample_depth(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 1, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 1, 8 * H, 8 * W)

    def forward(self, features, iters=6):
        """ Estimate depth for a single image """

        x1, x2, x3 = features
        x3 = self.pro(x3)
        window_sizes = [1, 2, 4]
        b,c,h,w = x3.shape
        xs  =  torch.split(x3, [256,256,256], dim=1)
        ys = []
        for idx, x_ in enumerate(xs):
                wsize = window_sizes[idx]
                q, v = rearrange(
                    x_, 'b (qv c) (h dh) (w dw) -> qv (b h w) (dh dw) c', 
                    qv=2, dh=wsize, dw=wsize
                )

                atn = (q @ q.transpose(-2, -1)) 
                atn = atn.softmax(dim=-1)
                y_ = (atn @ v)
                y_ = rearrange(
                    y_, '(b h w) (dh dw) c-> b (c) (h dh) (w dw)', 
                    h=h//wsize, w=w//wsize, dh=wsize, dw=wsize
                )
                ys.append(y_)
        
        y = torch.cat(ys, dim=1)            
        y = self.pro2(y)
        ys  =  torch.split(y, [128,128,128], dim=1)


        disp_predictions = {}
        dispFea = torch.zeros([b, 1, h, w], requires_grad=True).to(x1.device)
        net = torch.zeros([b, 256, h, w], requires_grad=True).to(x1.device)

        for itr in range(iters):
            if itr in [0]:
                corr = torch.tanh( ys[itr])
            elif itr in [1]:
                corrh = corr
                corr = self.gruc(corrh, ys[itr])
            elif itr in [2]:
                corrh = corr
                corr = self.gruc(corrh,  ys[itr])
            net, up_mask, delta_disp = self.update_block(net,  corr, dispFea)
            dispFea = dispFea + delta_disp

            disp = self.sigmoid(dispFea)
            # upsample predictions
   
            if self.training:
                disp_up = self.upsample_depth(disp, up_mask)
                disp_predictions[("disp_up", itr)] = disp_up
            else:
                # if (iters-1)==itr:
                disp_up = self.upsample_depth(disp, up_mask)
                disp_predictions[("disp_up", itr)] = disp_up


        return disp_predictions