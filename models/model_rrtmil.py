from torch import nn
from timm.models.layers import DropPath
from utils.utils import initialize_weights # type: ignore
import torch
import torch.nn.functional as F
import torch, einops
import numpy as np
from timm.models.layers import trunc_normal_
import math
from math import ceil
from torch import nn, einsum
from einops import rearrange, reduce



"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x 
    
""""""


""""""
class PositionEmbedding(nn.Module):
    def __init__(self, size, dim=512):
        super().__init__()
        self.size=size
        self.pe = nn.Embedding(size+1, dim, padding_idx=0)
        self.pos_ids = torch.arange(1, size+1, dtype=torch.long).cuda()
        
    def forward(self, emb):
        device = emb.device
        b, n, *_ = emb.shape
        pos_ids = self.pos_ids
        if n > self.size:
            zeros = torch.zeros(n-self.size, dtype=torch.long, device=device)
            pos_ids = torch.cat([pos_ids, zeros])
        pos_ids = einops.repeat(pos_ids, 'n -> b n', b=b)
        pos_emb = self.pe(pos_ids) # [b n pe_dim]
        embeddings = torch.cat([emb, pos_emb], dim=-1)
        return embeddings
        
class PPEG(nn.Module):
    def __init__(self, dim=512,k=7,conv_1d=False,bias=True):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (k,1), 1, (k//2,0), groups=dim,bias=bias)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (5,1), 1, (5//2,0), groups=dim,bias=bias)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (3,1), 1, (3//2,0), groups=dim,bias=bias)

    def forward(self, x):
        B, N, C = x.shape

        # padding
        H, W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))
        
        add_length = H * W - N
        # if add_length >0:
        x = torch.cat([x, x[:,:add_length,:]],dim = 1) 

        if H < 7:
            H,W = 7,7
            zero_pad = H * W - (N+add_length)
            x = torch.cat([x, torch.zeros((B,zero_pad,C),device=x.device)],dim = 1)
            add_length += zero_pad

        # H, W = int(N**0.5),int(N**0.5)
        # cls_token, feat_token = x[:, 0], x[:, 1:]
        # feat_token = x
        cnn_feat = x.transpose(1, 2).view(B, C, H, W)

        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        # print(add_length)
        if add_length >0:
            x = x[:,:-add_length]
        # x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class PEG(nn.Module):
    def __init__(self, dim=512,k=7,bias=True,conv_1d=False):
        super(PEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (k,1), 1, (k//2,0), groups=dim,bias=bias)

    def forward(self, x):
        B, N, C = x.shape

        # padding
        H, W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))
        add_length = H * W - N
        x = torch.cat([x, x[:,:add_length,:]],dim = 1)

        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat

        x = x.flatten(2).transpose(1, 2)
        if add_length >0:
            x = x[:,:-add_length]

        # x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class SINCOS(nn.Module):
    def __init__(self,embed_dim=512):
        super(SINCOS, self).__init__()
        self.embed_dim = embed_dim
        self.pos_embed = self.get_2d_sincos_pos_embed(embed_dim, 8)
    def get_1d_sincos_pos_embed_from_grid(self,embed_dim, pos):
        """
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        out: (M, D)
        """
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

        emb_sin = np.sin(out) # (M, D/2)
        emb_cos = np.cos(out) # (M, D/2)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb

    def get_2d_sincos_pos_embed_from_grid(self,embed_dim, grid):
        assert embed_dim % 2 == 0

        # use half of dimensions to encode grid_h
        emb_h = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

        emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
        return emb

    def get_2d_sincos_pos_embed(self,embed_dim, grid_size, cls_token=False):
        """
        grid_size: int of the grid height and width
        return:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
        """
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)

        grid = grid.reshape([2, 1, grid_size, grid_size])
        pos_embed = self.get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        if cls_token:
            pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
        return pos_embed

    def forward(self, x):
        #B, N, C = x.shape
        B,H,W,C = x.shape
        # # padding
        # H, W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))
        # add_length = H * W - N
        # x = torch.cat([x, x[:,:add_length,:]],dim = 1)

        # pos_embed = torch.zeros(1, H * W + 1, self.embed_dim)
        # pos_embed = self.get_2d_sincos_pos_embed(pos_embed.shape[-1], int(H), cls_token=True)
        #pos_embed = torch.from_numpy(self.pos_embed).float().unsqueeze(0).to(x.device)

        pos_embed = torch.from_numpy(self.pos_embed).float().to(x.device)

        # print(pos_embed.size())
        # print(x.size())
        x = x + pos_embed.unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
        

        #x = x + pos_embed[:, 1:, :]

        # if add_length >0:
        #     x = x[:,:-add_length]

        return x

class APE(nn.Module):
    def __init__(self,embed_dim=512,num_patches=64):
        super(APE, self).__init__()
        self.absolute_pos_embed = nn.Parameter(torch.zeros( num_patches, embed_dim))
        trunc_normal_(self.absolute_pos_embed, std=.02)
    
    def forward(self, x):
        B,H,W,C = x.shape
        return x + self.absolute_pos_embed.unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)

class RPE(nn.Module):
    def __init__(self,num_heads=8,region_size=(8,8)):
        super(RPE, self).__init__()
        self.region_size = region_size

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * region_size[0] - 1) * (2 * region_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the region
        coords_h = torch.arange(region_size[0])
        coords_w = torch.arange(region_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += region_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += region_size[1] - 1
        relative_coords[:, :, 0] *= 2 * region_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)
    
    def forward(self, x):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.region_size[0] * self.region_size[1], self.region_size[0] * self.region_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        print(relative_position_bias.size())

        return x + self.absolute_pos_embed.unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
""""""    

""""""
# --------------------------------------------------------
# Modified by Swin@Microsoft
# --------------------------------------------------------

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def region_partition(x, region_size):
    """
    Args:
        x: (B, H, W, C)
        region_size (int): region size
    Returns:
        regions: (num_regions*B, region_size, region_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // region_size, region_size, W // region_size, region_size, C)
    regions = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, region_size, region_size, C)
    return regions

def region_reverse(regions, region_size, H, W):
    """
    Args:
        regions: (num_regions*B, region_size, region_size, C)
        region_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(regions.shape[0] / (H * W / region_size / region_size))
    x = regions.view(B, H // region_size, W // region_size, region_size, region_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class InnerAttention(nn.Module):
    def __init__(self, dim, head_dim=None, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,epeg=True,epeg_k=15,epeg_2d=False,epeg_bias=True,epeg_type='attn'):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        if head_dim is None:
            head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, head_dim * num_heads * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(head_dim * num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.epeg_2d = epeg_2d
        self.epeg_type = epeg_type
        if epeg:
            padding = epeg_k // 2
            if epeg_2d:
                if epeg_type == 'attn':
                    self.pe = nn.Conv2d(num_heads, num_heads, epeg_k, padding = padding, groups = num_heads, bias = epeg_bias)
                else:
                    self.pe = nn.Conv2d(head_dim * num_heads, head_dim * num_heads, epeg_k, padding = padding, groups = head_dim * num_heads, bias = epeg_bias)
            else:
                if epeg_type == 'attn':
                    self.pe = nn.Conv2d(num_heads, num_heads, (epeg_k, 1), padding = (padding, 0), groups = num_heads, bias = epeg_bias)
                else:
                    self.pe = nn.Conv2d(head_dim *num_heads, head_dim *num_heads, (epeg_k, 1), padding = (padding, 0), groups = head_dim *num_heads, bias = epeg_bias)
        else:
            self.pe = None

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_regions*B, N, C)
        """
        B_, N, C = x.shape

        # x = self.pe(x)

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.pe is not None and self.epeg_type == 'attn':
            pe = self.pe(attn)
            attn = attn+pe

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        if self.pe is not None and self.epeg_type == 'value_bf':
            # B,H,N,C -> B,HC,N-0.5,N-0.5 
            pe = self.pe(v.permute(0,3,1,2).reshape(B_,C,int(np.ceil(np.sqrt(N))),int(np.ceil(np.sqrt(N)))))
            #pe = torch.einsum('ahbd->abhd',pe).flatten(-2,-1)
            v = v + pe.reshape(B_,self.num_heads, self.head_dim,N).permute(0,1,3,2)

        # print(v.size())

        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.num_heads*self.head_dim)
        
        if self.pe is not None and self.epeg_type == 'value_af':
            #print(v.size())
            pe = self.pe(v.permute(0,3,1,2).reshape(B_,C,int(np.ceil(np.sqrt(N))),int(np.ceil(np.sqrt(N)))))
            # print(pe.size())
            # print(v.size())
            x = x + pe.reshape(B_,self.num_heads*self.head_dim,N).transpose(-1,-2)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    #def extra_repr(self) -> str:
    #    return f'dim={self.dim}, region_size={self.region_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 region with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class RegionAttntion(nn.Module):
    def __init__(self, dim, head_dim=None,num_heads=8, region_size=0, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., region_num=8,epeg=False,min_region_num=0,min_region_ratio=0.,region_attn='native',**kawrgs):
        super().__init__()
 
        self.dim = dim
        self.num_heads = num_heads
        self.region_size = region_size if region_size > 0 else None
        self.region_num = region_num
        self.min_region_num = min_region_num
        self.min_region_ratio = min_region_ratio
        
        if region_attn == 'native':
            self.attn = InnerAttention(
                dim, head_dim=head_dim,num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,epeg=epeg,**kawrgs)
        elif region_attn == 'ntrans':
            self.attn = NystromAttention(
                dim=dim,
                dim_head=head_dim,
                heads=num_heads,
                dropout=drop
            )

    def padding(self,x):
        B, L, C = x.shape
        if self.region_size is not None:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.region_size
            H, W = H+_n, W+_n
            region_num = int(H // self.region_size)
            region_size = self.region_size
        else:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.region_num
            H, W = H+_n, W+_n
            region_size = int(H // self.region_num)
            region_num = self.region_num
        
        add_length = H * W - L

        # if padding much，i will give up region attention. only for ablation
        if (add_length > L / (self.min_region_ratio+1e-8) or L < self.min_region_num):
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % 2
            H, W = H+_n, W+_n
            add_length = H * W - L
            region_size = H
        if add_length > 0:
            x = torch.cat([x, torch.zeros((B,add_length,C),device=x.device)],dim = 1)
        
        return x,H,W,add_length,region_num,region_size

    def forward(self,x,return_attn=False):
        B, L, C = x.shape
        
        # padding
        x,H,W,add_length,region_num,region_size = self.padding(x)

        x = x.view(B, H, W, C)

        # partition regions
        x_regions = region_partition(x, region_size)  # nW*B, region_size, region_size, C

        x_regions = x_regions.view(-1, region_size * region_size, C)  # nW*B, region_size*region_size, C

        # R-MSA
        attn_regions = self.attn(x_regions)  # nW*B, region_size*region_size, C

        # merge regions
        attn_regions = attn_regions.view(-1, region_size, region_size, C)

        x = region_reverse(attn_regions, region_size, H, W)  # B H' W' C

        x = x.view(B, H * W, C)

        if add_length >0:
            x = x[:,:-add_length]

        return x

class CrossRegionAttntion(nn.Module):
    def __init__(self, dim, head_dim=None,num_heads=8, region_size=0, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., region_num=8,epeg=False,min_region_num=0,min_region_ratio=0.,crmsa_k=3,crmsa_mlp=False,region_attn='native',**kawrgs):
        super().__init__()
 
        self.dim = dim
        self.num_heads = num_heads
        self.region_size = region_size if region_size > 0 else None
        self.region_num = region_num
        self.min_region_num = min_region_num
        self.min_region_ratio = min_region_ratio
        
        self.attn = InnerAttention(
            dim, head_dim=head_dim,num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,epeg=epeg,**kawrgs)

        self.crmsa_mlp = crmsa_mlp
        if crmsa_mlp:
            self.phi = [nn.Linear(self.dim, self.dim // 4,bias=False)]
            self.phi += [nn.Tanh()]
            self.phi += [nn.Linear(self.dim // 4, crmsa_k,bias=False)]
            self.phi = nn.Sequential(*self.phi)
        else:
            self.phi = nn.Parameter(
                torch.empty(
                    (self.dim, crmsa_k),
                )
            )
            nn.init.kaiming_uniform_(self.phi, a=math.sqrt(5))

    def padding(self,x):
        B, L, C = x.shape
        if self.region_size is not None:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.region_size
            H, W = H+_n, W+_n
            region_num = int(H // self.region_size)
            region_size = self.region_size
        else:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.region_num
            H, W = H+_n, W+_n
            region_size = int(H // self.region_num)
            region_num = self.region_num
        
        add_length = H * W - L

        # if padding much，i will give up region attention. only for ablation
        if (add_length > L / (self.min_region_ratio+1e-8) or L < self.min_region_num):
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % 2
            H, W = H+_n, W+_n
            add_length = H * W - L
            region_size = H
        if add_length > 0:
            x = torch.cat([x, torch.zeros((B,add_length,C),device=x.device)],dim = 1)
        
        return x,H,W,add_length,region_num,region_size

    def forward(self,x,return_attn=False):
        B, L, C = x.shape
        
        # padding
        x,H,W,add_length,region_num,region_size = self.padding(x)

        x = x.view(B, H, W, C)

        # partition regions
        x_regions = region_partition(x, region_size)  # nW*B, region_size, region_size, C

        x_regions = x_regions.view(-1, region_size * region_size, C)  # nW*B, region_size*region_size, C

        # CR-MSA
        if self.crmsa_mlp:
            logits = self.phi(x_regions).transpose(1,2) # W*B, sW, region_size*region_size
        else:
            logits = torch.einsum("w p c, c n -> w p n", x_regions, self.phi).transpose(1,2) # nW*B, sW, region_size*region_size

        combine_weights = logits.softmax(dim=-1)
        dispatch_weights = logits.softmax(dim=1)

        logits_min,_ = logits.min(dim=-1)
        logits_max,_ = logits.max(dim=-1)
        dispatch_weights_mm = (logits - logits_min.unsqueeze(-1)) / (logits_max.unsqueeze(-1) - logits_min.unsqueeze(-1) + 1e-8)

        attn_regions =  torch.einsum("w p c, w n p -> w n p c", x_regions,combine_weights).sum(dim=-2).transpose(0,1) # sW, nW, C

        if return_attn:
            attn_regions,_attn = self.attn(attn_regions,return_attn)  # sW, nW, C
            attn_regions = attn_regions.transpose(0,1) # nW, sW, C
        else:
            attn_regions = self.attn(attn_regions).transpose(0,1)  # nW, sW, C

        attn_regions = torch.einsum("w n c, w n p -> w n p c", attn_regions, dispatch_weights_mm) # nW, sW, region_size*region_size, C
        attn_regions = torch.einsum("w n p c, w n p -> w n p c", attn_regions, dispatch_weights).sum(dim=1) # nW, region_size*region_size, C

        # merge regions
        attn_regions = attn_regions.view(-1, region_size, region_size, C)

        x = region_reverse(attn_regions, region_size, H, W)  # B H' W' C

        x = x.view(B, H * W, C)

        if add_length >0:
            x = x[:,:-add_length]

        return x
""""""    

""""""
# helper functions

def exists(val):
    return val is not None

def moore_penrose_iter_pinv(x, iters = 6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim = -1)
    row = abs_x.sum(dim = -2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device = device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z

# main attention class

class NystromAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        num_landmarks = 256,
        pinv_iterations = 6,
        residual = True,
        residual_conv_kernel = 33,
        eps = 1e-8,
        dropout = 0.
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)

    def forward(self, x, mask = None, return_attn = False):
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps

        # pad so that sequence can be evenly divided into m landmarks

        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value = 0)

            if exists(mask):
                mask = F.pad(mask, (padding, 0), value = False)

        # derive query, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # set masked positions to 0 in queries, keys, values

        if exists(mask):
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q = q * self.scale

        # generate landmarks by sum reduction, and then calculate mean using the mask

        l = ceil(n / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l = l)
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l = l)

        # calculate landmark mask, and also get sum of non-masked elements in preparation for masked mean

        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, '... (n l) -> ... n', 'sum', l = l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        # masked mean (if mask exists)

        q_landmarks /= divisor
        k_landmarks /= divisor

        # similarities

        einops_eq = '... i d, ... j d -> ... i j'
        attn1 = einsum(einops_eq, q, k_landmarks)
        attn2 = einsum(einops_eq, q_landmarks, k_landmarks)
        attn3 = einsum(einops_eq, q_landmarks, k)

        # masking

        if exists(mask):
            mask_value = -torch.finfo(q.dtype).max
            sim1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

        # eq (15) in the paper and aggregate values

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim = -1), (attn1, attn2, attn3))
        attn2 = moore_penrose_iter_pinv(attn2, iters)
        out = (attn1 @ attn2) @ (attn3 @ v)

        # add depth-wise conv residual of values
        if self.residual:
            out += self.res_conv(v)

        # merge and combine heads

        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)
        out = out[:, -n:]
        if return_attn:
            attn1 = attn1[:,:,0].unsqueeze(-2) @ attn2
            attn1 = (attn1 @ attn3)
        
            return out, attn1[:,:,0,-n+1:]

        return out

# transformer

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Nystromformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_landmarks = 256,
        pinv_iterations = 6,
        attn_values_residual = True,
        attn_values_residual_conv_kernel = 33,
        attn_dropout = 0.,
        ff_dropout = 0.   
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, NystromAttention(dim = dim, dim_head = dim_head, heads = heads, num_landmarks = num_landmarks, pinv_iterations = pinv_iterations, residual = attn_values_residual, residual_conv_kernel = attn_values_residual_conv_kernel, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim = dim, dropout = ff_dropout))
            ]))

    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask) + x
            x = ff(x) + x
        return x
""""""


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512,head=8,drop_out=0.1,drop_path=0.,ffn=False,ffn_act='gelu',mlp_ratio=4.,trans_dim=64,attn='rmsa',n_region=8,epeg=False,region_size=0,min_region_num=0,min_region_ratio=0,qkv_bias=True,crmsa_k=3,epeg_k=15,**kwargs):
        super().__init__()

        self.norm = norm_layer(dim)
        self.norm2 = norm_layer(dim) if ffn else nn.Identity()
        if attn == 'ntrans':
            self.attn = NystromAttention(
                dim = dim,
                dim_head = trans_dim,  # dim // 8
                heads = head,
                num_landmarks = 256,    # number of landmarks dim // 2
                pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
                residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
                dropout=drop_out
            )
        elif attn == 'rmsa':
            self.attn = RegionAttntion(
                dim=dim,
                num_heads=head,
                drop=drop_out,
                region_num=n_region,
                head_dim=dim // head,
                epeg=epeg,
                region_size=region_size,
                min_region_num=min_region_num,
                min_region_ratio=min_region_ratio,
                qkv_bias=qkv_bias,
                epeg_k=epeg_k,
                **kwargs
            )
        elif attn == 'crmsa':
            self.attn = CrossRegionAttntion(
                dim=dim,
                num_heads=head,
                drop=drop_out,
                region_num=n_region,
                head_dim=dim // head,
                epeg=epeg,
                region_size=region_size,
                min_region_num=min_region_num,
                min_region_ratio=min_region_ratio,
                qkv_bias=qkv_bias,
                crmsa_k=crmsa_k,
                **kwargs
            )
        else:
            raise NotImplementedError
        # elif attn == 'rrt1d':
        #     self.attn = RegionAttntion1D(
        #         dim=dim,
        #         num_heads=head,
        #         drop=drop_out,
        #         region_num=n_region,
        #         head_dim=trans_dim,
        #         conv=epeg,
        #         **kwargs
        #     )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = ffn
        act_layer = nn.GELU if ffn_act == 'gelu' else nn.ReLU
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop_out) if ffn else nn.Identity()

    def forward(self,x,need_attn=False):

        x,attn = self.forward_trans(x,need_attn=need_attn)
        
        if need_attn:
            return x,attn
        else:
            return x

    def forward_trans(self, x, need_attn=False):
        attn = None
        
        if need_attn:
            z,attn = self.attn(self.norm(x),return_attn=need_attn)
        else:
            z = self.attn(self.norm(x))

        x = x+self.drop_path(z)

        # FFN
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x,attn

class RRTEncoder(nn.Module):
    def __init__(self,mlp_dim=512,pos_pos=0,pos='none',peg_k=7,attn='rmsa',region_num=8,drop_out=0.1,n_layers=2,n_heads=8,drop_path=0.,ffn=False,ffn_act='gelu',mlp_ratio=4.,trans_dim=64,epeg=True,epeg_k=15,region_size=0,min_region_num=0,min_region_ratio=0,qkv_bias=True,peg_bias=True,peg_1d=False,cr_msa=True,crmsa_k=3,all_shortcut=False,crmsa_mlp=False,crmsa_heads=8,need_init=False,**kwargs):
        super(RRTEncoder, self).__init__()
        
        self.final_dim = mlp_dim

        self.norm = nn.LayerNorm(self.final_dim)
        self.all_shortcut = all_shortcut

        self.layers = []
        for i in range(n_layers-1):
            self.layers += [TransLayer(dim=mlp_dim,head=n_heads,drop_out=drop_out,drop_path=drop_path,ffn=ffn,ffn_act=ffn_act,mlp_ratio=mlp_ratio,trans_dim=trans_dim,attn=attn,n_region=region_num,epeg=epeg,region_size=region_size,min_region_num=min_region_num,min_region_ratio=min_region_ratio,qkv_bias=qkv_bias,epeg_k=epeg_k,**kwargs)]
        self.layers = nn.Sequential(*self.layers)
    
        # CR-MSA
        self.cr_msa = TransLayer(dim=mlp_dim,head=crmsa_heads,drop_out=drop_out,drop_path=drop_path,ffn=ffn,ffn_act=ffn_act,mlp_ratio=mlp_ratio,trans_dim=trans_dim,attn='crmsa',qkv_bias=qkv_bias,crmsa_k=crmsa_k,crmsa_mlp=crmsa_mlp,**kwargs) if cr_msa else nn.Identity()

        # only for ablation
        if pos == 'ppeg':
            self.pos_embedding = PPEG(dim=mlp_dim,k=peg_k,bias=peg_bias,conv_1d=peg_1d)
        elif pos == 'sincos':
            self.pos_embedding = SINCOS(embed_dim=mlp_dim)
        elif pos == 'peg':
            self.pos_embedding = PEG(mlp_dim,k=peg_k,bias=peg_bias,conv_1d=peg_1d)
        else:
            self.pos_embedding = nn.Identity()

        self.pos_pos = pos_pos

        if need_init:
            self.apply(initialize_weights)

    def forward(self, x):
        shape_len = 3
        # for N,C
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            shape_len = 2
        # for B,C,H,W
        if len(x.shape) == 4:
            x = x.reshape(x.size(0),x.size(1),-1)
            x = x.transpose(1,2)
            shape_len = 4

        batch, num_patches, C = x.shape 
        x_shortcut = x

        # PEG/PPEG
        if self.pos_pos == -1:
            x = self.pos_embedding(x)
        
        # R-MSA within region
        for i,layer in enumerate(self.layers.children()):
            if i == 1 and self.pos_pos == 0:
                x = self.pos_embedding(x)
            x = layer(x)

        x = self.cr_msa(x)

        if self.all_shortcut:
            x = x+x_shortcut

        x = self.norm(x)

        if shape_len == 2:
            x = x.squeeze(0)
        elif shape_len == 4:
            x = x.transpose(1,2)
            x = x.reshape(batch,C,int(num_patches**0.5),int(num_patches**0.5))
        return x

class RRTMIL(nn.Module):
    def __init__(self, input_dim=1024,mlp_dim=512,act='relu',n_classes=2,dropout=0.25,pos_pos=0,pos='none',peg_k=7,attn='rmsa',region_num=8,n_layers=2,n_heads=8,drop_path=0.,trans_dropout=0.1,ffn=False,ffn_act='gelu',feat_type='uni',mlp_ratio=4., trans_dim=64,epeg=True,min_region_num=0,qkv_bias=True,**kwargs):
        super().__init__()
        if feat_type == 'uni':
            size = [1024, 512, 256]
        elif feat_type == 'gigapath':
            size = [1536, 768, 384]

        input_dim = size[0]
        mlp_dim = size[1]

        self.patch_to_emb = [nn.Linear(input_dim, mlp_dim)]

        if act.lower() == 'relu':
            self.patch_to_emb += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self.patch_to_emb += [nn.GELU()]

        self.dp = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.patch_to_emb = nn.Sequential(*self.patch_to_emb)

        self.online_encoder = RRTEncoder(mlp_dim=mlp_dim,pos_pos=pos_pos,pos=pos,peg_k=peg_k,attn=attn,region_num=region_num,n_layers=n_layers,n_heads=n_heads,drop_path=drop_path,drop_out=trans_dropout,ffn=ffn,ffn_act=ffn_act,mlp_ratio=mlp_ratio,trans_dim=trans_dim,epeg=epeg,min_region_num=min_region_num,qkv_bias=qkv_bias,**kwargs)

        self.pool_fn = Attn_Net_Gated(L = self.online_encoder.final_dim, D = size[2], dropout = True, n_classes = 1)
        
        self.predictor = nn.Linear(self.online_encoder.final_dim + 30, n_classes)

        self.apply(initialize_weights)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.patch_to_emb= self.patch_to_emb.to(device)
        self.online_encoder = self.online_encoder.to(device)
        self.pool_fn = self.pool_fn.to(device)
        self.predictor = self.predictor.to(device)

    def forward(self, x, coords= None, clinic= None, attention_only=False):
        x = self.patch_to_emb(x) # n*512
        x = self.dp(x)
        
        # feature re-embedding
        x = self.online_encoder(x)
        
        # feature aggregation
        A, x = self.pool_fn(x)

        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A

        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, x) 
        # clinical feature 
        M = torch.cat([M, clinic], dim=1)
        
        # prediction
        logits = self.predictor(M)
        
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        
        return logits, Y_prob, Y_hat, A_raw, {}
    

if __name__ == "__main__":
    x = torch.rand(100,1024)

    # rrt+abmil
    rrt_mil = RRTMIL(n_classes=2,epeg_k=15,crmsa_k=3)
    x_o = rrt_mil(x)  # 1,N,D -> 1,C
