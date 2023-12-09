import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.ops import edge_softmax
import dgl.function as fn
from dgl.utils import expand_as_pair

from graphmae.utils import create_activation


class GAT(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 nhead,
                 nhead_out,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 norm,
                 concat_out=False,
                 encoding=False,
                 p_out_dim=False
                 ):
        super(GAT, self).__init__()
        self.out_dim = out_dim
        self.num_heads = nhead
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.concat_out = concat_out

        last_activation = create_activation(activation) if encoding else None
        last_residual = (encoding and residual)
        last_norm = norm if encoding else None

        
        if num_layers == 1:
            self.gat_layers.append(GATConv(
                in_dim, out_dim, nhead_out,
                feat_drop, attn_drop, negative_slope, last_residual, norm=last_norm, concat_out=concat_out))
        else:
            # input projection (no residual)
            self.gat_layers.append(GATConv(
                in_dim, num_hidden, nhead,
                feat_drop, attn_drop, negative_slope, residual, create_activation(activation), norm=norm, concat_out=concat_out))
            # hidden layers
            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gat_layers.append(GATConv(
                    num_hidden * nhead, num_hidden, nhead,
                    feat_drop, attn_drop, negative_slope, residual, create_activation(activation), norm=norm, concat_out=concat_out))
            # output projection
            self.gat_layers.append(GATConv(
                num_hidden * nhead, out_dim, nhead_out,
                feat_drop, attn_drop, negative_slope, last_residual, activation=last_activation, norm=last_norm, concat_out=concat_out))

        # if norm is not None:
        #     self.norms = nn.ModuleList([
        #         norm(num_hidden * nhead)
        #         for _ in range(num_layers - 1)
        #     ])
        #     if self.concat_out:
        #         self.norms.append(norm(num_hidden * nhead))
        # else:
        #     self.norms = None
    
        self.head = nn.Identity()
        self.p_branch = nn.Identity()
        self.final_out = nn.Identity()
    # def forward(self, g, inputs):
    #     h = inputs
    #     for l in range(self.num_layers):
    #         h = self.gat_layers[l](g, h)
    #         if l != self.num_layers - 1:
    #             h = h.flatten(1)
    #             if self.norms is not None:
    #                 h = self.norms[l](h)
    #     # output projection
    #     if self.concat_out:
    #         out = h.flatten(1)
    #         if self.norms is not None:
    #             out = self.norms[-1](out)
    #     else:
    #         out = h.mean(1)
    #     return self.head(out)
        self.patch_h = 12
        self.patch_w = 12
    
    def forward(self, g, inputs, return_hidden=False, pixel=None, patch=None):
        h = inputs
        hidden_list = []
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h)
            hidden_list.append(h)
            # h = h.flatten(1)
        # output projection
        if pixel: # 当用到像素特征时，先将超像素还原为像素，然后将超像素还原的像素信息与pbranch生成的像素信息连接起来，最后用全连接层输出类型
            h = self.head(h) # 将超像素的维度调整
            s_pixel = h[pixel.label_superpixel] # 根据超像素还原像素属性
            if patch != None: # 当使用patch卷积
                s_pixel = s_pixel[pixel.use_mask]
                p_branch_batch = 3000
                batch_num = patch.shape[0] // p_branch_batch + 1 if patch.shape[0]//p_branch_batch < patch.shape[0]/p_branch_batch else patch.shape[0]//p_branch_batch
                for i in range(batch_num): # 解决爆显存
                    if i == 0:
                        p_pixel = self.p_branch(patch[0:min(patch.shape[0], (i+1)*p_branch_batch)])
                    else:
                        out_p = self.p_branch(patch[i*p_branch_batch:min(patch.shape[0], (i+1)*p_branch_batch)])
                        p_pixel = torch.cat((p_pixel, out_p), dim=0)
                if s_pixel.shape[0] < p_pixel.shape[0]:
                    s_pixel = torch.cat([s_pixel, s_pixel, s_pixel], dim=0)
                out = self.final_out(torch.cat((s_pixel, p_pixel), dim=1))
                return F.softmax(out, dim=1)
            # p_pixel = self.p_branch(pixel.feature) # 当使用全图卷积时
            # h = s_pixel.shape[0]
            # w = s_pixel.shape[1]
            # p = torch.cat((s_pixel, p_pixel), dim=2).view(h*w, -1)
            # p = self.final_out(p).view(h, w, -1)
            # return F.softmax(p, dim=2)
            
        else:
            if return_hidden: # 这里做修改，原本的eval模块在这里就分类了，但是如果要加入像素特征可能还得改变一下输出并且再多加几层线性层
                return self.head(h), hidden_list
            else:
                return self.head(h)

    def reset_classifier(self, num_classes, feat_dim, s_out_dim=128, p_out_dim=64):
        self.s_out_dim = s_out_dim
        self.head = nn.Linear(self.num_heads * self.out_dim, s_out_dim)

        # self.p_out_dim = p_out_dim # 当使用全像素卷积
        # self.p_branch = pixel_Conv(9, self.p_out_dim) # 此处超参数可定义
        
        self.p_out_dim = p_out_dim# 当使用patch
        self.p_branch = patch_Conv(feat_dim, p_out_dim, self.patch_h, self.patch_w)

        self.final_out = nn.Linear(self.p_out_dim+self.s_out_dim, num_classes)
    
    def clear_classifier(self):
        self.head = nn.Identity()
        self.p_branch = nn.Identity()
        self.final_out = nn.Identity()
        
class GATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True,
                 norm=None,
                 concat_out=True):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._concat_out = concat_out

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        # if norm is not None:
        #     self.norm = norm(num_heads * out_feats)
        # else:
        #     self.norm = None
    
        self.norm = norm
        if norm is not None:
            self.norm = norm(num_heads * out_feats)

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise RuntimeError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
                    dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            # e[e == 0] = -1e3
            # e = graph.edata.pop('e')
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']

            # bias
            if self.bias is not None:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)

            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self._out_feats)
                rst = rst + resval

            if self._concat_out:
                rst = rst.flatten(1)
            else:
                rst = torch.mean(rst, dim=1)

            if self.norm is not None:
                rst = self.norm(rst)

            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst

class pixel_Conv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3):
        super().__init__()
        self.Conv1 = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
        self.Conv2 = nn.Conv2d(out_c, out_c, kernel_size=kernel_size, padding=kernel_size//2, groups=out_c) # 不知道为何要用分组卷积，测试吧
        self.BN1 = nn.BatchNorm2d(in_c)
    def forward(self, x):
        x = x.unsqueeze(0).transpose(1, 3)
        x = self.Conv1(self.BN1(x))
        x = F.leaky_relu(x)
        x = self.Conv2(x)
        return F.leaky_relu(x).squeeze(0).transpose(0, 2)
    
        
class patch_Conv(nn.Module):
    def __init__(self, in_c, out_c, h, w):
        super().__init__()
        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_c, 128, 3, padding=1, bias=False),
            nn.ReLU())
        self.Conv2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.AvgPool2d(2, 2),
            nn.ReLU())
        self.Conv3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU())
        self.Conv4 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.AvgPool2d(2, 2),
            nn.ReLU())
        self.Linear = nn.Linear(1024*h//4*w//4, 1024)
        self.final_out = nn.Linear(1024, out_c)
    
    def forward(self, x):
        batch_num = x.shape[0]
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x = self.Linear(x.view(batch_num, -1))
        return F.softmax(self.final_out(x), dim=1)
    