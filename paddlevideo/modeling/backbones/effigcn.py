import paddle
import paddle.nn as nn
from .attentions import Attention_Layer
from .effigcn_init import EffigcnInit
from paddlevideo.modeling.weight_init import weight_init_
from ..registry import BACKBONES

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

@BACKBONES.register()
class EfficientGCN(nn.Layer):

    def __init__(self):
        super(EfficientGCN, self).__init__()

        block_args=[[96, 1, 2], [48, 1, 2], [128, 2, 3], [272, 2, 3]]
        data_shape=[3, 4, 500, 25, 1]
        fusion_stage=2
        stem_channel=64

        kwargs=EffigcnInit.getargs()
        kwargs.pop('block_args')
        kwargs.pop('data_shape')
        kwargs.pop('fusion_stage')
        kwargs.pop('stem_channel')

        num_input, num_channel, _, _, _ = data_shape

        # input branches
        self.input_branches = nn.LayerList([EfficientGCN_Blocks(
            init_channel = stem_channel,
            block_args = block_args[:fusion_stage],
            input_channel = num_channel,
            **kwargs
        ) for _ in range(num_input)])

        # main stream
        last_channel = stem_channel if fusion_stage == 0 else block_args[fusion_stage-1][0]
        self.main_stream = EfficientGCN_Blocks(
            init_channel = num_input * last_channel,
            block_args = block_args[fusion_stage:],
            **kwargs
        )

        # output
        last_channel = num_input * block_args[-1][0] if fusion_stage == len(block_args) else block_args[-1][0]
        # self.classifier = EfficientGCN_Classifier(last_channel, **kwargs)

        # init parameters
        self.init_param(self.sublayers())

    def init_param(self,modules):
        for m in modules:
            if isinstance(m, nn.Conv1D) or isinstance(m, nn.Conv2D):
                # nn.initializer.KaimingNormal(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                m.weight = paddle.create_parameter(
                    shape=m.weight.shape,
                    dtype='float32',
                    default_initializer=paddle.nn.initializer.KaimingNormal()
                )
                if m.bias is not None:
                    m.bias = paddle.create_parameter(
                        shape=m.bias.shape,
                        dtype='float32',
                        default_initializer=paddle.nn.initializer.Constant(0)
                    )
            elif isinstance(m, nn.BatchNorm1D) or isinstance(m, nn.BatchNorm2D) or isinstance(m, nn.BatchNorm3D):

                m.weight = paddle.create_parameter(
                    shape=m.weight.shape,
                    dtype='float32',
                    default_initializer=paddle.nn.initializer.Constant(1.0)
                )

                m.bias = paddle.create_parameter(
                    shape=m.bias.shape,
                    dtype='float32',
                    default_initializer=paddle.nn.initializer.Constant(0)
                )

            elif isinstance(m, nn.Conv3D) or isinstance(m, nn.Linear):
                # nn.initializer.normal(m.weight, std=0.001)
                weight_init_(m, 'Normal', mean=0.0, std=0.001)
                if m.bias is not None:
                    m.bias = paddle.create_parameter(
                        shape=m.bias.shape,
                        dtype='float32',
                        default_initializer=paddle.nn.initializer.Constant(0)
                    )


    def forward(self, x):

        N, I, C, T, V, M = x.shape

      #  x = x.permute(1, 0, 5, 2, 3, 4).contiguous().view(I, N*M, C, T, V)

       # x=x.transpose(1, 0, 5, 2, 3, 4).reshape(I, N*M, C, T, V)

        x = paddle.transpose(x,perm=[1, 0, 5, 2, 3, 4])
        x = paddle.reshape(x,[I, N * M, C, T, V])



        # input branches
        # y=self.input_branches[0
        for i,branch in enumerate(self.input_branches):
            y=branch(x[i])
            # x=paddle.concat()
        x = paddle.concat([branch(x[i]) for i, branch in enumerate(self.input_branches)], axis=1)
        # paddle

        # main stream
        x = self.main_stream(x)

        # output
        _, C, T, V = x.shape
        #feature = x.view(N, M, C, T, V).permute(0, 2, 3, 4, 1)
        x=paddle.reshape(x,[N, M, C, T, V])
        feature=paddle.transpose(x,perm=[0, 2, 3, 4, 1])

        #out = self.classifier(feature).view(N, -1)
        # out = paddle.reshape(self.classifier(feature),[N,-1])
        # return out, feature
        return feature


class EfficientGCN_Blocks(nn.Sequential):
    def __init__(self, init_channel, block_args, layer_type, kernel_size, input_channel=0, **kwargs):
        super(EfficientGCN_Blocks, self).__init__()

        temporal_window_size, max_graph_distance = kernel_size

        if input_channel > 0:  # if the blocks in the input branches
            self.add_sublayer('init_bn', nn.BatchNorm2D(input_channel))
            self.add_sublayer('stem_scn', Spatial_Graph_Layer(input_channel, init_channel, max_graph_distance, **kwargs))
            self.add_sublayer('stem_tcn', Temporal_Basic_Layer(init_channel, temporal_window_size, **kwargs))

        last_channel = init_channel
        temporal_layer = import_class(f'paddlevideo.modeling.backbones.effigcn.Temporal_{layer_type}_Layer')

        for i, [channel, stride, depth] in enumerate(block_args):
            self.add_sublayer(f'block-{i}_scn', Spatial_Graph_Layer(last_channel, channel, max_graph_distance, **kwargs))
            for j in range(depth):
                s = stride if j == 0 else 1
                self.add_sublayer(f'block-{i}_tcn-{j}', temporal_layer(channel, temporal_window_size, stride=s, **kwargs))
            self.add_sublayer(f'block-{i}_att', Attention_Layer(channel, **kwargs))
            last_channel = channel


class Basic_Layer(nn.Layer):
    def __init__(self, in_channel, out_channel, residual, bias, act, **kwargs):
        super(Basic_Layer, self).__init__()

        self.conv = nn.Conv2D(in_channel, out_channel, 1, bias_attr=bias)
        self.bn = nn.BatchNorm2D(out_channel)

        self.residual = nn.Identity() if residual else Zero_Layer()
        self.act = act

    def forward(self, x):
        res = self.residual(x)
        x = self.act(self.bn(self.conv(x)) + res)
        return x


class Spatial_Graph_Layer(Basic_Layer):
    def __init__(self, in_channel, out_channel, max_graph_distance, bias, residual=True, **kwargs):
        super(Spatial_Graph_Layer, self).__init__(in_channel, out_channel, residual, bias, **kwargs)

        self.conv = SpatialGraphConv(in_channel, out_channel, max_graph_distance, bias, **kwargs)
        if residual and in_channel != out_channel:
            self.residual = nn.Sequential(
                nn.Conv2D(in_channel, out_channel, 1, bias_attr=bias),
                nn.BatchNorm2D(out_channel),
            )


class Temporal_Basic_Layer(Basic_Layer):
    def __init__(self, channel, temporal_window_size, bias, stride=1, residual=True, **kwargs):
        super(Temporal_Basic_Layer, self).__init__(channel, channel, residual, bias, **kwargs)

        padding = (temporal_window_size - 1) // 2
        self.conv = nn.Conv2D(channel, channel, (temporal_window_size,1), (stride,1), (padding,0), bias_attr=bias)
        if residual and stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2D(channel, channel, 1, (stride,1), bias_attr=bias),
                nn.BatchNorm2D(channel),
            )


class Temporal_Bottleneck_Layer(nn.Layer):
    def __init__(self, channel, temporal_window_size, bias, act, reduct_ratio, stride=1, residual=True, **kwargs):
        super(Temporal_Bottleneck_Layer, self).__init__()

        inner_channel = channel // reduct_ratio
        padding = (temporal_window_size - 1) // 2
        self.act = act

        self.reduct_conv = nn.Sequential(
            nn.Conv2D(channel, inner_channel, 1, bias_attr=bias),
            nn.BatchNorm2D(inner_channel),
        )
        self.conv = nn.Sequential(
            nn.Conv2D(inner_channel, inner_channel, (temporal_window_size,1), (stride,1), (padding,0), bias_attr=bias),
            nn.BatchNorm2D(inner_channel),
        )
        self.expand_conv = nn.Sequential(
            nn.Conv2D(inner_channel, channel, 1, bias_attr=bias),
            nn.BatchNorm2D(channel),
        )

        if not residual:
            self.residual = Zero_Layer()
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2D(channel, channel, 1, (stride,1), bias_attr=bias),
                nn.BatchNorm2D(channel),
            )

    def forward(self, x):
        res = self.residual(x)
        x = self.act(self.reduct_conv(x))
        x = self.act(self.conv(x))
        x = self.act(self.expand_conv(x) + res)
        return x


class Temporal_Sep_Layer(nn.Layer):
    def __init__(self, channel, temporal_window_size, bias, act, expand_ratio, stride=1, residual=True, **kwargs):
        super(Temporal_Sep_Layer, self).__init__()

        padding = (temporal_window_size - 1) // 2
        self.act = act

        if expand_ratio > 0:
            inner_channel = channel * expand_ratio
            self.expand_conv = nn.Sequential(
                nn.Conv2D(channel, inner_channel, 1, bias_attr=bias),
                nn.BatchNorm2D(inner_channel),
            )
        else:
            inner_channel = channel
            self.expand_conv = None

        self.depth_conv = nn.Sequential(
            nn.Conv2D(inner_channel, inner_channel, (temporal_window_size,1), (stride,1), (padding,0), groups=inner_channel, bias_attr=bias),
            nn.BatchNorm2D(inner_channel),
        )
        self.point_conv = nn.Sequential(
            nn.Conv2D(inner_channel, channel, 1, bias_attr=bias),
            nn.BatchNorm2D(channel),
        )
        if not residual:
            self.residual = Zero_Layer()
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2D(channel, channel, 1, (stride,1), bias_attr=bias),
                nn.BatchNorm2D(channel),
            )

    def forward(self, x):
        res = self.residual(x)
        if self.expand_conv is not None:
            x = self.act(self.expand_conv(x))
        x = self.act(self.depth_conv(x))
        x = self.point_conv(x)
        return x + res


class Temporal_SG_Layer(nn.Layer):
    def __init__(self, channel, temporal_window_size, bias, act, reduct_ratio, stride=1, residual=True, **kwargs):
        super(Temporal_SG_Layer, self).__init__()

        padding = (temporal_window_size - 1) // 2
        inner_channel = channel // reduct_ratio
        self.act = act

        self.depth_conv1 = nn.Sequential(
            nn.Conv2D(channel, channel, (temporal_window_size,1), 1, (padding,0), groups=channel, bias_attr=bias),
            nn.BatchNorm2D(channel),
        )
        self.point_conv1 = nn.Sequential(
            nn.Conv2D(channel, inner_channel, 1, bias_attr=bias),
            nn.BatchNorm2D(inner_channel),
        )
        self.point_conv2 = nn.Sequential(
            nn.Conv2D(inner_channel, channel, 1, bias_attr=bias),
            nn.BatchNorm2D(channel),
        )
        self.depth_conv2 = nn.Sequential(
            nn.Conv2D(channel, channel, (temporal_window_size,1), (stride,1), (padding,0), groups=channel, bias_attr=bias),
            nn.BatchNorm2D(channel),
        )

        if not residual:
            self.residual = Zero_Layer()
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2D(channel, channel, 1, (stride,1), bias_attr=bias),
                nn.BatchNorm2D(channel),
            )

    def forward(self, x):
        res = self.residual(x)
        x = self.act(self.depth_conv1(x))
        x = self.point_conv1(x)
        x = self.act(self.point_conv2(x))
        x = self.depth_conv2(x)
        return x + res


class Zero_Layer(nn.Layer):
    def __init__(self):
        super(Zero_Layer, self).__init__()

    def forward(self, x):
        return 0


# Thanks to YAN Sijie for the released code on Github (https://github.com/yysijie/st-gcn)
class SpatialGraphConv(nn.Layer):
    def __init__(self, in_channel, out_channel, max_graph_distance, bias, edge, A, **kwargs):
        super(SpatialGraphConv, self).__init__()

        self.s_kernel_size = max_graph_distance + 1
        self.gcn = nn.Conv2D(in_channel, out_channel*self.s_kernel_size, 1, bias_attr=bias)
        x1=(A[:self.s_kernel_size]).numpy()
        #self.A = nn.Parameter(A[:self.s_kernel_size], requires_grad=False)
        b=paddle.nn.initializer.Assign(x1)
        self.A=paddle.create_parameter(x1.shape,dtype='float32' ,
                                       default_initializer=b)


        if edge:
            x2=paddle.ones_like((self.A))
            self.edge = paddle.create_parameter(x2.shape,dtype=str(x2.numpy().dtype),
                                             default_initializer=paddle.nn.initializer.Assign(x2))
        else:
            self.edge = 1

    def forward(self, x):
        x = self.gcn(x)
        n, kc, t, v = x.shape
     #  x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)

        x=paddle.reshape(x,[n, self.s_kernel_size, kc//self.s_kernel_size, t, v])
        #x = paddle.einsum('nkctv,kvw->nctw', (x, self.A * self.edge)).contiguous()
        x = paddle.einsum('nkctv,kvw->nctw', x, self.A * self.edge)
        return x
