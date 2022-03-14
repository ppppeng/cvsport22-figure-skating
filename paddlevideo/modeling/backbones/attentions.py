import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class Attention_Layer(nn.Layer):
    def __init__(self, out_channel, att_type, act, **kwargs):
        super(Attention_Layer, self).__init__()

        __attention = {
            'stja': ST_Joint_Att,
            'pa': Part_Att,
            'ca': Channel_Att,
            'fa': Frame_Att,
            'ja': Joint_Att,
        }

        self.att = __attention[att_type](channel=out_channel, **kwargs)
        self.bn = nn.BatchNorm2D(out_channel)
        self.act = act

    def forward(self, x):
        res = x
        x = x * self.att(x)
        return self.act(self.bn(x) + res)


class ST_Joint_Att(nn.Layer):
    def __init__(self, channel, reduct_ratio, bias, **kwargs):
        super(ST_Joint_Att, self).__init__()

        inner_channel = channel // reduct_ratio

        self.fcn = nn.Sequential(
            nn.Conv2D(channel, inner_channel, kernel_size=1, bias_attr=bias),
            nn.BatchNorm2D(inner_channel),
            nn.Hardswish(),
        )
        self.conv_t = nn.Conv2D(inner_channel, channel, kernel_size=1)
        self.conv_v = nn.Conv2D(inner_channel, channel, kernel_size=1)

    def forward(self, x):
        N, C, T, V = x.shape
        x_t = paddle.mean(x,3, keepdim=True)
        #x_v = paddle.mean(x,2, keepdim=True).transpose(2, 3)
        x_v = paddle.mean(x, 2, keepdim=True)
        x_v = paddle.transpose(x_v,(0,1,3,2))
        x_att = self.fcn(paddle.concat([x_t, x_v], axis=2))
        x_t, x_v = paddle.split(x_att, [T, V], axis=2)
        #x_t_att = self.conv_t(x_t).sigmoid()
        x_t_att=F.sigmoid(self.conv_t(x_t))
        #x_v_att = self.conv_v(x_v.transpose(2, 3)).sigmoid()
        x_v_att=F.sigmoid(self.conv_v(paddle.transpose(x_v,(0,1,3,2))))
        x_att = x_t_att * x_v_att
        return x_att


class Part_Att(nn.Layer):
    def __init__(self, channel, parts, reduct_ratio, bias, **kwargs):
        super(Part_Att, self).__init__()

        self.parts = parts

        #self.joints = nn.Parameter(self.get_corr_joints(), requires_grad=False)
        x1=self.get_corr_joints()
        self.joints = paddle.create_parameter(x1.shape,dtype=str(x1.numpy().dtype),
                      default_initializer=paddle.nn.initializer.assign(x1))
        inner_channel = channel // reduct_ratio

        self.softmax = nn.Softmax(dim=3)
        self.fcn = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(channel, inner_channel, kernel_size=1, bias_attr=bias),
            nn.BatchNorm2D(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2D(inner_channel, channel*len(self.parts), kernel_size=1, bias_attr=bias),
        )

    def forward(self, x):
        N, C, T, V = x.size()
        x_att = self.softmax(self.fcn(x).view(N, C, 1, len(self.parts)))
        x_att = x_att.index_select(3, self.joints).expand_as(x)
        return x_att

    def get_corr_joints(self):
        num_joints = sum([len(part) for part in self.parts])
        joints = [j for i in range(num_joints) for j in range(len(self.parts)) if i in self.parts[j]]
        #return torch.LongTensor(joints)
        return paddle.to_tensor(joints,dtype='int64')

class Channel_Att(nn.Layer):
    def __init__(self, channel, **kwargs):
        super(Channel_Att, self).__init__()

        self.fcn = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(channel, channel//4, kernel_size=1),
            nn.BatchNorm2D(channel//4),
            nn.ReLU(inplace=True),
            nn.Conv2D(channel//4, channel, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.fcn(x)


class Frame_Att(nn.Layer):
    def __init__(self, **kwargs):
        super(Frame_Att, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.max_pool = nn.AdaptiveMaxPool2D(1)
        self.conv = nn.Conv2D(2, 1, kernel_size=(9,1), padding=(4,0))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = paddle.concat([self.avg_pool(x), self.max_pool(x)], dim=2).transpose(1, 2)
        return self.conv(x)


class Joint_Att(nn.Layer):
    def __init__(self, parts, **kwargs):
        super(Joint_Att, self).__init__()

        num_joint = sum([len(part) for part in parts])

        self.fcn = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(num_joint, num_joint//2, kernel_size=1),
            nn.BatchNorm2D(num_joint//2),
            nn.ReLU(inplace=True),
            nn.Conv2D(num_joint//2, num_joint, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fcn(x.transpose(1, 3)).transpose(1, 3)
