# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn

from .base import BaseHead
from ..registry import HEADS
from ..weight_init import weight_init_
import sys

sys.path.append("....")



@HEADS.register()
class EfficientGCNHead(BaseHead):
    """
    Head for ST-GCN model.
    Args:
        in_channels: int, input feature channels. Default: 256.
        num_classes: int, number classes. Default: 10.
    """

    # todo test code in_channels=512

    def __init__(self,in_channels=272,num_classes=30,**kwargs):
        super(EfficientGCNHead, self).__init__(num_classes, in_channels)

        curr_channel=272
        num_class=30
        drop_prob=0.25

       # self.add_sublayer('gap', nn.AdaptiveAvgPool3D(1))
        self.gap=nn.AdaptiveAvgPool3D(1)
       # self.add_sublayer('dropout', nn.Dropout(drop_prob))
        self.dropout=nn.Dropout(drop_prob)
        #self.add_sublayer('fc', nn.Conv3D(curr_channel, num_class, kernel_size=1))
        self.fc=nn.Conv3D(curr_channel, num_class, kernel_size=1)


    def forward(self, x):
        N=x.shape[0]

        x=self.gap(x)
        x=self.dropout(x)
        x=self.fc(x)

        out = paddle.reshape(x, [N, -1])

        return out
