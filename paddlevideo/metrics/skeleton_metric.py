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
import copy

import numpy as np
import paddle
import csv
import paddle.nn.functional as F

from .registry import METRIC
from .base import BaseMetric
from paddlevideo.utils import get_logger

logger = get_logger("paddlevideo")


@METRIC.register
class SkeletonMetric(BaseMetric):
    """
    Test for Skeleton based model.
    note: only support batch size = 1, single card test.

    Args:
        out_file: str, file to save test results.
    """

    def __init__(self,
                 data_size,
                 batch_size,
                 out_dir=".",
                 out_file='submission.csv',
                 log_interval=1,
                 out_score='score.npy'):
        """prepare for metrics
        """
        super().__init__(data_size, batch_size, log_interval)
        self.top1 = []
        self.top5 = []
        self.values = []
        self.out_dir = out_dir
        self.out_file = self.out_dir + '/' + out_file
        self.out_score = self.out_dir + '/' + out_score
        self.np_data = None

    def vote(self, label, confi):
        '''
            根据投票标签和置信度，选出票数最多的标签中置信度最高的下标
        '''
        count = [0] * 30
        for i in range(0, len(label)):
            count[label[i]] += 1
        out_label = np.argmax(count)
        update_confi = copy.deepcopy(confi)
        update_confi[~(label == out_label)] = 0
        # if not np.argmax(confi) == np.argmax(update_confi):
        #     print("WARNING!!!!!!!!!!!!")
        #     print(label)
        #     print(confi)
        # if not np.argmax(confi)==0:
        #     print("WARNING!!!!!!!!!!!!")
        #     print(label)
        #     print(confi)
        return np.argmax(update_confi)
        # return np.argmax(confi)

    def vote_updata(self, batch_id, outputs):
        prob = F.softmax(outputs)
        prob = prob.numpy()
        predict_label = np.argmax(prob, axis=-1)
        predict_confi = np.max(prob, axis=-1)
        select = self.vote(predict_label, predict_confi)
        # prob = prob[select].mean(axis=0)
        prob = prob[select]
        prob = np.expand_dims(prob, axis=0)

        # print(predict_label)
        # print(predict_confi)
        if self.np_data is None:
            self.np_data = prob
        else:
            self.np_data = np.concatenate((self.np_data, prob), axis=0)
        # clas = paddle.argmax(prob, axis=1).numpy()[0]
        # self.values.append((batch_id, clas))

        # preds ensemble
        if batch_id % self.log_interval == 0:
            logger.info("[TEST] Processing batch {}/{} ...".format(
                batch_id,
                self.data_size // (self.batch_size * self.world_size)))

    def update(self, batch_id, data, outputs):
        """update metrics during each iter
        """
        if len(data) == 2:  # data with label
            labels = data[1]
            top1 = paddle.metric.accuracy(input=outputs, label=labels, k=1)
            top5 = paddle.metric.accuracy(input=outputs, label=labels, k=5)
            if self.world_size > 1:
                top1 = paddle.distributed.all_reduce(
                    top1, op=paddle.distributed.ReduceOp.SUM) / self.world_size
                top5 = paddle.distributed.all_reduce(
                    top5, op=paddle.distributed.ReduceOp.SUM) / self.world_size
            self.top1.append(top1.numpy())
            self.top5.append(top5.numpy())
        else:  # data without label, only support batch_size=1. Used for fsd-10.
            prob = F.softmax(outputs)
            if self.np_data is None:
                self.np_data = prob
            else:
                self.np_data = np.concatenate((self.np_data, prob), axis=0)
            # clas = paddle.argmax(prob, axis=1).numpy()[0]
            # self.values.append((batch_id, clas))

        # preds ensemble
        if batch_id % self.log_interval == 0:
            logger.info("[TEST] Processing batch {}/{} ...".format(
                batch_id,
                self.data_size // (self.batch_size * self.world_size)))

    def accumulate(self):
        """accumulate metrics when finished all iters.
        """
        if self.top1:  # data with label
            logger.info('[TEST] finished, avg_acc1= {}, avg_acc5= {}'.format(
                np.mean(np.array(self.top1)), np.mean(np.array(self.top5))))
        else:

            np.save(self.out_score, self.np_data)
            logger.info("Score saved in {} !".format(self.out_score))
