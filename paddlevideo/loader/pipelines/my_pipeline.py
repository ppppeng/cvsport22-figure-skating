import numpy as np
import random
from ..registry import PIPELINES


@PIPELINES.register()
class Mixup(object):
    """
    We compose all batch pipeline in this classes.

    Mixup operator.
    Args:
        alpha(float): alpha value.
    """

    def __init__(self, alpha=0.2):
        assert alpha > 0., \
            'parameter alpha[%f] should > 0.0' % (alpha)
        self.alpha = alpha
        self.xaxistrun = XaxisTurn()
        self.scaling = Scaling()
        # self.cutmixup = CutMixup()

    def __call__(self, batch):
        # NCTVM
        batch = self.xaxistrun(batch)
        batch = self.mixup(batch)
        return batch

    def mixup(self, batch):
        imgs, labels = list(zip(*batch))
        imgs = np.array(imgs)  # NCTVM
        labels = np.array(labels)
        bs = len(batch)
        idx = np.random.permutation(bs)  # 对[0,bs)区间随机排列
        lam = np.random.beta(self.alpha, self.alpha)
        lams = np.array([lam] * bs, dtype=np.float32)
        imgs = lam * imgs + (1 - lam) * imgs[idx]
        return list(zip(imgs, labels, labels[idx], lams))


@PIPELINES.register()
class CutOut(object):
    """
    Random cut out the frame skeleton feature. Special for FSD dataset.
        style   probability
        0       base     头身体
        1       base     左手
        2       base     右手
        3       base     左脚
        4       base     右脚
        sum of all prob is 1

    Args:
        alpha: float, the probability to CutOut.
    """

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.style_to_point = [[17, 18, 15, 16, 0, 1],
                               [4, 3, 2],
                               [7, 6, 5],
                               [23, 22, 24, 11, 10, 9],
                               [20, 19, 21, 14, 13, 12]]

    def __call__(self, results):
        if (np.random.rand() > self.alpha):
            return results

        data = results['data']  # CTVM
        style = np.random.randint(0, 5)
        points = self.style_to_point[style]
        length = np.random.randint(0, len(points))
        data[:, :, points[0:length], :] = 0

        results['data'] = data

        return results


@PIPELINES.register()
class Rotate(object):
    """
    Random rotating the frame skeleton feature.
    Args:
        alpha: float, represent the bound of variation of Rotate.
    """

    def __init__(self, alpha=10):
        self.alpha = alpha

    def __call__(self, results):
        # angle = (np.random.randn() / 2) * self.alpha
        angle = (np.random.rand() - 0.5) / 0.5 * self.alpha
        radian = self.angle_to_radian(angle)
        data = results['data']  # CTVM

        x = data[0:1]
        y = data[1:2]
        data[0:1] = x * np.cos(radian) + y * np.sin(radian)
        data[1:2] = -x * np.sin(radian) + y * np.cos(radian)

        results['data'] = data
        return results

    def angle_to_radian(self, x):
        return x * np.pi / 180


@PIPELINES.register()
class XaxisTurn(object):
    """
    Half to chance turning the frame skeleton feature in dimension X.
    """

    def __init__(self):
        pass

    def __call__(self, batch):
        imgs, _ = list(zip(*batch))
        imgs = np.array(imgs)  # NCTVM
        bs = len(batch)
        select = np.random.rand(bs) > 0.5
        # print(select)
        imgs[select, :1] = -imgs[select, :1]
        if imgs.shape[1] == 4:  # DataPad
            imgs[select, 2:3] = -imgs[select, 2:3]
        return list(zip(imgs, _))


@PIPELINES.register()
class CutMixup(object):
    """
    CutMixup the data
    """

    def __init__(self):
        # NCTVM
        '''
                style   lam
                0       0.9     头
                1       0.9     身体
                2       0.8     左手
                3       0.8     右手
                4       0.8     左脚
                5       0.8     右脚
        '''
        self.lam = [0.9, 0.9, 0.8, 0.8, 0.8, 0.8]
        self.style_to_point = [[0, 15, 16, 17, 18],
                               [1, 8, 9, 12],
                               [2, 3, 4],
                               [5, 6, 7],
                               [10, 11, 22, 23, 24],
                               [13, 14, 19, 20, 21]]

    def __call__(self, batch):
        style = np.random.randint(0, 6)
        points = self.style_to_point[style]
        imgs, labels = list(zip(*batch))
        imgs = np.array(imgs)  # NCTVM
        labels = np.array(labels)
        bs = len(batch)
        idx = np.random.permutation(bs)  # 对[0,bs)区间随机排列
        lams = np.array([self.lam[style]] * bs, dtype=np.float32)
        tmp = imgs[idx]
        imgs[:, :, :, points, :] = tmp[:, :, :, points, :]
        return list(zip(imgs, labels, labels[idx], lams))


@PIPELINES.register()
class YaxisScaling(object):
    """
    Random scaling the frame skeleton feature in dimension Y
    Args:
        alpha: float, represent the bound of variation of YaxisScaling.
    """

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def __call__(self, results):
        # C,T,V,M
        data = results['data']
        # scale = ((np.random.randn() / 2) * self.alpha) + 1
        scale = (((np.random.rand() - 0.5) / 0.5) * self.alpha) + 1
        data[1:2] = data[1:2] * scale
        results['data'] = data
        return results


@PIPELINES.register()
class Scaling(object):
    """
    Random Scaling the frame skeleton feature.
    Args:
        alpha: float, represent the bound of variation of Scaling.
    """

    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, results):
        # C,T,V,M
        data = results['data']
        # scale = ((np.random.randn() / 2) * self.alpha) + 1
        scale = (((np.random.rand() - 0.5) / 0.5) * self.alpha) + 1
        data[0:2] = data[0:2] * scale
        results['data'] = data
        return results


@PIPELINES.register()
class AutoPadding(object):
    """
    Sample or Padding frame skeleton feature.
    Args:
        window_size: int, temporal size of skeleton feature.
        random_pad: bool, whether do random padding when frame length < window size. Default: False.
        random_size: bool, whether do random refresh window_size or not. Default: False.
        min_window_size: int, the lower bound when refresh window_size, work when random_size=True.
        max_window_size: int, the upper bound when refresh window_size, work when random_size=True.
    """

    def __init__(self, window_size, random_size=False, min_window_size=350, max_window_size=500, random_pad=False):
        self.window_size = window_size
        self.random_pad = random_pad
        self.random_size = random_size
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size

    def refresh_window_size(self):
        if self.random_size == True:
            self.window_size = np.random.randint(self.min_window_size, self.max_window_size)
            print("window_size = %s " % self.window_size)

    def get_frame_num(self, data):
        C, T, V, M = data.shape
        for i in range(T - 1, -1, -1):
            tmp = np.sum(data[:, i, :, :])
            if not tmp == 0:
                T = i + 1
                break
        return T

    def __call__(self, results):
        data = results['data']
        C, T, V, M = data.shape
        T = self.get_frame_num(data)
        # print(self.window_size, end=' ')
        if T == self.window_size:
            data_pad = data[:, :self.window_size, :, :]
        elif T < self.window_size:
            begin = random.randint(0, self.window_size -
                                   T) if self.random_pad else 0
            data_pad = np.zeros((C, self.window_size, V, M))
            data_pad[:, begin:begin + T, :, :] = data[:, :T, :, :]
        else:
            if self.random_pad:
                index = np.random.choice(T, self.window_size,
                                         replace=False).astype('int64')
            else:
                index = np.linspace(0, T, self.window_size).astype("int64")
            data_pad = data[:, index, :, :]

        results['data'] = data_pad
        return results


@PIPELINES.register()
class SkeletonNorm(object):
    """
    We had normalized the data by function in data_process.py
    Args:
        aixs: dimensions of vertex coordinate. 2 for (x,y), 3 for (x,y,z). Default: 2.
    """

    # todo testcode axis=2
    def __init__(self, axis=2, squeeze=False):
        self.axis = axis
        self.squeeze = squeeze

    def __call__(self, results):
        data = results['data']
        # C,T,V,M

        # Centralization
        data = data[:self.axis, :, :, :]  # get (x,y) from (x,y, acc)
        data = data - data[:, :, 8:9, :]
        C, T, V, M = data.shape
        if self.squeeze:
            data = data.reshape((C, T, V))  # M = 1

        results['data'] = data.astype('float32')
        if 'label' in results:
            label = results['label']
            results['label'] = np.expand_dims(label, 0).astype('int64')
        return results


@PIPELINES.register()
class DataPad(object):
    """
    Padding the data feature by dx,dy that mean the variation of coordinate in dimension X,Y
    """

    def __init__(self):
        pass

    def __call__(self, results):
        # C,T,V,M
        data = results['data']
        data1 = data[:, :-1]
        data2 = data[:, 1:]
        diff = data2 - data1
        data = np.concatenate((data[:, 1:], diff), axis=0)

        results['data'] = data
        return results


@PIPELINES.register()
class EffPipeLine(object):

    def __init__(self):

        self.T = 500
        self.conn = np.array(
            # [0  1  2  3  4   5  6  7  8  9  10  11  12  13  14  15  16  17  18  19  20  21   22   23  24 ]
            [1, 1, 1, 2, 3, 1, 5, 6, 1, 8, 9, 10, 8, 12, 13, 0, 0, 15, 16, 14, 19, 14, 11, 22, 11]
        )

    def __call__(self, results):
        data = results['data']
        joint, velocity, bone = self.multi_input(data[:, :self.T, :, :])
        data_new = []

        data_new.append(joint)
        data_new.append(velocity)
        data_new.append(bone)

        data_new = np.stack(data_new, axis=0)
        results['data'] = data_new
        return results

    def multi_input(self, data):
        data = data[0:2, :, :, :]  # 去掉z
        C, T, V, M = data.shape
        joint = np.zeros((C * 2, T, V, M))
        velocity = np.zeros((C * 2, T, V, M))
        bone = np.zeros((C * 2, T, V, M))
        # joint = np.zeros((5, T, V, M))
        # velocity = np.zeros((5, T, V, M))
        # bone = np.zeros((5, T, V, M))
        joint[:C, :, :, :] = data
        for i in range(V):
            joint[C:, :, i, :] = data[:, :, i, :] - data[:, :, 1, :]
        for i in range(T - 2):
            velocity[:C, i, :, :] = data[:, i + 1, :, :] - data[:, i, :, :]
            velocity[C:, i, :, :] = data[:, i + 2, :, :] - data[:, i, :, :]
        for i in range(len(self.conn)):
            bone[:C, :, i, :] = data[:, :, i, :] - data[:, :, self.conn[i], :]
        bone_length = 0
        for i in range(C):
            bone_length += bone[i, :, :, :] ** 2
        bone_length = np.sqrt(bone_length) + 0.0001
        for i in range(C):
            bone[C + i, :, :, :] = np.arccos(bone[i, :, :, :] / bone_length)
        return joint, velocity, bone
