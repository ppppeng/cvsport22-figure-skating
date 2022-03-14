import numpy as np
import os
import matplotlib.pyplot as plt
from os.path import join

# os.environ['CUDA_VISIBLE_DEVICES'] = "2"


def get_frame_num(data):
    C, T, V, M = data.shape
    for i in range(T - 1, -1, -1):
        tmp = np.sum(data[:, i, :, :])
        if not tmp == 0:
            T = i + 1
            break
    return T


def vis_one_frame(action_idx, frame_idx, data):
    # 提取x,y,acc值 shape为(25,)
    x = data[0].reshape(-1)
    y = data[1].reshape(-1)
    y = -y
    acc = data[2].reshape(-1)

    # 添加要画的连线
    seg = []
    seg.append([17, 15, 16, 18])
    seg.append([0, 1, 8])
    seg.append([8, 9, 10, 11, 22, 23])
    seg.append([8, 12, 13, 14, 19, 20])
    seg.append([4, 3, 2, 1, 5, 6, 7])
    seg.append([11, 24])
    seg.append([14, 21])

    plt.figure(0)
    for j in range(0, len(seg)):  # 画线
        plt.plot(x[seg[j]], y[seg[j]])
    plt.scatter(x, y, s=3)  # 画点
    dir_name = join('tmp_data_vis', str(action_idx))
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plt.savefig(join(dir_name, str(frame_idx) + '.png'), format='png')
    plt.close(0)


def vis_one_action(action_idx, data):
    T = get_frame_num(data)
    data = data[:, :T, :, :]
    print(data.shape)

    # 对每一帧进行处理
    for i in range(0, T):
        cur = data[:, i:i + 1]
        vis_one_frame(action_idx, i, cur)


if __name__ == "__main__":
    data = np.load('/data1/lzp/skating/fixed_test_A_data.npy')
    N, C, T, V, M = data.shape
    print(N, C, T, V, M)
    i=14
    vis_one_action(i,data[i])
    # for i in range(0,5):
        # vis_one_action(i,data[i])