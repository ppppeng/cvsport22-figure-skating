import os.path

import numpy as np
from os.path import basename, dirname, join


def K_fold(data_path, label_path, dir_path, k=5):
    '''
    This function aims to divide data into fold for cross validation
    Args:
        data_path: path to load data
        label_path: path to load label
        dir_path: path to save output
        k: number of folds

    '''
    from sklearn.model_selection import StratifiedKFold

    data = np.load(data_path)
    label = np.load(label_path)
    os.makedirs(dir_path) if not os.path.exists(dir_path) else None

    kf = StratifiedKFold(n_splits=k, shuffle=False).split(data, label)
    # 分成5份的下标
    train_data_index = []
    val_data_index = []
    for fold, (trn_idx, val_idx) in enumerate(kf):
        train_data_index.append(trn_idx)
        val_data_index.append(val_idx)

    for fold in range(0, k):
        fold_train_data = data[train_data_index[fold]]
        fold_train_label = label[train_data_index[fold]]
        fold_valid_data = data[val_data_index[fold]]
        fold_valid_label = label[val_data_index[fold]]
        np.save("%s/fold%s_train_data.npy" % (dir_path, fold), fold_train_data)
        np.save("%s/fold%s_train_label.npy" % (dir_path, fold), fold_train_label)
        np.save("%s/fold%s_valid_data.npy" % (dir_path, fold), fold_valid_data)
        np.save("%s/fold%s_valid_label.npy" % (dir_path, fold), fold_valid_label)


def get_frame_num(data):
    C, T, V, M = data.shape
    for i in range(T - 1, -1, -1):
        tmp = np.sum(data[:, i, :, :])
        if not tmp == 0:
            T = i + 1
            break
    return T


def clear_one_action(data):
    '''
        修复nan问题
        把数据中间置信度全为0的帧删去
    '''
    zero = np.zeros(data.shape, dtype='float32')
    T = get_frame_num(data)
    data = data[:, :T]
    select_nan = np.isnan(data)
    data[select_nan] = 0
    confi = data[2:3]
    select = ~(confi.max(axis=2) == 0).reshape(-1)
    true_T = select.sum()
    zero[:, 0:true_T] = data[:, select]
    return zero


def fix_one_action(data):
    '''
        修补中心点(下标8)坐标，防止中心化失效
    '''
    T = get_frame_num(data)
    pre = data
    data = data[:, :T]
    # C,T,V,M
    select = data[2:3, :, 8:9].reshape(-1)
    select = select == 0
    count = select.sum()
    if count == 0:
        return pre
    # print(count)
    select_data = data[:, select]
    ans = None
    for t in range(0, count):
        tmp = select_data[:, t]
        mid_select = tmp[2:3].reshape(-1) > 0
        if mid_select.sum() == 0:
            mid = np.zeros((3, 1))
        else:
            mid = tmp[:, mid_select].mean(axis=1)
        if ans is None:
            ans = mid
        else:
            ans = np.concatenate((ans, mid), axis=1)
    if not ans is None:
        ans = ans.reshape((3, count, 1, 1))
        data[:, select, 8:9] = ans
    pre[:, :T] = data
    return pre


def Norm_one_action(data):
    '''
        中心化，只中心化置信度非0的点
    '''
    # C, T, V, M = data.shape
    T = get_frame_num(data)
    for i in range(0, T):
        confidence = data[2, i, :, 0]
        select = confidence > 0
        if select.sum() == 0:
            continue
        data[0:2, i, select] = data[0:2, i, select] - data[0:2, i, 8:9]
    return data


def update_one_file_fixed1(path):
    print(path)
    data = np.load(path)
    dir = dirname(path)
    file = basename(path)

    for i in range(0, data.shape[0]):
        if (i % 100 == 0):
            print("%s / %s" % (i, data.shape[0]))
        action = data[i]
        action = clear_one_action(action)
        pre_frame = get_frame_num(action)
        action = fix_one_action(action)
        action = Norm_one_action(action)
        if not get_frame_num(action) == pre_frame:
            print(i, "warning!")
        data[i] = action

    target_path = dir + '/fixed1_' + file
    np.save(target_path, data)
    return target_path


def update_one_file_fixed2(path):
    print(path)
    data = np.load(path)
    dir = dirname(path)
    file = basename(path)

    for i in range(0, data.shape[0]):
        if (i % 100 == 0):
            print("%s / %s" % (i, data.shape[0]))
        action = data[i]
        # action = clear_one_action(action)
        pre_frame = get_frame_num(action)
        action = fix_one_action(action)
        action = Norm_one_action(action)
        if not get_frame_num(action) == pre_frame:
            print(i, "warning!")
        data[i] = action

    target_path = dir + '/fixed2_' + file
    np.save(target_path, data)
    return target_path


if __name__ == "__main__":
    base_path = "/home/data2/lzp_dataset/skating"
    train_data_path = join(base_path, "train_data.npy")
    train_label_path = join(base_path, "train_label.npy")
    # test_data_path = join(base_path, "test_B_data.npy")

    # 对数据集进行修复，fixed1删去空帧，fixed2不删空帧
    # fixed1_train_data_path = update_one_file_fixed1(train_data_path)
    # fixed2_train_data_path = update_one_file_fixed2(train_data_path)
    # update_one_file_fixed1(test_data_path)
    # update_one_file_fixed2(test_data_path)

    # 生成五折数据集
    # K_fold(fixed1_train_data_path, train_label_path, dir_path=join(base_path, "fixed1_fold"), k=5)
    K_fold(train_data_path, train_label_path, dir_path=join(base_path, "origin_fold"), k=5)
    # K_fold(fixed2_train_data_path, train_label_path, dir_path=join(base_path, "fixed2_fold"), k=5)
