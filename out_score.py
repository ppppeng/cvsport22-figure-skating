import numpy as np
import csv


def get_model(model_name, power, test_window_size, power_window_size):
    """
    get score of model
    Args:
        model_name: str, we use model_name to find the folder
        power:  ndarray, power of each fold to fuse
        test_window_size: list, the window_size we selected to fuse
        power_window_size: ndarray, power of each window_size to fuse

    Returns:
        score: score of model
    """
    out = np.zeros((634, 30), dtype='float32')
    power = power / power.sum()
    power_window_size = power_window_size / power_window_size.sum()

    for i in range(0, 5):
        for j in range(0, len(test_window_size)):
            path = "./output/%s_fold%s/score_%s.npy" % (model_name, i, test_window_size[j])
            print(path)
            data = np.load(path)
            out = out + data * power[i] * power_window_size[j]
    return out


def get_agcn():
    model_name = 'AGCN'
    power = np.array([1.5, 1, 1.5, 1, 2], dtype='float32')
    test_window_size = [300, 350, 500]
    power_window_size = np.array([1, 1, 1], dtype='float32')
    return get_model(model_name, power, test_window_size, power_window_size)


def get_ctrgcn():
    model_name = 'CTRGCN'
    power = np.array([1, 1, 2, 2, 3], dtype='float32')
    test_window_size = [300, 350, 450]
    power_window_size = np.array([1, 1, 1], dtype='float32')
    return get_model(model_name, power, test_window_size, power_window_size)


def get_ctrgcn2():
    model_name = 'CTRGCN2'
    power = np.array([1, 1, 2, 2, 3], dtype='float32')
    test_window_size = [350]
    power_window_size = np.array([1], dtype='float32')
    return get_model(model_name, power, test_window_size, power_window_size)


def get_stgcn():
    model_name = 'STGCN'
    power = np.array([1, 1, 2, 1, 2], dtype='float32')
    test_window_size = [350]
    power_window_size = np.array([1], dtype='float32')
    return get_model(model_name, power, test_window_size, power_window_size)


def get_effgcn():
    model_name = 'EFFGCN'
    power = np.array([1, 1, 2, 1, 3], dtype='float32')
    test_window_size = [500]
    power_window_size = np.array([1], dtype='float32')
    return get_model(model_name, power, test_window_size, power_window_size)


def get_all():
    out = np.zeros((634, 30), dtype='float32')
    data = []
    data.append(get_agcn())
    data.append(get_ctrgcn())
    data.append(get_ctrgcn2())
    data.append(get_stgcn())
    data.append(get_effgcn())

    power = np.array([2, 0.75, 1.5, 3.75, 2], dtype='float32')
    power = power / power.sum()
    for i in range(0, len(data)):
        out = out + data[i] * power[i]
    return out


def get_values_for_csv(data):
    values = []
    for i in range(0, data.shape[0]):
        j = np.argmax(data[i], axis=0)
        values.append((i, j))
    return values


def out_file(data, out_file):
    with open(
            out_file,
            'w',
    ) as fp:
        writer = csv.writer(fp)
        if out_file == "submission.csv":
            writer.writerow(['sample_index', 'predict_category'])
        writer.writerows(data)


if __name__ == "__main__":
    out_file(get_values_for_csv(get_all()), 'submission.csv')
