import numpy as np
import math
import copy
import pickle

def SlidingWindow(image, stride, window_size, window_center=None, padding=True, channel=False, print_show=True):
    # 输入图像（C， W， H）， 输入标签（W， H）
    # zero padding

    if channel:
        name = "data"
        a = np.zeros((9, window_size, window_size), dtype=np.float32)

    else:
        name = "labels"
        a = np.zeros((window_size, window_size), dtype=int)
    width = image.shape[-2]
    height = image.shape[-1]
    total_num = math.ceil(width / stride) * math.ceil(height / stride)
    # 为什么是向上取整？？？？
    # total_num: SlidingWindow截得的样本总数

    if padding:
        if window_center is None:
            if window_size % 2:  # 如果window_size为奇数,设定（int(window_size/2)，int(window_size/2)）为样本中心
                W_padding_ahead = W_padding_behind = H_padding_ahead = H_padding_behind = int(window_size / 2)
                window_center = (W_padding_ahead + 1, H_padding_ahead + 1)

            else:  # 如果window_size为偶数,设定（(window_size/2)，(window_size/2)）为样本中心
                W_padding_ahead = H_padding_ahead = int(window_size / 2)
                W_padding_behind = H_padding_behind = int(window_size / 2) - 1
                window_center = (W_padding_ahead, H_padding_ahead)
        else:
            H_padding_ahead = window_center[1]
            H_padding_behind = window_size - window_center[1] - 1
            W_padding_ahead = window_center[0]
            W_padding_behind = window_size - window_center[0] - 1

        if channel:
            image = np.pad(image, ((0, 0), (W_padding_ahead, W_padding_behind), (H_padding_ahead, H_padding_behind)),
                              'constant', constant_values=0)
        else:
            image = np.pad(image, ((W_padding_ahead, W_padding_behind), (H_padding_ahead, H_padding_behind)),
                              'constant',
                              constant_values=0)
            # data zero padding(如：样本中心（6，6）时，前补6位，后补5位)
    position = []
    b = []
    num = 0
    for i in range(0, width, stride):
        for j in range(0, height, stride):
            for m in range(window_size):
                for n in range(window_size):
                    if channel:
                        a[:, m, n] = image[:, i + m, j + n]
                    else:
                        a[m, n] = image[i + m, j + n]
            position.append((i, j))
            b.append(copy.deepcopy(a))
            # 使用 append() 函数添加列表时，是添加列表的「引用地址」而不是添加列表内容，当被添加的列表发生变化时，添加后的列表也会同步发生变化;
            # 使用「深拷贝」添加列表的内容而不是引用地址，从而解决列表同步的问题
            num += 1
            if print_show:
                print("\rCollecting {} ...{:.2%} total:{}".format(name, num / total_num, num), end="")
    print(" Done! total:{}".format(num))
    return b, window_center, position
def delete_0(data, labels, window_center):
    data_without_0 = []
    labels_without_0 = []
    labels_patch = []
    for i in range(len(labels)):
        if labels[i][window_center] != 0:
            data_without_0.append(data[i])
            labels_without_0.append(labels[i])
            labels_patch.append(labels[i][window_center])
    print("Delete class 0！ Remained:{}".format(len(data_without_0)))
    return data_without_0, labels_patch
            
if __name__ == '__main__':
    with open('SLIC_out/Graph/T_notstd.pkl', 'rb') as f:
        T = pickle.load(f)
    with open('SLIC_out/Graph/GT_notstd.pkl', 'rb') as f:
        GT = pickle.load(f)
    patch, _, _ = SlidingWindow(T.transpose(2, 0, 1), 1, 12, channel=True)
    label, window_center, _ = SlidingWindow(GT, 1, 12)
    patch = np.array(patch)
    data, labels = delete_0(patch, label, window_center)
    with open('SLIC_out/Graph/patch_notstd_without0.pkl', 'wb') as f:
        pickle.dump(data, f)
    with open('SLIC_out/Graph/label_notstd_without0.pkl', 'wb') as f:
        pickle.dump(labels, f)