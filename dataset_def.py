import math
import numpy
import numpy as np
import torch
from matplotlib import pyplot as plt, colors
from numpy import float32, log10, sqrt, flip
import h5py
from torch.utils.data import Dataset, DataLoader
import copy
from scipy.io import loadmat


def SlidingWindow(image, stride, window_size, window_center=None, padding=True, channel=False, print_show=True):
    # 输入图像（C， W， H）， 输入标签（W， H）
    # zero padding

    if channel:
        name = "data"
        a = numpy.zeros((6, window_size, window_size), dtype=numpy.complex64)

    else:
        name = "labels"
        a = numpy.zeros((window_size, window_size), dtype=int)
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
            image = numpy.pad(image, ((0, 0), (W_padding_ahead, W_padding_behind), (H_padding_ahead, H_padding_behind)),
                              'constant', constant_values=0)
        else:
            image = numpy.pad(image, ((W_padding_ahead, W_padding_behind), (H_padding_ahead, H_padding_behind)),
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
    # 得到截取样本list b，其中每个元素为滑动窗口截取的样本
    # position为返回的样本中心在整个image的位置


def get_mean_and_std(T):
    mean = np.zeros([6], dtype=np.complex64)
    std = np.zeros([6], dtype=np.complex64)
    for i in range(T.shape[0]):
        mean[i] = np.mean(T[i, :, :])

        if type(T[i, :, :]) == complex:
            std[i] = sqrt(((T[i, :, :] - mean) * np.conj(T[i, :, :] - mean)).sum() / np.size(T[i, :, :]))
        else:
            std[i] = np.std(T[i, :, :])

    return mean, std


class DataSet(Dataset):
    """   class_num:该数据集类别总数  """
    """   dataset_path：数据集所在文件夹路径   """
    """   delete_0：初始化数据集时删去标签为0的数据   """
    """   window_size: 窗口滑动的尺寸   """
    """   stride： 在整张图上进行窗口滑动截取样本数据时的步长   """
    """   window_center(输入元组)：截取样本的样本中心（默认为 window_size/2 ）"""
    '''   std: dataset是否进行标准化    '''

    def __init__(self, dataset_path=None, delete_0=True, window_size=12, stride=1, window_center=(6, 6), std=True,
                 print_show=True, class_num=0):
        self.delete_0 = delete_0
        # assert (window_center is None) or (0 <= int(window_center[0]) < window_size and 0 <= int(window_center[1]) < window_size)

        if dataset_path is not None:
            if 'Flevo1991' in dataset_path:
                self.name = 'Flevo1991'
                class_num = 14
            elif 'FlveoBig' in dataset_path:
                self.name = 'FlveoBig'
                class_num = 15
            elif 'Oberpfa' in dataset_path:
                self.name = 'Oberpfa'
                class_num = 3
            elif 'San900690' in dataset_path:
                self.name = 'San900690'
                class_num = 5
            else:
                self.name = None
                assert 0
            # 确定数据集所含类别数（不包括无标签的0类）
            self.class_num = class_num

            total_data, total_labels = ReadData(dataset_path, std=std)  # 返回T：(6, W, H) ，GT：(W, H)
            self.total_data = total_data
            self.total_labels = total_labels

            self.shape = total_labels.shape

            data_list, _, _ = SlidingWindow(total_data, stride, window_size, window_center, channel=True,
                                            print_show=print_show)
            labels_list, window_center, data_position = SlidingWindow(total_labels, stride, window_size, window_center,
                                                                      print_show=print_show)
            # 利用12*12（stride=1）的滑动窗口来构造对应训练数据,返回list

            # 去除掉其中的第0类(背景)
            data_list_without_0 = []
            labels_list_without_0 = []
            data_position_without_0 = []

            if delete_0:
                for i in range(len(labels_list)):
                    if labels_list[i][window_center] != 0:
                        data_list_without_0.append(data_list[i])
                        labels_list_without_0.append(labels_list[i])
                        data_position_without_0.append(data_position[i])

                print("Delete class 0！ Remained:{}".format(len(data_list_without_0)))
                self.data = data_list_without_0
                self.labels = labels_list_without_0
                self.position = data_position_without_0

            else:
                print("You haven't delete samples of 0 class!")
                self.data = data_list
                self.labels = labels_list
                self.position = data_position

            # l = []
            # for labels in self.labels:
            #     l.append(labels[window_center])
            # total_class_num = set(l)  # 统计数据集共含有哪些类别
            # print("该数据集含有这些类别：{}".format(total_class_num))

        else:
            self.data = []
            self.labels = []
            self.position = []
            self.class_num = class_num
            self.shape = (0, 0)
            self.total_data = np.zeros([6, self.shape[0], self.shape[1]])
            self.total_labels = np.zeros(self.shape)
        self.window_center = window_center

    def __getitem__(self, index):
        z = torch.torch.as_tensor(self.data[index])  # nparray --> torch
        label = copy.deepcopy(self.labels[index][self.window_center])
        # 如取 12*12 样本中的（6，6）的作为样本中心
        # 需要使用深拷贝，否则会改变原来的样本中心label

        label -= 1  # 如： 1-15类 要换成 0-14类
        if (not 0 <= label <= self.class_num) and self.delete_0:
            print("label={}:should 0 <= label <= self.class_num !!!".format(label))
            assert 0

        one_hot_label = one_hot(label, num_classes=self.class_num)
        return z, one_hot_label  # 返回复数数据及其复数one_hot编码

    def __len__(self):
        return len(self.data)

    def __get_mean_and_std__(self):
        # 得到该数据集所有数据的均值、方差
        mean, std = get_mean_and_std(self.total_data)  # 统计数据集中的均值和方差，用于标准化
        return mean, std

    def __get_position__(self):
        # 得到所有样本在image上的位置坐标
        return self.position

    def __get_class_num__(self):
        return self.class_num

    def __get_size__(self):
        return self.shape

    def __get_whole_image__(self):
        # 返回整张图的data、labels
        return self.total_data, self.total_labels

    def __get_name__(self):
        return self.name

    def __get_window_center__(self):
        return self.window_center


def one_hot(label, num_classes):
    ones = torch.eye(num_classes) * (1 + 1j)
    onehot = ones[label, :]
    return onehot.numpy()


def standardization(data):
    mean = numpy.mean(data)
    if type(data) == complex:
        std = sqrt(((data - mean) * numpy.conj(data - mean)).sum() / numpy.size(data))
    else:
        std = numpy.std(data)
    return (data - mean) / std


def load_matrix_from_file(file_path, load_list, shape, dtype, San900690=False):
    data_list = []
    for i in range(len(load_list)):
        load_path = file_path + load_list[i]
        data = ReadFile(load_path, shape, San900690)
        data_list.append(data)
    T11 = data_list[0]
    T12 = data_list[1] + 1j * data_list[2]
    T13 = data_list[3] + 1j * data_list[4]
    T22 = data_list[5]
    T23 = data_list[6] + 1j * data_list[7]
    T33 = data_list[8]
    return T11.astype(dtype), T12.astype(dtype), T13.astype(dtype), T22.astype(dtype), T23.astype(dtype), T33.astype(
        dtype)


def ReadFile(filepath, shape, San900690):
    """读取数据集中T矩阵各个元素"""
    # 注意： San900690 的data需要进行步长为2的降采样（切片符号为start:stop:step）！！！
    data = numpy.fromfile(filepath, dtype=float32)
    if San900690:
        data = data.reshape((shape[0] * 2, shape[1] * 2))
        data = data[0:data.shape[0] - 1:2, 0:data.shape[1] - 1:2]
    else:
        data = data.reshape(shape)
    return data


def ReadFileFlveoBig(filepath):
    data = np.fromfile(filepath, dtype=float32)
    data = data.reshape(750, 1024)
    return data


def ReadData(dataset_path, std=True, dtype=numpy.complex64):
    """  读取数据集：Flevo1991, FlveoBig, Oberpfa, San900690  """
    """  std：是否进行Z-Score标准化  """
    """  dtype：输出数据类型(复数 np.complex64)  """

    global T11, T12, T13, T22, T23, T33, GT
    if 'Flevo1991' in dataset_path:  # Flevo1991：1020*1024
        GT = loadmat(dataset_path + "/GT_Flevo1991_14C.mat")
        GT = GT['label'][:]  # #取出主键为data的所有的键值

        data_mat_path = dataset_path + "/T.mat"
        data = loadmat(data_mat_path)
        T11 = data['T11'][:].astype(dtype)
        T12 = data['T12'][:].astype(dtype)
        T13 = data['T13'][:].astype(dtype)
        T22 = data['T22'][:].astype(dtype)
        T23 = data['T23'][:].astype(dtype)
        T33 = data['T33'][:].astype(dtype)

    elif 'FlveoBig' in dataset_path:  # FlveoBig：750*1024

        GT = h5py.File(dataset_path + "/label.mat", 'r')
        GT = GT['label'][:]  # #取出主键为data的所有的键值

        load_list = ["T11.bin", 'T12_imag.bin', "T12_real.bin", "T13_imag.bin", "T13_real.bin", "T22.bin",
                     "T23_imag.bin", "T23_real.bin", "T33.bin"]
        # data_mat_path = dataset_path + "/LEE_T3/"

        # 提取数据特征（bin文件）
        bin_path = dataset_path + '/LEE_T3/T11.bin'
        T11 = ReadFileFlveoBig(bin_path)
        T11 = T11.transpose(1, 0)

        bin_path = dataset_path + '/LEE_T3/T12_imag.bin'
        T12_imag = ReadFileFlveoBig(bin_path)
        bin_path = dataset_path + '/LEE_T3/T12_real.bin'
        T12_real = ReadFileFlveoBig(bin_path)
        T12 = T12_real + 1j * T12_imag
        T12 = T12.transpose(1, 0)

        bin_path = dataset_path + '/LEE_T3/T13_imag.bin'
        T13_imag = ReadFileFlveoBig(bin_path)
        bin_path = dataset_path + '/LEE_T3/T13_real.bin'
        T13_real = ReadFileFlveoBig(bin_path)
        T13 = T13_real + 1j * T13_imag
        T13 = T13.transpose(1, 0)

        bin_path = dataset_path + '/LEE_T3/T22.bin'
        T22 = ReadFileFlveoBig(bin_path)
        T22 = T22.transpose(1, 0)

        bin_path = dataset_path + '/LEE_T3/T23_imag.bin'
        T23_imag = ReadFileFlveoBig(bin_path)
        bin_path = dataset_path + '/LEE_T3/T23_real.bin'
        T23_real = ReadFileFlveoBig(bin_path)
        T23 = T23_real + 1j * T23_imag
        T23 = T23.transpose(1, 0)

        bin_path = dataset_path + '/LEE_T3/T33.bin'
        T33 = ReadFileFlveoBig(bin_path)
        T33 = T33.transpose(1, 0)

    elif 'Oberpfa' in dataset_path:  # Oberpfa: 1300*1200
        GT = loadmat(dataset_path + "/GT_Oberpfa_3C.mat")
        GT = GT['label'][:]  # #取出主键为data的所有的键值

        load_list = ["T11.bin", 'T12_imag.bin', "T12_real.bin", "T13_imag.bin", "T13_real.bin", "T22.bin",
                     "T23_imag.bin", "T23_real.bin", "T33.bin"]
        data_mat_path = dataset_path + "/LEE_T3/"
        T11, T12, T13, T22, T23, T33 = load_matrix_from_file(data_mat_path, load_list, GT.shape, dtype)

    elif 'San900690' in dataset_path:  # San900690: 1800*1380
        GT = loadmat(dataset_path + "/GT_Sanfran_5Classes.mat")
        GT = GT['GT'][:]  # #取出主键为data的所有的键值

        load_list = ["T11.bin", 'T12_imag.bin', "T12_real.bin", "T13_imag.bin", "T13_real.bin", "T22.bin",
                     "T23_imag.bin", "T23_real.bin", "T33.bin"]
        data_mat_path = dataset_path + "/LEE_T3/"
        T11, T12, T13, T22, T23, T33 = load_matrix_from_file(data_mat_path, load_list, GT.shape, dtype, San900690=True)

    if std:
        # 是否对T矩阵的输入参数进行标准化
        T11 = standardization(T11)
        T12 = standardization(T12)
        T13 = standardization(T13)
        T22 = standardization(T22)
        T23 = standardization(T23)
        T33 = standardization(T33)
    T_matrix = numpy.dstack((T11, T12, T13, T22, T23, T33))  # (1024, 750, 6)

    if 'Flevo1991' in dataset_path or 'Oberpfa' in dataset_path:
        T_matrix = numpy.swapaxes(T_matrix, 0, 1)
        T_matrix = numpy.flip(T_matrix, 1)

        GT = numpy.swapaxes(GT, 0, 1)
        GT = numpy.flip(GT, 1)

    elif 'FlveoBig' in dataset_path:  # FlveoBig: 1024 * 750
        T_matrix = numpy.flip(T_matrix, 1)
        GT = numpy.flip(GT, 1)

    elif 'San900690' in dataset_path:  # San900690: 1800*1380-->900*690
        T_matrix = numpy.swapaxes(T_matrix, 0, 1)
        T_matrix = numpy.flip(T_matrix, 1)  # 这里T矩阵跟GT不对应，需要沿着长边（W=900）翻转一次！！！

        GT = numpy.swapaxes(GT, 0, 1)

    T_matrix = numpy.transpose(T_matrix, (2, 0, 1))  # 重新指定轴的顺序:(6, W, H)

    return T_matrix, GT


def extract_physical_features(T11, T12, T13, T22, T23, T33):
    """ 提取物理特征 """
    # SPAN为能量，为sqrt(T11 ^ 2 + T22 ^ 2 + T33 ^ 2); 表示为db形式
    SPAN = T11 + T22 + T33  # sqrt呢?????

    # 六维特征
    modT12 = abs(T12)
    modT13 = abs(T13)
    modT23 = abs(T23)

    A = 10 * log10(SPAN)
    B = T22 / SPAN
    C = T33 / SPAN
    D = modT12 / sqrt(T11 * T22)
    E = modT13 / sqrt(T11 * T33)
    F = modT23 / sqrt(T22 * T33)  # 逐元素相乘、除、开方对吗？

    B = (B - B.min()) / (B.max() - B.min())
    C = (C - C.min()) / (C.max() - C.min())
    D = (D - D.min()) / (D.max() - D.min())
    E = (E - E.min()) / (E.max() - E.min())
    F = (F - F.min()) / (F.max() - F.min())

    Feature = numpy.dstack((A, B, C, D, E, F))
    # 在第三维度上进行深度拼接，得到（W，H，6）的数组Feature
    # Feature = np.transpose(Feature, (2, 0, 1))
    return Feature


def set_colors_bar(dataset_name):
    """设置颜色条"""
    global colors_list, label_list, sm


    if 'FlveoBig' in dataset_name:
        colors_list = ['white', 'navy', 'darkred', 'indigo', 'red', 'm', 'forestgreen', 'darkgoldenrod', 'lime',
                       'orange', 'cyan', 'orchid', 'pink', 'navajowhite', 'yellow', 'lightgreen']

        label_list = ['', '1:Water', '2:Barley', '3:Peas', '4:Stembeans', '5:Beet', '6:Forest',
                      '7:Bare soil', '8:Grass',
                      '9:Rapeseed', '10:Lucerne', '11:Wheat 2', '12:Wheat', '13:Buildings', '14:Potatoes', '15:Wheat 3']


    assert len(colors_list) == len(label_list)
    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 设置c_map
    c_map = colors.LinearSegmentedColormap.from_list('mylist', colors_list, N=len(colors_list))
    sm = plt.cm.ScalarMappable(cmap=c_map)
    return sm, label_list, c_map


def draw_ground_truth(full_dataset):
    # 读取GT
    _, GT = full_dataset.__get_whole_image__()
    dataset_name = full_dataset.__get_name__()

    # 画mesh网格图
    x = numpy.arange(GT.shape[0])
    y = numpy.arange(GT.shape[1])
    x, y = numpy.meshgrid(x, y)
    z = GT[x, y]

    # 添加颜色条
    sm, label_list, c_map = set_colors_bar(dataset_name)

    # 绘图GT
    plt.figure(figsize=(GT.shape[0] / 100, GT.shape[1] / 100))
    plt.pcolor(x, y, z, cmap=c_map)
    plt.title("{}\n图像大小：{}\n".format(dataset_name, GT.shape), fontsize=20)

    # 去除坐标刻度
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    plt.grid(True)

    # 设置colorbar
    cbar = plt.colorbar(sm, ticks=numpy.linspace(0, 1, len(label_list), endpoint=False) + 1 / (2 * len(label_list)),
                        label='Classes', )
    cbar.ax.set_yticklabels(label_list)
    cbar.ax.axes.tick_params(length=0)

    plt.show()


if __name__ == '__main__':
    # 设置数据集
    dataset_path1 = "../benchmark_dataset/Flevo1991"  # 14类  span与GT对应！！！
    dataset_path2 = "FlveoBig"  # 15类 span与GT对应！！！
    dataset_path3 = "../benchmark_dataset/Oberpfa"  # 3类 span与GT对应！！！
    dataset_path4 = "../benchmark_dataset/San900690"  # 5类 这里T矩阵跟GT不对应，需要沿着长边（W=900）翻转一次！！！
    dataset_path = dataset_path2

    # 测试数据集的定义。。。。
    train_data = DataSet(dataset_path=dataset_path, stride=10, std=False)
    # draw_ground_truth(train_data)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    T, GT = ReadData(dataset_path=dataset_path)
    dataset_name = train_data.__get_name__()

    # 画mesh网格图
    x = numpy.arange(GT.shape[0])
    y = numpy.arange(GT.shape[1])
    x, y = numpy.meshgrid(x, y)
    z = GT[x, y]

    # 添加颜色条
    sm, label_list, c_map = set_colors_bar(dataset_name)

    # 绘图GT
    plt.figure(figsize=(GT.shape[0] / 100, GT.shape[1] / 100))
    plt.pcolor(x, y, z, cmap=c_map)
    plt.title("{}\n图像大小：{}\n".format(dataset_name, GT.shape), fontsize=20)

    # # 去除坐标刻度
    # ax = plt.gca()
    # ax.axes.xaxis.set_ticks([])
    # ax.axes.yaxis.set_ticks([])
    # plt.grid(True)

    # 设置colorbar
    cbar = plt.colorbar(sm, ticks=numpy.linspace(0, 1, len(label_list), endpoint=False) + 1 / (2 * len(label_list)),
                        label='Classes', )
    cbar.ax.set_yticklabels(label_list)
    cbar.ax.axes.tick_params(length=0)

    plt.show()
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    SPAN = abs((T[0, :, :]) + (T[3, :, :]) + (T[5, :, :]))
    # SPAN = (SPAN-np.min(SPAN))/(np.max(SPAN)-np.min(SPAN))
    SPAN = standardization(SPAN)

    print(SPAN.max(), SPAN.min())
    SPAN = SPAN * 255  # 变换为0-255的灰度值
    print(SPAN)

    # # 画mesh网格图
    # x = numpy.arange(SPAN.shape[0])
    # y = numpy.arange(SPAN.shape[1])
    # x, y = numpy.meshgrid(x, y)
    # z = SPAN[x,y]
    # # 绘图: SPAN!!!
    # plt.figure(figsize=(SPAN.shape[0] / 100, SPAN.shape[1] / 100))
    # plt.pcolor(x, y, z, cmap='gray')
    # plt.show()

    from PIL import Image

    im = Image.fromarray(SPAN)
    im = im.convert('L').transpose(Image.ROTATE_90)  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
    plt.figure(figsize=(GT.shape[0] / 100, GT.shape[1] / 100))
    plt.title("{}---SPAN\n图像大小：{}\n".format(dataset_name, GT.shape), fontsize=20)
    plt.imshow(im)
    plt.show()

    # # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # mean, std = train_data.__get_mean_and_std__()
    # print(train_data.__get_mean_and_std__())
    # train_dataloader = DataLoader(train_data, 128)
    # for data in train_dataloader:
    #     imgs, targets = data
    #     print(abs(targets).argmax(1))
