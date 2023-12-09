from math import inf
import numpy
import numpy as np
from numpy import float32, sqrt
import h5py
import torch
from torch import nn
from torch.nn import functional as F
import cv2
import dgl
import pickle
from matplotlib import colors as colors


def standardization(data):
    mean = numpy.mean(data)
    if type(data) == complex:
        std = sqrt(((data - mean) * numpy.conj(data - mean)).sum() / numpy.size(data))
    else:
        std = numpy.std(data)
    return (data - mean) / std

def ReadFileFlveoBig(filepath):
    data = np.fromfile(filepath, dtype=float32)
    data = data.reshape(750, 1024)
    return data


def ReadData(dataset_path, std=True, dtype=numpy.complex64):
    """  读取数据集：Flevo1991, FlveoBig, Oberpfa, San900690  """
    """  std：是否进行Z-Score标准化  """
    """  dtype：输出数据类型(复数 np.complex64)  """

    GT = h5py.File(dataset_path + "/label.mat", 'r')
    GT = GT['label'][:]  # #取出主键为data的所有的键值

    # data_mat_path = dataset_path + "/LEE_T3/"

    # 提取数据特征（bin文件）
    bin_path = dataset_path + '/LEE_T3/T11.bin'
    T11 = ReadFileFlveoBig(bin_path)
    T11 = T11.transpose(1, 0)
    bin_path = dataset_path + '/LEE_T3/T12_imag.bin'
    T12_imag = ReadFileFlveoBig(bin_path)
    bin_path = dataset_path + '/LEE_T3/T12_real.bin'
    T12_real = ReadFileFlveoBig(bin_path)
    T12_real, T12_imag = T12_real.transpose(1, 0), T12_imag.transpose(1, 0)
    T12 = T12_real + 1j * T12_imag
    T12 = T12.transpose(1, 0)

    bin_path = dataset_path + '/LEE_T3/T13_imag.bin'
    T13_imag = ReadFileFlveoBig(bin_path)
    bin_path = dataset_path + '/LEE_T3/T13_real.bin'
    T13_real = ReadFileFlveoBig(bin_path)
    T13_real, T13_imag = T13_real.transpose(1, 0), T13_imag.transpose(1, 0)
    T13 = T13_real + 1j * T13_imag
    T13 = T13.transpose(1, 0)

    bin_path = dataset_path + '/LEE_T3/T22.bin'
    T22 = ReadFileFlveoBig(bin_path)
    T22 = T22.transpose(1, 0)

    bin_path = dataset_path + '/LEE_T3/T23_imag.bin'
    T23_imag = ReadFileFlveoBig(bin_path)
    bin_path = dataset_path + '/LEE_T3/T23_real.bin'
    T23_real = ReadFileFlveoBig(bin_path)
    T23_real, T23_imag = T23_real.transpose(1, 0), T23_imag.transpose(1, 0)
    T23 = T23_real + 1j * T23_imag
    T23 = T23.transpose(1, 0)

    bin_path = dataset_path + '/LEE_T3/T33.bin'
    T33 = ReadFileFlveoBig(bin_path)
    T33 = T33.transpose(1, 0)

    if std:
        # 是否对T矩阵的输入参数进行标准化
        T11 = standardization(T11)
        T12 = standardization(T12)
        T13 = standardization(T13)
        T22 = standardization(T22)
        T23 = standardization(T23)
        T33 = standardization(T33)
    T_matrix = numpy.dstack((T11, T12_real, T12_imag, T13_real, T13_imag, T22, T23_real, T23_imag, T33))  # (1024, 750, 9)
    # T_matrix = numpy.flip(T_matrix, 1)
    # GT = numpy.flip(GT, 1)
    draw(GT, 11)
    for i in range(15):
        a = GT == i+1
        print(i+1, ' ', a.sum())
    return T_matrix,  GT

def get_RGB(path):
    img = cv2.imread(path)
    return torch.transpose(torch.tensor(img), 0, 1)

class SLIC(): # SLIC分割时用RGB图像，之后再用遥感数据
    def __init__(self, h, w, k, m, RGB_feature, SAR_featrue, ground_truth, device): # 这里的k不一定为最后的k的数值，因为要开方、除等操作
        self.device = device
        self.h = h # 1024
        self.w = w # 750
        self.m = m
        local = torch.cat([torch.arange(self.h).view(-1, 1).unsqueeze(1).expand(h, w, 1), torch.arange(self.w).view(-1, 1).unsqueeze(0).expand(h, w, 1)], dim=-1)
        self.RGB_feature = torch.cat((local, RGB_feature), dim=2).to(self.device)
        self.SAR_feature = torch.tensor(SAR_featrue.copy()).to(self.device)
        self.ground_truth = torch.tensor(ground_truth.copy()).to(self.device)

        self.label = torch.full((h, w), -1).to(self.device)
        self.distance = torch.full((h, w), float(inf)).to(self.device)
        center_h = h / k**0.5
        center_w = w / k**0.5
        center_distence = (h * w / k) ** 0.5
        center_h_num = int(h / center_distence) + 1
        center_w_num = int(w / center_distence) + 1
        self.k = center_h_num * center_w_num # k可能与输入不同
        self.center = torch.zeros((self.k, 5)).to(self.device) # 每一个聚类中心有5个属性，分别为x、y与RGB
        self.s = ((h * w / self.k)**0.5)
        self.center_SAR = torch.zeros((self.k, 11), dtype=torch.float32).to(self.device) # 这里从complex改到float
        self.center_label = torch.zeros((self.k, ), dtype=torch.int8).to(self.device)
        for i in range(center_h_num): # 初始化生成聚类中心，此时先不考虑移动到梯度最小地方的问题
            for j in range(center_w_num):
                self.center[i*center_w_num + j][0:2] = torch.tensor((round((i+0.5)*center_distence), round((j+0.5)*center_distence)))
        
    def kmeans(self, times):
        # for time in range(times):
        time = 0
        while True:
            most = 0
            for n, center in enumerate(self.center):
                x_min = int(max(0, center[0] - self.s))
                x_max = int(min(self.h, center[0] + self.s))
                y_min = int(max(0, center[1] - self.s))
                y_max = int(min(self.w, center[1] + self.s))
                # if y_max == 
                search_space = self.RGB_feature[x_min: x_max, y_min: y_max]
                center_distance = distance(search_space, center, self.s, self.m)
                search_space_distance = self.distance[x_min: x_max, y_min: y_max]
                nearer = center_distance < search_space_distance # 生成一个bool的tensor
                if nearer.sum() > most:
                    most = nearer.sum()
                print('\rSLIC epoch{} {:.2%}\tgenerating with changing pixel{:02d} most{:02d}'.format(time, n/len(self.center), nearer.sum(), most), end='')
                search_space_distance[nearer] = center_distance[nearer]
                self.distance[x_min: x_max, y_min: y_max] = search_space_distance
                search_space_label = self.label[x_min: x_max, y_min: y_max]
                search_space_label[nearer] = n
                self.label[x_min: x_max, y_min: y_max] = search_space_label
            for n in range(self.k):
                label_n = self.label == n # 神奇的操作，可做出bool张量
                super_pixel = self.RGB_feature[label_n]
                self.center[n] = torch.mean(super_pixel.float(), dim=0)
            time += 1
            if most == 0:
                print('')
                break
            elif time >= times:
                break
        null_data = 0
        for n in range(self.k):
            label_n = self.label == n
            super_pixel = self.SAR_feature[label_n]
            self.center_SAR[n] = torch.cat((self.center[n, :2], torch.mean(super_pixel, dim=0)))
            super_pixel_label = self.ground_truth[label_n]
            label, counts = torch.unique(super_pixel_label, return_counts=True) # 返回每一个标签及其数目
            if label.numel():
                most_label = label[torch.argmax(counts)]
            else:
                null_data += 1
                continue
            if most_label != 0:
                self.center_label[n] = most_label
        print('null node', null_data)
        print(self.label.numel())
        return self.center_SAR, self.center_label, self.label, self.center

def distance(search_space, center, s, m):
    space_d = ((search_space[:, :, 0] - center[0])**2 + (search_space[:, :, 1] - center[1])**2)**0.5
    color_d = ((search_space[:, :, 2] - center[2])**2 + (search_space[:, :, 3] - center[3])**2 + (search_space[:, :, 4] - center[4])**2)**0.5
    return ((color_d / m)**2 + (space_d / s)**2)**0.5
        
def euclidean(a, b):
    a = a.to(float)
    b = b.to(float)
    return ((a[0] - b[:, 0])**2 + (a[1] - b[:, 1])**2)**0.5

def get_graph(path, RGB_path, k, m, t, s, train_rate, val_rate, device): # k-superpixel_num, m-hyperparameter in SLIC, t-epoch in SLIC, s-hyperparameter when seting edge
    # T, GT = ReadData(path)
    global h, w
    data_RGB = get_RGB(RGB_path)
    h = data_RGB.shape[0]
    w = data_RGB.shape[1]
    T,  GT = ReadData('Dataset')
    SLIC_net = SLIC(h, w, k, m, data_RGB, T, GT, device=device)
    centers, center_label, label, center_RGB = SLIC_net.kmeans(t)

    # 检验SLIC输出的操作
    new = torch.zeros((h, w, 3))
    for n,c in enumerate(center_RGB):
        one_label = label == n
        new[one_label] = c[2:].to('cpu')
    new = torch.transpose(new, 0, 1).int().numpy().astype(np.uint8)
    # cv2.imshow('123', new)
    # cv2.waitKey(0)
    cv2.imwrite('SLIC_out/image/k={},m={},t={}.jpg'.format(k, m, t), new)
    
    k_ture = len(centers)
    location = centers[:, :2]
    node_feature = centers[:, 2:].clone().to(device)
    
    # 使用距离作为判断是否加边的依据
    '''
    threshold = (h * w / k)**0.5 * s # 超像素间平均距离 * s
    nodes = torch.arange(0, k_ture).to(device)
    for n, center in enumerate(centers):
        e_distance = euclidean(center[:2], location) # 返回一个结点与每个结点的欧氏距离
        new_edge_dst = nodes[e_distance < threshold]
        # new_edge_dst = new_edge_dst[new_edge_dst != n] # 若不需要自环去掉注释
        new_edge_src = torch.full(new_edge_dst.shape, n).to(device)
        if n == 0:
            edge_dst = new_edge_dst
            edge_src = new_edge_src
        else:
            edge_dst = torch.cat((edge_dst, new_edge_dst))
            edge_src = torch.cat((edge_src, new_edge_src))
    edge = torch.stack([edge_src, edge_dst]).to(device)
    '''
    
    # 使用是否相邻作为是否加边的依据
    flag = 0
    for x in range(w):
        for y in range(h):
            curr_pixel_label = label[y, x]
            right_pixel_label = None if x == w - 1 else label[y, x+1]
            down_pixel_label = None if y == h - 1 else label[y+1, x]
            if right_pixel_label:
                if curr_pixel_label != right_pixel_label:
                    if flag == 0:
                        edge = torch.tensor([[curr_pixel_label, right_pixel_label]])
                        edge = torch.cat([edge, torch.tensor([[right_pixel_label, curr_pixel_label]])], 0)
                        flag = 1
                    else:
                        edge = torch.cat([edge, torch.tensor([[curr_pixel_label, right_pixel_label]])], 0)
                        edge = torch.cat([edge, torch.tensor([[right_pixel_label, curr_pixel_label]])], 0)
            if down_pixel_label:
                if curr_pixel_label != down_pixel_label:
                    if flag == 0:
                        edge = torch.tensor([[curr_pixel_label, down_pixel_label]])
                        edge = torch.cat([edge, torch.tensor([[down_pixel_label, curr_pixel_label]])], 0)
                        flag = 1
                    else:
                        edge = torch.cat([edge, torch.tensor([[curr_pixel_label, down_pixel_label]])], 0)
                        edge = torch.cat([edge, torch.tensor([[down_pixel_label, curr_pixel_label]])], 0)
    edge = edge.tolist()
    edge = set(map(tuple,edge))
    edge = torch.tensor(list(edge))
    edge = edge.T
            
    
    # graph = pyg_Graph(x=node_feature, edge_index=edge, y=center_label)
    graph = dgl.graph(data=(edge[0], edge[1])).to(device)
    graph.ndata['label'] = center_label
    graph.ndata['feat'] = node_feature
    have_label = center_label != 0
    have_label_index = torch.nonzero(have_label)
    h = have_label.sum()
    rand_index = torch.randperm(h)

    graph.ndata['train_mask'] = torch.full((k_ture,), False).to(device)
    graph.ndata['val_mask'] = torch.full((k_ture,), False).to(device)
    graph.ndata['test_mask'] = torch.full((k_ture,), False).to(device)

    graph.ndata['train_mask'][have_label_index[rand_index[: int(train_rate*h)]]] = True
    graph.ndata['val_mask'][have_label_index[rand_index[int(train_rate*h): int(train_rate*h)+int(val_rate*h)]]] = True
    graph.ndata['test_mask'][have_label_index[rand_index[int(train_rate*h)+int(val_rate*h):]]] = True

    with open('SLIC_out/Graph/graph_of_k={}_m={}_t={}_s={}_trainrate{}_val_rate{}.pkl'.format(k, m, t, s, train_rate, val_rate), 'wb') as f:
        pickle.dump(graph, f)
    with open('SLIC_out/Graph/pixel_label_of_k={}_m={}_t={}_s={}_trainrate{}_val_rate{}.pkl'.format(k, m, t, s, train_rate, val_rate), 'wb') as f:
        pickle.dump(label, f)
    with open('SLIC_out/Graph/pixel_feature_of_k={}_m={}_t={}_s={}_trainrate{}_val_rate{}.pkl'.format(k, m, t, s, train_rate, val_rate), 'wb') as f:
        pickle.dump(T, f)
    with open('SLIC_out/Graph/pixel_GT.pkl', 'wb') as f:
        pickle.dump(GT, f)

    return graph, label, T,  GT

def draw(pred, seed):
    a = pred == 14
    x = a.sum()
    colors_list = ['white','navy', 'darkred', 'indigo', 'red', 'm', 'forestgreen', 'darkgoldenrod', 'lime',
                       'orange', 'cyan', 'orchid', 'pink', 'navajowhite', 'yellow', 'lightgreen']
    class_colors = torch.tensor([colors.to_rgb(color) for color in colors_list]) * 255
    b = class_colors[14]
    c = torch.zeros((10, 10))
    c[:, :, 0] = b[0]
    c[:, :, 1] = b[1]
    c[:, :, 2] = b[2]
    cv2.imwrite('GraphMAE_out/out of seed{}.jpg'.format(123), c.int().numpy().astype(np.uint8))
    '''
    class_colors = torch.tensor([[255, 255, 255],
                                 [0, 0, 128],
                                 [139, 0, 0],
                                 [75, 0, 130],
                                 [255, 0, 0],
                                 [255, 0, 255],
                                 [34, 139, 34],
                                 [184, 134, 11],
                                 [0, 255, 0],
                                 [255, 165, 0],
                                 [0, 255, 255],
                                 [218, 112, 214],
                                 [255, 192, 203],
                                 [255, 222, 173],
                                 [255, 255, 0],
                                 [144, 238, 144]], dtype=torch.float32)
    '''
    # p_argmax = torch.argmax(pred, dim=2).to('cpu')
    new = class_colors[torch.tensor(pred.copy(), dtype=torch.int32)]
    # new = torch.transpose(new, 0, 1).int().numpy().astype(np.uint8)
    cv2.imwrite('GraphMAE_out/out of seed{}.jpg'.format(seed), new.int().numpy().astype(np.uint8))
    
if __name__ == '__main__':
    T, GT = ReadData('Dataset', std=False)
    with open('SLIC_out/Graph/T_notstd.pkl', 'wb') as f:
        pickle.dump(T, f)
    with open('SLIC_out/Graph/GT_notstd.pkl', 'wb') as f:
        pickle.dump(GT, f)