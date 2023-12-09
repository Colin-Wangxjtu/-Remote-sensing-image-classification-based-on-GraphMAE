import numpy
import math
import copy
import torch
import torch.nn.functional as F
import dgl
from sklearn.preprocessing import StandardScaler
from graphmae.datasets.dataset_process import get_graph
import pickle


def preprocess(graph):
    feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat

    graph = graph.remove_self_loop().add_self_loop()
    graph.create_formats_()
    return graph


def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats

class pixel_data():
    def __init__(self, GT, graph, label, feature, device):
        train_mask = graph.ndata['train_mask']
        val_mask = graph.ndata['val_mask']
        test_mask = graph.ndata['test_mask']
        self.train = torch.full(label.shape, False)
        self.val = torch.full(label.shape, False)
        self.test = torch.full(label.shape, False)
        for i in range(len(train_mask)):  # 如果第i个超像素是某种掩码时，给其像素也打上对应掩码
            if train_mask[i]:
                self.train[label==i] = True
            if test_mask[i]:
                self.test[label==i] = True
            if val_mask[i]:
                self.val[label==i] = True
        self.train[GT==0] = self.val[GT==0] = self.test[GT==0] = False # 给原本没有标签的像素点去掉掩码
        self.label_superpixel = label
        self.feature = torch.tensor(feature.copy())
        self.GT = torch.tensor(GT.copy())
        self.use_mask = 1
        self.random_train = torch.full(label.shape, False)
        self.random_val = torch.full(label.shape, False)
        self.random_test = torch.full(label.shape, False)
        
class patch_data():
    def __init__(self, pixel, h, w):
        self.train = self.get_patch(pixel.feature, pixel.train, h, w)
        self.val = self.get_patch(pixel.feature, pixel.val, h, w)
        self.test = self.get_patch(pixel.feature, pixel.test, h, w)
        
    def get_patch(self, feature, train_mask, h, w):
        True_index = torch.nonzero(train_mask)
        pixel_data = torch.zeros((True_index.shape[0], h, w, feature.shape[2]))
        for i, index in enumerate(True_index):
            if index[0]-w//2 < 0:
                left = 0
                right = w
            elif index[0]+w//2 > feature.shape[0]:
                left = feature.shape[0] - w
                right = feature.shape[0]
            else:
                left = index[0] - w//2
                right = index[0] + w//2
                
            if index[1]-h//2 < 0:
                up = 0
                down = h
            elif index[1]+h//2 > feature.shape[1]:
                up = feature.shape[1] - h
                down = feature.shape[1]
            else:
                up = index[1] - h//2
                down = index[1] + h//2
            pixel_data[i] = feature[left:right, up:down]
        return pixel_data
    
def SlidingWindow(image, num_feature, stride, window_size, window_center=None, padding=True, print_show=True):
# 输入图像（C， W， H）， 输入标签（W， H）
# zero padding
    name = "data"
    a = torch.zeros((num_feature, window_size, window_size), dtype=torch.float32)

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

        image = F.pad(image, (W_padding_ahead, W_padding_behind, H_padding_ahead, H_padding_behind), value=0)
            # data zero padding(如：样本中心（6，6）时，前补6位，后补5位)
    position = []
    b = torch.zeros((width, height, num_feature, window_size, window_size), dtype=torch.float32)
    num = 0
    for i in range(0, width, stride):
        for j in range(0, height, stride):
            for m in range(window_size):
                for n in range(window_size):
                    a[:, m, n] = image[:, i + m, j + n]
            b[i, j] = a
            # 使用 append() 函数添加列表时，是添加列表的「引用地址」而不是添加列表内容，当被添加的列表发生变化时，添加后的列表也会同步发生变化;
            # 使用「深拷贝」添加列表的内容而不是引用地址，从而解决列表同步的问题
            num += 1
            if print_show:
                print("\rCollecting {} ...{:.2%} total:{}".format(name, num / total_num, num), end="")
    print(" Done! total:{}".format(num))
    with open('SLIC_out/Graph/pixel_notstd_patch.pkl', 'wb') as f:
        pickle.dump(b, f)
    return b, window_center, position
    # 得到截取样本list b，其中每个元素为滑动窗口截取的样本
    # position为返回的样本中心在整个image的位置

def random_select(pixel, train_rate, val_rate, test_rate):
    labels = pixel.GT.clone()
    indices = torch.nonzero(labels)
    label_data_num = len(indices)
    labels_num = len(labels.unique()) - 1
    one_label_train_num = int(label_data_num * train_rate // labels_num)
    train_indices = torch.zeros(((one_label_train_num*labels_num), 2), dtype=torch.int64)
    for l in range(labels_num): # 先获得每个标签的mask，然后随机选择一些作为训练集，并将其在labels中改为0，以免训练集与验证集中包含
        label_l_indices = torch.nonzero(labels == l+1)
        perm = torch.randperm(len(label_l_indices))
        train_indices[l*one_label_train_num:(l+1)*one_label_train_num] = label_l_indices[perm[:one_label_train_num]]
    labels[train_indices[:, 0], train_indices[:, 1]] = False

    indices = torch.nonzero(labels)
    label_data_num = len(indices)
    randperm = torch.randperm(label_data_num)
    val_indices = indices[randperm[:int(label_data_num*(val_rate))]]
    test_indices = indices[randperm[int(label_data_num*(val_rate)):int(label_data_num*(val_rate+test_rate))]]
    pixel.random_train[train_indices[:, 0], train_indices[:, 1]] = True
    pixel.random_val[val_indices[:, 0], val_indices[:, 1]] = True
    pixel.random_test[test_indices[:, 0], test_indices[:, 1]] = True
    

def patch_standard(patch):
    mean = patch.mean((-1, -2)).unsqueeze(-1).unsqueeze(-1).expand(patch.shape[0], patch.shape[1], patch.shape[2], patch.shape[-1], patch.shape[-1])
    std = torch.tensor(patch.numpy().std((-1, -2)), dtype=torch.float32).unsqueeze(-1).unsqueeze(-1).expand(patch.shape[0], patch.shape[1], patch.shape[2], patch.shape[-1], patch.shape[-1])
    patch = (patch - mean) / std
    patch = patch.transpose(0, 1)
    with open('SLIC_out/Graph/pixel_stdpatch.pkl', 'wb') as f:
        pickle.dump(patch, f)
    return patch
def load_dataset(dataset_name, path, RGB_path, k, m, t, s, train_rate, val_rate, test_rate, device):
    try:
        raise NotImplementedError
        with open('SLIC_out/Graph/graph_of_k={}_m={}_t={}_s={}_trainrate{}_val_rate{}.pkl'.format(k, m, t, s, train_rate, val_rate), 'rb') as f:
            graph = pickle.load(f)
        with open('SLIC_out/Graph/pixel_label_of_k={}_m={}_t={}_s={}_trainrate{}_val_rate{}.pkl'.format(k, m, t, s, train_rate, val_rate), 'rb') as f:
            label = pickle.load(f)
        with open('SLIC_out/Graph/pixel_feature_of_k={}_m={}_t={}_s={}_trainrate{}_val_rate{}.pkl'.format(k, m, t, s, train_rate, val_rate), 'rb') as f:
            feature = pickle.load(f)
        with open('SLIC_out/Graph/pixel_GT.pkl', 'rb') as f:
            GT = pickle.load(f)
    except:
        graph, label, feature,  GT = get_graph(path, RGB_path, k, m, t, s, train_rate, val_rate, device)
    graph = graph.remove_self_loop()
    graph = graph.add_self_loop()
    num_features = graph.ndata["feat"].shape[1]
    num_classes = 15
    pixel = pixel_data(GT, graph, label, feature, device)
    random_select(pixel, train_rate, val_rate, test_rate)
    # patch = patch_data(pixel, 12, 12)
    try:
        # raise NotImplementedError
        with open('SLIC_out/Graph/pixel_stdpatch.pkl', 'rb') as f:
            patch = pickle.load(f)
    except:
        patch, _, _ = SlidingWindow(pixel.feature.transpose(0, 2), num_features, 1, 12)
        patch = patch_standard(patch)
    return graph, (num_features, num_classes), pixel, patch
