import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange

class patch_Conv(nn.Module):
    def __init__(self, in_c, out_c, h, w):
        super().__init__()
        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_c, 128, 3, padding=1, bias=False),
            nn.ReLU())
        self.Conv2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.AvgPool2d(2, 2),
            nn.ReLU())
        self.Conv3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU())
        self.Conv4 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.AvgPool2d(2, 2),
            nn.ReLU())
        self.Linear = nn.Linear(1024*h//4*w//4, 1024)
        self.final_out = nn.Linear(1024, out_c)
    
    def forward(self, x):
        batch_num = x.shape[0]
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x = self.Linear(x.view(batch_num, -1))
        return F.softmax(self.final_out(x), dim=1)
    
class pixel_data():
    def __init__(self, GT, graph, label, feature, device):
        train_mask = graph.ndata['train_mask']
        val_mask = graph.ndata['val_mask']
        test_mask = graph.ndata['test_mask']
        self.train = torch.full(label.shape, False).to(device)
        self.val = torch.full(label.shape, False).to(device)
        self.test = torch.full(label.shape, False).to(device)
        for i in range(len(train_mask)):  # 如果第i个超像素是某种掩码时，给其像素也打上对应掩码
            if train_mask[i]:
                self.train[label==i] = True
            if test_mask[i]:
                self.test[label==i] = True
            if val_mask[i]:
                self.val[label==i] = True
        self.train[GT==0] = self.val[GT==0] = self.test[GT==0] = False # 给原本没有标签的像素点去掉掩码
        self.label_superpixel = label
        self.feature = torch.tensor(feature)
        self.GT = torch.tensor(GT)
        self.use_mask = 1
        
def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct

class MyData(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

class RealNet(nn.Module):

    def __init__(self, class_num=15):
        super(RealNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(9, 9, 3, 1),
            nn.AvgPool2d((2, 2), 2),
            nn.Conv2d(9, 12, 3, 1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(108, class_num),
            nn.Softmax(dim=1)
        )

    def forward(self, input):
        x = self.model(input)
        # print(x.shape)
        x = self.classifier(x)
        return x

def patch_std(patch):
    mean = patch.mean(dim=(-1, -2))
    std = patch.std(dim=(-1, -2))
    mean_patch = mean.unsqueeze(-1).unsqueeze(-1).expand(patch.shape[0], patch.shape[1], patch.shape[2], patch.shape[3])
    std_patch = std.unsqueeze(-1).unsqueeze(-1).expand(patch.shape[0], patch.shape[1], patch.shape[2], patch.shape[3])
    return(patch - mean_patch) / std_patch

def add_noise(tensor, std):
    noise = torch.randn(tensor.shape) * std
    return noise + tensor

def plot_acc(e, acc):
    e_range = range(1, 1+e)
    plt.plot(e_range, acc, marker='o', linestyle='-')
    plt.title('Training Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()
    plt.savefig('test/patch_only_acc/plot.jpg')

def data_augmentation(data, label):
    column_flip = transforms.RandomVerticalFlip(p=1)
    row_flip = transforms.RandomHorizontalFlip(p=1)
    data_column_flip = column_flip(data)
    data_row_flip = row_flip(data)
    return torch.cat([data, data_column_flip, data_row_flip], dim=0), torch.cat([label, label, label], dim=0)
if __name__ == '__main__':
    seed = 2
    torch.manual_seed(seed)
    device = 'cuda:2'
    train_rate = 0.01
    test_rate = 0.1
    p_branch = patch_Conv(9, 15, 12, 12)
    realnet = RealNet(15)
    
    model = p_branch
    model.to(device)
    print(f'model has {sum(p.numel() for p in model.parameters())} parameters')
    
    with open('SLIC_out/Graph/patch_notstd_without0.pkl', 'rb') as f:
        notstd_patch = pickle.load(f)
        notstd_patch = torch.tensor(np.array(notstd_patch))
        
    with open('SLIC_out/Graph/pixel_stdpatch.pkl', 'rb') as f:
        patch = pickle.load(f)
    with open('SLIC_out/Graph/label_notstd_without0.pkl', 'rb') as f:
        GT = pickle.load(f)
        GT = torch.tensor(GT)
    randperm = torch.randperm(len(GT))
    GT -= 1
    
    train_mask = randperm[:int(train_rate*len(GT))]
    test_mask = randperm[int(train_rate*len(GT)):int(train_rate*len(GT)) + int(test_rate*len(GT))]

    # train_patch = patch[train_mask[:, 0], train_mask[:, 1], :, :, :]
    # test_patch = patch[test_mask[:, 0], test_mask[:, 1], :, :, :]
    
    # 重新标准化
    train_notstd = notstd_patch[train_mask]
    train_newstd = patch_std(train_notstd)
    train_label = GT[train_mask]
    train_newstd, train_label = data_augmentation(train_newstd, train_label)
    # sub_patch = torch.abs(train_newstd - train_patch)
    
    test_notstd = notstd_patch[test_mask]
    test_newstd = patch_std(test_notstd)
    test_label = GT[test_mask]
    
    train_set = MyData(train_newstd, train_label)
    train_Loader = DataLoader(train_set, batch_size=400, shuffle=True)

    test_set = MyData(test_newstd, test_label)
    test_Loader = DataLoader(test_set, batch_size=400, shuffle=True)
    
    epoch = 100
    cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    patch = patch
    acc_all = []
    acc_best = (0, 0)
    for e in range(epoch):
        model.train()
        for x, y in train_Loader:
            x = add_noise(x, 0.1)
            x, y = x.to(device), y.to(device).long()
            optimizer.zero_grad()
            out = model(x)
            loss = cost(out, y)
            loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
            optimizer.step()
        model.eval()
        test_acc = 0
        for x, y in test_Loader:
            x, y = x.to(device), y.to(device)
            pred_test = model(x)
            test_acc += accuracy(pred_test, y)
        test_acc /= len(test_set)
        print("\r{:.6f}\t {}".format(test_acc, e), end='')
        acc_all.append(test_acc)
        if test_acc > acc_best[0]:
            acc_best = test_acc, e
    plot_acc(epoch, acc_all)
    print('\n', acc_best)