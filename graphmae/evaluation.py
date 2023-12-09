import copy
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms

from graphmae.utils import create_optimizer, accuracy


def node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, s_channel, p_channel, device, linear_prob=True, mute=False, pixel=None, patch=None, final_out=False):
    model.eval()
    if linear_prob:
        with torch.no_grad():
            x = model.embed(graph.to(device), x.to(device))
            in_feat = x.shape[1]
        encoder = LogisticRegression(in_feat, num_classes)
    else:
        encoder = model.encoder
        encoder.reset_classifier(num_classes, 9, s_channel, p_channel)

    num_finetune_params = [p.numel() for p in encoder.parameters() if  p.requires_grad]
    if not mute:
        print(f"num parameters for finetuning: {sum(num_finetune_params)}")
    
    encoder.to(device)
    optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)
    final_acc, estp_acc, pred = linear_probing_for_transductive_node_classiifcation(encoder, graph, x, num_classes, optimizer_f, max_epoch_f, device, mute, pixel=pixel, patch=patch, final_out=final_out)
    return final_acc, estp_acc, pred


def linear_probing_for_transductive_node_classiifcation(model, graph, feat, num_classes, optimizer, max_epoch, device, mute=False, pixel=None, patch=None, final_out=False):
    criterion = torch.nn.CrossEntropyLoss()

    graph = graph.to(device)
    x = feat.to(device)
    if pixel:
        train_mask = pixel.train
        val_mask = pixel.val
        test_mask = pixel.test
        labels = pixel.GT.clone()
        labels -= 1 # 由于要用到的都是有标签的像素点，所以分为15类
    else:
        train_mask = graph.ndata["train_mask"]
        val_mask = graph.ndata["val_mask"]
        test_mask = graph.ndata["test_mask"]
        labels = graph.ndata["label"]
    
    all_random = True
    if all_random:
        train_mask = pixel.random_train
        val_mask = pixel.random_val
        test_mask = pixel.random_test
    best_val_acc = 0
    best_val_epoch = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()
        # pixel.use_mask = pixel.train
        pixel.use_mask = pixel.random_train # 使用与超像素分割单个像素的标签
        use_patch, use_labels = patch[pixel.use_mask.to('cpu')], labels[train_mask].long()
        # use_patch, use_labels = data_augmentation(patch[pixel.use_mask.to('cpu')].clone(), labels[train_mask].clone().long())
        use_patch = use_patch + torch.randn(use_patch.shape) * 0.3 # 加入一个噪声提高健壮性
        out = model(graph, x, pixel=pixel, patch=use_patch.to(device))
        loss = criterion(out, use_labels)
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
        optimizer.step()

        with torch.no_grad():
            model.eval()
            # pred = model(graph, x, pixel=pixel) # 当使用全像素卷积时
            # val_acc = accuracy(pred[val_mask], labels[val_mask])
            # val_loss = criterion(pred[val_mask], labels[val_mask].long())
            # test_acc = accuracy(pred[test_mask], labels[test_mask])
            # test_loss = criterion(pred[test_mask], labels[test_mask].long())
            
            # pixel.use_mask = pixel.val
            pixel.use_mask = pixel.random_val
            pred_val = model(graph, x, pixel=pixel, patch=patch[pixel.use_mask.to('cpu')].to(device))
            # pixel.use_mask = pixel.test
            pixel.use_mask = pixel.random_test
            pred_test = model(graph, x, pixel=pixel, patch=patch[pixel.use_mask.to('cpu')].to(device))
            val_acc, _ = accuracy(pred_val, labels[val_mask])
            val_loss = criterion(pred_val, labels[val_mask].long())
            test_OA, test_AA = accuracy(pred_test, labels[test_mask])
            test_loss = criterion(pred_test, labels[test_mask].long())
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = copy.deepcopy(model)

        if not mute:
            epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_acc:{val_acc: .4f}, test_loss:{test_loss.item(): .4f}, test_OA:{test_OA: .4f}, test_AA:{test_AA: .4f}")

    best_model.eval()
    with torch.no_grad():
        if final_out:
            height = pixel.test.shape[0] // 8
            pred = torch.zeros((labels.shape[0], labels.shape[1], num_classes)).to(device)
            for i in range(8):
                pixel.use_mask = torch.zeros(pixel.random_test.shape, dtype=bool).to(device)
                if i < 7:
                    j = (i+1)*height
                else:
                    j = pixel.test.shape[0]
                pixel.use_mask[i*height : j] = True
                out_i = best_model(graph, x, pixel=pixel, patch=patch[pixel.use_mask.to('cpu')].to(device))
                out_i = out_i.view(j-i*height, labels.shape[1], -1)
                pred[i*height: j] = out_i
            estp_test_OA, estp_test_AA = accuracy(pred[test_mask], labels[test_mask])
        else:
            pixel.use_mask = pixel.random_test
            pred = best_model(graph, x, pixel=pixel, patch=patch[pixel.use_mask.to('cpu')].to(device))
            estp_test_OA, estp_test_AA = accuracy(pred, labels[test_mask])
    if mute:
        print(f"# IGNORE: --- TestAcc: {test_OA:.4f}, early-stopping-TestOA: {estp_test_OA:.4f}, AA: {estp_test_AA:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")
    else:
        print(f"--- TestAcc: {test_OA:.4f}, early-stopping-TestOA: {estp_test_OA:.4f}, AA: {estp_test_AA:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")

    # (final_acc, es_acc, best_acc)
    return test_OA, estp_test_OA, pred


def linear_probing_for_inductive_node_classiifcation(model, x, labels, mask, optimizer, max_epoch, device, mute=False):
    if len(labels.shape) > 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    train_mask, val_mask, test_mask = mask

    best_val_acc = 0
    best_val_epoch = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)  

        best_val_acc = 0

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()
        out = model(None, x)
        loss = criterion(out[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
        optimizer.step()

        with torch.no_grad():
            model.eval()
            pred = model(None, x)
            val_acc = accuracy(pred[val_mask], labels[val_mask])
            val_loss = criterion(pred[val_mask], labels[val_mask])
            test_acc = accuracy(pred[test_mask], labels[test_mask])
            test_loss = criterion(pred[test_mask], labels[test_mask])
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = copy.deepcopy(model)

        if not mute:
            epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_acc:{val_acc}, test_loss:{test_loss.item(): .4f}, test_acc:{test_acc: .4f}")

    best_model.eval()
    with torch.no_grad():
        pred = best_model(None, x)
        estp_test_acc = accuracy(pred[test_mask], labels[test_mask])
    if mute:
        print(f"# IGNORE: --- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} ")
    else:
        print(f"--- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch}")

    return test_acc, estp_test_acc


class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)

    def forward(self, g, x, *args):
        logits = self.linear(x)
        return logits
    
def get_patch(feature, train_mask, h, w, num_class):
    True_index = torch.nonzero(train_mask)
    pixel_data = torch.zeros((True_index.shape[0], h, w, num_class))
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

def data_augmentation(data, label):
    column_flip = transforms.RandomVerticalFlip(p=1)
    row_flip = transforms.RandomHorizontalFlip(p=1)
    data_column_flip = column_flip(data)
    data_row_flip = row_flip(data)
    return torch.cat([data, data_column_flip, data_row_flip], dim=0), torch.cat([label, label, label], dim=0)