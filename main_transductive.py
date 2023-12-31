from cmath import inf
import logging
import numpy as np
from tqdm import tqdm
import torch

from graphmae.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)
from graphmae.datasets.data_util import load_dataset, pixel_data
from graphmae.evaluation import node_classification_evaluation
from graphmae.models import build_model
from graphmae.datasets.dataset_process import draw


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def pretrain(model, graph, feat, pixel, patch, optimizer, max_epoch, s_branch_hid, p_branch_hid, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger=None):
    logging.info("start training..")
    graph = graph.to(device)
    x = feat.to(device)
    epoch_iter = tqdm(range(max_epoch))

    for epoch in epoch_iter:
        model.train()
        loss, loss_dict = model(graph, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)

        if (epoch + 1) % 200 == 0:
            node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, s_branch_hid, p_branch_hid, device, linear_prob, mute=True, pixel=pixel, patch=patch)
            model.encoder.clear_classifier() # evaluate之后模型就被更改了，因为输出是7类，所以要改回来

    # return best_model
    return model


def main(args):
    device = 'cuda:1'
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch # 200
    max_epoch_f = args.max_epoch_f # 30
    num_hidden = args.num_hidden # 嵌入属性向量长度
    num_layers = args.num_layers # 2
    encoder_type = args.encoder # gat
    decoder_type = args.decoder # gat
    replace_rate = args.replace_rate # 0

    optim_type = args.optimizer # adam
    loss_fn = args.loss_fn # sce

    lr = args.lr # 0.005
    weight_decay = args.weight_decay # 0.0005
    lr_f = args.lr_f # 0.001
    weight_decay_f = args.weight_decay_f # 0
    linear_prob = args.linear_prob # False
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging # False
    use_scheduler = args.scheduler # False
    s_branch_hid = args.s_branch_hid
    p_branch_hid = args.p_branch_hid
    train_rate = 0.002
    val_rate = 0.1
    test_rate = 0.2
    
    for i, seed in enumerate(seeds):
        graph, (num_features, num_classes), pixel, patch = load_dataset(dataset_name, 'Dataset', 'Dataset/LEE_T3/PauliRGB.bmp', k=38000, m=25, t=inf, s=1.2, train_rate=train_rate, val_rate=val_rate, test_rate=test_rate, device=device)
        pixel.feature = pixel.feature.to(device)
        pixel.GT = pixel.GT.to(device)
        pixel.label_superpixel = pixel.label_superpixel.to(device)
        # patch = patch.to(device)
        print(len(graph.ndata['train_mask']), graph.ndata['train_mask'].sum(), graph.ndata['val_mask'].sum(), graph.ndata['test_mask'].sum())
        print(pixel.train.numel(), pixel.train.sum(), pixel.val.sum(), pixel.test.sum())
        args.num_features = num_features

        acc_list = []
        estp_acc_list = []

        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None

        model = build_model(args)
        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
            # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                    # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None
            
        x = graph.ndata["feat"]
        if not load_model:
            model = pretrain(model, graph, x, pixel, patch, optimizer, max_epoch, s_branch_hid, p_branch_hid, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger)
            model = model.cpu()

        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load("checkpoint.pt"))
        if save_model:
            logging.info("Saveing Model ...")
            torch.save(model.state_dict(), "checkpoint.pt")
        
        model = model.to(device)
        model.eval()

        final_acc, estp_acc, pred = node_classification_evaluation(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f, s_branch_hid, p_branch_hid, device, linear_prob, pixel=pixel, patch=patch, final_out=True)
        acc_list.append(final_acc)
        estp_acc_list.append(estp_acc)

        if logger is not None:
            logger.finish()
        draw(pred, seed)
    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    estp_acc, estp_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    print(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")
    print(1)


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    print(args)
    main(args)
