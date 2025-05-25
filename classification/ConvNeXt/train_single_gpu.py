import os
import math
import time
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
from model import ConvNeXt_T, ConvNeXt_S, ConvNeXt_B, ConvNeXt_L, ConvNeXt_XL
from dataset import MyDataSet
from multi_train_utils.train_eval_utils import train_one_epoch, evaluate

def main(args):
    device =torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:8080/')
    tb_writer = SummaryWriter()

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    train_data_set = MyDataSet(os.path.join(args.data_path, "train"), 'train')
    val_data_set = MyDataSet(os.path.join(args.data_path, "validation"), 'val')

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=batch_size,
                                               shuffle=True, pin_memory=True,
                                               num_workers=nw, collate_fn=train_data_set.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_data_set, batch_size=1,
                                            shuffle=False, pin_memory=True,
                                            num_workers=nw, collate_fn=val_data_set.collate_fn)

    # 如果存在预训练权重则载入
    model = ConvNeXt_S(num_classes=args.num_class).to(device)
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            # 比较参数数量，如果不一致则不载入
            # todo: 但应该还有更好的办法
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
        else:
            raise FileNotFoundError(f"weights file {args.weights} not found")

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后全连接层外，其他权重全部冻结
            if 'fc' not in name:
                para.requires_grad = False

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5e-3)
    # scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
        # train
        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        scheduler.step()

        # validate
        sum_num = evaluate(model=model,
                           data_loader=val_loader,
                           device=device)
        acc = sum_num / len(val_data_set)
        print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]['lr'], epoch)

        if epoch % 10 == 0 and epoch != 0:
            torch.save(model.state_dict(), "./weights/model-epoch{}.pth".format(epoch))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_class', type=int, default=11)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lrf', type=float, default=0.1)

    parser.add_argument('--data_path', type=str, default="/root/Desktop/course/datasets/food11")

    parser.add_argument('--weights', type=str, default="")
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    begin = time.time()
    main(opt)
    end = time.time()
    print("Training time: ", end - begin)