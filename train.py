import os
import math
import argparse
from matplotlib.pyplot import axis

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader,random_split
import scipy.io as sio

from meet.models.vit import meet_small_patch8 as create_model
from utils import train_one_epoch, evaluate
from eegUtils import *

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights/SEED/" + args.task) is False:
        os.makedirs("./weights/SEED/" + args.task)
    if os.path.exists("./Results/SEED/") is False:
        os.makedirs("./Results/SEED/")

    tb_writer = SummaryWriter("./Summary/SEED/" + args.task + "/")

    # Load data
    for sbj in range(1,2):
        raw_data = sio.loadmat("./dataset/SEED/eeg_convert_to_image/S" + str(sbj) + "_32@32.mat")   # Change to your dataset path
        sbj_data = raw_data["img"]
        Label = (raw_data['label']).astype(int)
        if sbj == 1:
            All_data = sbj_data
        else:
            All_data = np.concatenate((All_data, sbj_data), axis=0)

    Label = np.tile(Label, (1,1))
    EEG_Images = np.transpose(All_data, (0,2,1,3,4))
    Label = np.reshape(Label, (-1,1))[:,0]

    EEG = EEGImagesDataset(label=Label, image=EEG_Images)
    lengths = [int(len(EEG)*0.8), int(len(EEG)*0.2)]
    if sum(lengths) != len(EEG):
            lengths[0] = lengths[0] + 1
    Train, Test = random_split(EEG, lengths)
    batch_size = args.batch_size
    Trainloader = DataLoader(Train, batch_size=batch_size,shuffle=True)
    Testloader = DataLoader(Test, batch_size=batch_size,shuffle=True)
    
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    model = create_model(num_classes=3).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # Delete unnecessary weights
        del_keys = ['head.weight', 'head.bias'] if False \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_acc = 0.
    acc_log = []
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=Trainloader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=model,
                                    data_loader=Testloader,
                                    device=device,
                                    epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        acc_log.append([train_loss, train_acc, val_loss, val_acc, optimizer.param_groups[0]["lr"]])

        # Save the weights
        if best_acc < val_acc:
            torch.save(model.state_dict(), "./weights/SEED/" + args.task + "/best_model.pth")
            best_acc = val_acc
        
        # Save the results
        result_file = os.path.join("./Results/SEED/", 'Result_%s.mat'%args.task)
        Acc_Log = np.array(acc_log)
        sio.savemat(result_file, {'Acclog': Acc_Log.astype(np.double)})

        print(best_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # Task type
    parser.add_argument('--task', default="SEED_S1")

    # The path of pretrained weight, set to null if you don't want to load it
    parser.add_argument('--weights', type=str, default='./weights/SEED/SEED_S1/best_model.pth',
                        help='initial weights path')

    # Freeze weight or not
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()

    main(opt)