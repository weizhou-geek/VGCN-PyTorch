import os
import time
import argparse
import torch
import math
import numpy as np
import cv2
import torch.optim as optim
import torch.optim.lr_scheduler as LS
from torch.autograd import Variable
from torchvision import models
import scipy.io as scio
from scipy import stats
import torch.nn as nn
from torchvision import models
import random

import utils
from datasets.cviqd_gl import get_dataset
from model.final_model import VGCN

os.environ['CUDA_VISIBLE_DEVICES'] = "7"
# Training settings
parser = argparse.ArgumentParser(description='VR Image Quality Assessment')
parser.add_argument('--start_epoch', type=int, default=1)
parser.add_argument('--total_epochs', type=int, default=20)
parser.add_argument('--total_iterations', type=int, default=10000)
parser.add_argument('--batch_size', '-b', type=int, default=12, help="Batch size")
parser.add_argument('--lr', type=float, default=1e-2, metavar=' LR', help='learning rate (default: 0.01)')
parser.add_argument('--number_workers', '-nw', '--num_workers', type=int, default=4)
parser.add_argument('--save', '-s', default='work', type=str, help='directory for saving')
parser.add_argument('--skip_training', default=False, action='store_true')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--root1', default='', type=str, metavar='PATH', help='path to pretrained local branch')
parser.add_argument('--root2', default='', type=str, metavar='PATH', help='path to pretrained global branch')

main_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(main_dir)

args = parser.parse_args()

# seed = [random.randint(0, 10000) for _ in range(4)]
seed = [7021, 9042, 9042, 8264]
torch.manual_seed(seed[0])
torch.cuda.manual_seed_all(seed[1])
np.random.seed(seed[2])
random.seed(seed[3])
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# print(seed)


kwargs = {'num_workers': args.number_workers}
if not args.skip_training:
    train_set = get_dataset(is_training=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, **kwargs)

test_set = get_dataset(is_training=False)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, **kwargs)

model = VGCN(root1=args.root1, root2=args.root2).cuda()


OIQA_params = list(map(id, model.OIQA_branch.parameters()))
DBCNN_params = list(map(id, model.DBCNN_branch.parameters()))
base_params = filter(lambda p: id(p) not in OIQA_params + DBCNN_params, model.parameters())
optimizer = optim.Adam([
    {'params': base_params},
    {'params': model.OIQA_branch.parameters(), 'lr': 1e-5},
    {'params': model.DBCNN_branch.parameters(), 'lr': 1e-5}], lr=args.lr)

# scheduler = LS.MultiStepLR(optimizer, milestones=[10, 30, 60], gamma=0.1)


def train(epoch, iteration):
    model.train()
    # scheduler.step()
    end = time.time()
    log = [0 for _ in range(1)]
    for batch_idx, batch in enumerate(train_loader):
        data, label, _, A, wimg = batch
        data = Variable(data.cuda())
        label = Variable(label.cuda())
        A = Variable(A.cuda())
        wimg = Variable(wimg.cuda())
        optimizer.zero_grad()
        _, _, batch_info = model(data, wimg, label, A, requires_loss=True)
        batch_info.backward()
        optimizer.step()
        # print(batch_info)

        log = [log[i] + batch_info.item() * len(data) for i in range(1)]
        iteration += 1

    log = [log[i] / len(train_loader.dataset) for i in range(1)]
    epoch_time = time.time() - end
    end = time.time()
    print('Train Epoch: {}, Loss: {:.6f}'.format(epoch, log[0]))
    print('LogTime: {:.4f}s'.format(epoch_time))
    return log


def eval():

    model.eval()
    log = 0
    score_list = []
    label_list = []
    name_list = []

    for batch_idx, batch in enumerate(test_loader):
        data, label, imgname, A, wimg = batch
        data = Variable(data.cuda())
        label = Variable(label.cuda())
        A = Variable(A.cuda())
        wimg = Variable(wimg.cuda())

        score, label = model(data, wimg, label, A, requires_loss=False)

        score = score.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        res = (score - label)*(score - label)
        score_list.append(score)
        label_list.append(label)
        name_list.append(imgname[0])

        ## release memory
        torch.cuda.empty_cache()

        log += res

    log = log / len(test_loader)

    print('Average LOSS: %.2f' % (log))
    score_list = np.reshape(np.asarray(score_list), (-1,))
    label_list = np.reshape(np.asarray(label_list), (-1,))
    name_list = np.reshape(np.asarray(name_list), (-1,))
    scio.savemat('cviqd_VGCN.mat', {'score': score_list, 'label': label_list, 'name': name_list})
    srocc = stats.spearmanr(label_list, score_list)[0]
    plcc = stats.pearsonr(label_list, score_list)[0]
    print('SROCC: %.4f, PLCC: %.4f\n' % (srocc, plcc))
    return srocc, plcc


if not args.skip_training:
    if args.resume:
        utils.load_model(model, args.resume)
        print('Train Load pre-trained model!')
    best = 0
    for epoch in range(args.start_epoch, args.total_epochs+1):
        iteration = (epoch-1) * len(train_loader) + 1
        log = train(epoch, iteration)
        log2 = eval()

        srocc = log2[0]
        plcc = log2[1]
        current_cc = srocc + plcc
        if current_cc > best:
            best = current_cc
            checkpoint = os.path.join(args.save, 'checkpoint')
            utils.save_model(model, checkpoint, epoch, is_epoch=True)

else:
    print('Test Load pre-trained model!')
    utils.load_model(model, args.resume)
    eval()
