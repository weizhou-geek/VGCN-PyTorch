import os
import six
import torch
import random
import numbers
import numpy as np
import cv2
import math
import scipy.io as scio

import torch.utils.data as data
from torchvision import transforms
import torchvision.transforms.functional as tF

import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def get_dataset(is_training):
    img_path_list = []
    img_name_list = []
    datasets_list = []
    sets_path = get_setspath(is_training)
    print(sets_path)
    labels_path = get_labelspath(is_training)
    wholeimg_path = get_imgpath(is_training)
    transform = get_transform()
    for set_path in sets_path:
        subset_names = os.listdir(set_path)
        for subset_name in subset_names:
            subset_path = os.path.join(set_path, subset_name)
            img_name_list.append(subset_name)
            img_path_list.append(subset_path)

    datasets_list.append(
        ImgDataset(
            img_path=img_path_list,
            img_name=img_name_list,
            transform=transform,
            is_training=is_training,
            label_path=labels_path[0],
            wholeimg_path=wholeimg_path[0]
        )
    )
    return data.ConcatDataset(datasets_list)


def get_setspath(is_training):
    sets_root = './database/'
    if is_training:
        sets = [
            'cviqd_resize_imgtrain'
        ]
    else:
        sets = [
            'cviqd_resize_imgtest'
        ]
    return [os.path.join(sets_root, set) for set in sets]

def get_imgpath(is_training):
    sets_root = './database/'
    if is_training:
        sets = [
            'cviqd_all_imgtrain'
        ]
    else:
        sets = [
            'cviqd_all_imgtest'
        ]
    return [os.path.join(sets_root, set) for set in sets]

def get_labelspath(is_training):
    sets_root = './database/'
    if is_training:
        sets = [
            'cviqd_fovall_label'
        ]
    else:
        sets = [
            'cviqd_fovall_label'
        ]
    return [os.path.join(sets_root, set) for set in sets]


def get_transform():
    return transforms.Compose([
        transforms.ToTensor()
    ])


class ImgDataset(data.Dataset):
    def __init__(self, img_path, img_name, transform, is_training, label_path, wholeimg_path, shuffle=False):
        self.img_path = img_path
        self.img_name = img_name
        self.nSamples = len(self.img_path)
        self.indices = range(self.nSamples)
        if shuffle:
            random.shuffle(self.indices)
        self.transform = transform
        self.is_training = is_training
        self.label_path = label_path
        self.wholeimg_path = wholeimg_path

    def __getitem__(self, index):
        imgpath = self.img_path[index]
        imagename = self.img_name[index]
        img_group = []
        sub_names = os.listdir(imgpath)
        for sub_name in sub_names:
            subimg_path = os.path.join(imgpath, sub_name)
            img = cv2.imread(subimg_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1))
            img_group.append(img)
        img_group = np.array(img_group)

        labelname = imagename + '.mat'
        labelname = os.path.join(self.label_path, labelname)
        label_content = scio.loadmat(labelname)
        label = label_content['score']
        label = label[0]

        wholeimgname = imagename + '.png'
        wimgname = os.path.join(self.wholeimg_path, wholeimgname)
        wimg = cv2.imread(wimgname)
        wimg = cv2.resize(wimg, (512, 1024), interpolation=cv2.INTER_CUBIC)
        wimg = cv2.cvtColor(wimg, cv2.COLOR_BGR2RGB)
        wimg = np.transpose(wimg, (2, 0, 1))
        A = label_content['A']

        data = torch.from_numpy(img_group).float()
        label = torch.from_numpy(label).float()
        wimg = torch.from_numpy(wimg).float()
        A = torch.from_numpy(A).float()

        return data, label, imagename, A, wimg

    def __len__(self):
        return self.nSamples

