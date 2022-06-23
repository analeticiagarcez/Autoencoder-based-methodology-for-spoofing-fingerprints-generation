import numpy as np
import argparse
import os, random
from PIL import Image
from collections import defaultdict
import glob

import torch
import torch.utils.data as data
import torchvision.transforms as transforms


parser = argparse.ArgumentParser()

parser.add_argument('--gpu_ids', default='0', type=str)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--lr', default=0.0002, type=float)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--print_iter', default=20, type=int, help='print frequency')
parser.add_argument('--save_epoch', default=1, type=int)
parser.add_argument('--output_path', default='./results', type=str)


#parser.add_argument('--img_root',  default='C:\\Users\\ana_l\\Desktop\\TCC-Fingerprint\\2000\\DB3_B-opticalsensor', type=str)
#parser.add_argument('--img_root1',  default='C:\\Users\\ana_l\\Desktop\\TCC-Fingerprint\\2000\\DB3_B-opticalsensor\\test', type=str)



class Dataset(data.Dataset):
    def __init__(self, args, data):
        super(Dataset, self).__init__()
        
        if data == "train":
            self.img_root = args.img_root_train
        else: 
            self.img_root = args.img_root_test
        self.img_list = []
        
        for filename in os.listdir(self.img_root):
            label = filename.strip().split('_')[0]
            if label != "test" and label != "train":
                img_name = filename
                label = int(label)
                label = str(label - 101)
                self.img_list.append((img_name, label))
                

        self.transform = transforms.Compose([
            transforms.Resize((128,128)),
            #transforms.CenterCrop(128),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img_name, label = self.img_list[index]

        img_path = os.path.join(self.img_root, img_name)
        img = Image.open(img_path)
        img = self.transform(img)

        return {'img': img, 'label': int(label)}

    def __len__(self):
        return len(self.img_list)
    
