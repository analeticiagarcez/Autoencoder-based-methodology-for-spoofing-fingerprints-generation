import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from utils import *
from model import define_G
from data1 import Dataset 


parser = argparse.ArgumentParser()

parser.add_argument('--gpu_ids', default='0', type=str)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--lr', default=0.0002, type=float)
parser.add_argument('--batch_size', default=8, type=int) #8
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--print_iter', default=20, type=int, help='print frequency')
parser.add_argument('--save_epoch', default=1, type=int)
parser.add_argument('--output_path', default='./results', type=str)
parser.add_argument('--output1_path', default='./test', type=str)


#parser.add_argument('--img_root',  default='C:\\Users\\ana_l\\Desktop\\TCC-Fingerprint\\2000\\DB3_B-opticalsensor', type=str)
#DB1_A train e DB1_B test (2004)
parser.add_argument('--img_root_train',  default='C:\\Users\\ana_l\\Desktop\\TCC-Fingerprint\\2004\\DB1_A', type=str)
parser.add_argument('--img_root_test',  default='C:\\Users\\ana_l\\Desktop\\TCC-Fingerprint\\2004\\DB1_B-opticalsensor', type=str)



def main():
    global args
    args = parser.parse_args()
    print(args)
    
    minloss = 100
    minepoch = 0
    
    graph_loss = []

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    cudnn.benchmark = True

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        
    if not os.path.exists(args.output1_path):
        os.makedirs(args.output1_path)

    # generator
    encoder, decoder = define_G(input_dim=1, output_dim=1, ndf=32)


    # dataset
    train_loader = torch.utils.data.DataLoader(
        Dataset(args, "train"), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        Dataset(args, "test"), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    # optimizer
    optimizer = optim.Adam(list(encoder.parameters()) +
                           list(decoder.parameters()), lr=args.lr, betas=(0.5, 0.999))

    # criterion
    criterionPix = torch.nn.L1Loss().cuda()
    criterionL2 = torch.nn.MSELoss().cuda()

    # train
    for epoch in range(args.epochs):

        # creat random index
        arange = torch.arange(args.batch_size).cuda()
        idx = torch.randperm(args.batch_size).cuda()
        while 0.0 in (idx - arange):
            idx = torch.randperm(args.batch_size).cuda()

        for iteration, data in enumerate(train_loader, start=1):
            # get data
            img = Variable(data["img"].cuda())
            batch_size = img.size(0)
            if batch_size < args.batch_size:
                continue

            noise = torch.zeros(batch_size, 256).normal_(0, 1).cuda()
            #id_noise = dec(noise)
            # forward
            z_img = encoder(img, "enc")

            style_img = encoder(z_img, "style")
            
            #mu, logvar = encoder(img, "enc")
            #z = reparameterize(mu, logvar)

            assign_adain_params(style_img, decoder)
            #rec_nir = decoder(torch.cat([id_vis, z_nir], dim=1), "img")
            #rec_nir_idx = decoder(torch.cat([id_vis[idx, :], z_nir], dim=1), "img")
            #fake_img = decoder(torch.cat([id_noise, z_img], dim=1), "img")
            fake_img = decoder(torch.cat([noise, z_img], dim=1))
            #fake_img = decoder(z_img)


            # orthogonal loss
            #loss_ort = 50 * (ort_loss(z_nir, id_vis) + ort_loss(z_vis, id_vis))

            # pixel loss
            loss_pix = 100 * (criterionPix(fake_img, img))
            
            #loss function
            #loss_func = loss_function(fake_img, img, mu, logvar)

            # all losses
            loss = loss_pix #+ loss_func

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            info = "====> Epoch[{}][{}/{}] | ".format(epoch, iteration, len(train_loader))
            #info += "Loss: pix: {:4.2f} ort: {:4.2f} | Ang-real rec: {:4.2f} pair: {:4.2f} | Ang-fake rec: {:4.2f} pair: {:4.2f}".format(
            #    loss_pix.item(), loss_ort.item(), real_ang_rec.item(), real_ang_pair.item(), fake_ang_rec.item(), fake_ang_pair.item())
            info += "Loss pix: {:4.2f}, Total Loss: {:4.2f}".format(loss_pix.item(), loss.item())
            print(info)
            #if epoch>800:
            vutils.save_image(torch.cat([img, fake_img], dim=0).data,
                              "{}/Epoch_{:03d}_Iter_{:06d}_img.tif".format(args.output_path, epoch, iteration), nrow=batch_size)
            
            # print log
            
            '''
            if iteration % args.print_iter == 0:
                info = "====> Epoch[{}][{}/{}] | ".format(epoch, iteration, len(train_loader))
                #info += "Loss: pix: {:4.2f} ort: {:4.2f} | Ang-real rec: {:4.2f} pair: {:4.2f} | Ang-fake rec: {:4.2f} pair: {:4.2f}".format(
                #    loss_pix.item(), loss_ort.item(), real_ang_rec.item(), real_ang_pair.item(), fake_ang_rec.item(), fake_ang_pair.item())
                info += "Loss pix: {:4.2f}, Total Loss: {:4.2f}".format(loss_pix.item(), loss.item())
                print(info)

            # save images
            if iteration % 500 == 0:
                vutils.save_image(torch.cat([img, fake_img], dim=0).data,
                                  "{}/Epoch_{:03d}_Iter_{:06d}_img.tif".format(args.output_path, epoch, iteration), nrow=batch_size)
            '''

            if loss.item() < minloss:
                minloss = loss.item()
                minepoch = epoch
                
            
        
        graph_loss.append(loss.item())           
        # save model
        if epoch % args.save_epoch == 0:
            save_checkpoint(encoder, epoch, "encoder")
            save_checkpoint(decoder, epoch, "decoder")
            
    with torch.no_grad():
        for iteration, data in enumerate(test_loader, start=1):
            
            img = Variable(data["img"].cuda())
            batch_size = img.size(0)
            if batch_size < args.batch_size:
                continue
            
            noise = torch.zeros(batch_size, 256).normal_(0, 1).cuda()
            z_img = encoder(img, "enc")
            
            fake_img = decoder(torch.cat([noise, z_img], dim=1))
            
            vutils.save_image(torch.cat([img, fake_img], dim=0).data,
                        "{}/Test_Iter_{:06d}_img.tif".format(args.output1_path, iteration), nrow=batch_size)
        
    print("Loss minimun {:4.2f} at {:03d}".format(minloss, minepoch))
    print(graph_loss)
    
    
    plt.figure()
    plt.plot(graph_loss)
    plt.title('Model Train Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend('train', loc='upper left')
    plt.savefig('trainloss.png')
    plt.show()
    
if __name__ == "__main__":
    main()
    
