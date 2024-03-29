
import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os

#reconstruction_function = nn.MSELoss(size_average=False)
'''
reconstruction_function = nn.MSELoss(size_average=None, reduce=None, reduction='sum')

def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD
'''

def reparameterize(mu, logvar):
    std = logvar.mul(0.5).exp_()
    eps = torch.cuda.FloatTensor(std.size()).normal_()
    return eps.mul(std).add_(mu)

def kl_loss(mu, logvar, prior_mu=0):
    v_kl = mu.add(-prior_mu).pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    v_kl = v_kl.sum(dim=-1).mul_(-0.5)  # (batch, 2)
    return v_kl

def reconstruction_loss(prediction, target, size_average=False):
    error = (prediction - target).view(prediction.size(0), -1)
    error = error ** 2
    error = torch.sum(error, dim=-1)

    if size_average:
        error = error.mean()
    else:
        error = error.sum()
    return error




def ort_loss(x, y):
    loss = torch.abs((x * y).sum(dim=1)).sum()
    loss = loss / float(x.size(0))
    return loss


def ang_loss(x, y):
    loss = (x * y).sum(dim=1).sum()
    loss = loss / float(x.size(0))
    return loss


def rgb2gray(img):
    r, g, b = torch.split(img, 1, dim=1)
    return torch.mul(r, 0.299) + torch.mul(g, 0.587) + torch.mul(b, 0.114)


def save_checkpoint(model, epoch, name=""):
    if not os.path.exists("model/"):
        os.makedirs("model/")
    model_path = "model/" + name + "_epoch_{}.pth.tar".format(epoch)
    state = {"epoch": epoch, "state_dict": model.state_dict()}
    torch.save(state, model_path)
    print("checkpoint saved to {}".format(model_path))


def load_model(model, pretrained):
    weights = torch.load(pretrained)
    pretrained_dict = weights["state_dict"]
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


# assign adain_params to AdaIN layers
def assign_adain_params(adain_params, model):
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            mean = adain_params[:, :m.num_features]
            std = adain_params[:, m.num_features:2*m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2*m.num_features:
                adain_params = adain_params[:, 2*m.num_features:]


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def adjust_learning_rate(lr, step, optimizer, epoch):
    scale = 0.457305051927326
    lr = lr * (scale ** (epoch // step))
    print('lr: {}'.format(lr))
    if (epoch != 0) & (epoch % step == 0):
        print('Change lr')
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * scale


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




