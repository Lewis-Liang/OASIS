import numpy as np
import math
import torch
import torch.nn.init as init
import models.losses as losses
import models.models as models
import models.generator as generators
import models.discriminator as discriminators
import dataloaders.dataloaders as dataloaders
# import utils.utils as utils
# from utils.fid_scores import fid_jittor
import config


def get_numpy_weight(initialization, out_dim, in_dim):
    if initialization == 'zeros':
        return np.zeros([out_dim, in_dim]), np.zeros([out_dim])
    elif initialization == 'ones':
        return np.ones([out_dim, in_dim]), np.ones([out_dim])
    elif initialization == 'normal':
        return np.random.normal(loc=1., scale=0.02, size=[out_dim, in_dim]), \
                np.random.normal(loc=1., scale=0.02, size=[out_dim])
    elif initialization == 'xavier_Glorot_normal':
        return np.random.normal(loc=0., scale=1., size=[out_dim, in_dim]) / np.sqrt(in_dim), \
                np.random.normal(loc=0., scale=1., size=[out_dim]) / np.sqrt(in_dim)
    elif initialization == 'xavier_normal':
        matsize = out_dim * in_dim
        fan = (out_dim * matsize) + (in_dim * matsize)
        std = 0.02 * math.sqrt(2.0/fan)
        return np.random.normal(loc=0., scale=std, size=[out_dim, in_dim]), \
                np.random.normal(loc=0., scale=std, size=[out_dim])
    elif initialization == 'uniform':
        a = np.sqrt(1. / in_dim)
        return np.random.uniform(low=-a, high=a, size=[out_dim, in_dim]), \
                np.random.uniform(low=-a, high=a, size=[out_dim])
    elif initialization == 'xavier_uniform':
        a = np.sqrt(6. / (in_dim + out_dim))
        return np.random.uniform(low=-a, high=a, size=[out_dim, in_dim]), \
                np.random.uniform(low=-a, high=a, size=[out_dim])


def init_networks(networks):
    def init_weights(m, gain=0.02):
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                data = get_numpy_weight("normal", m.weight.shape[2], m.weight.shape[3])
                weight = torch.cuda.FloatTensor(data[0].astype(np.float32))
                m.weight.data = weight.expand_as(m.weight.data)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias, 0.0)
        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            data = get_numpy_weight("xavier_normal", m.weight.shape[2], m.weight.shape[3])
            weight = torch.cuda.FloatTensor(data[0].astype(np.float32))
            m.weight.data = weight.expand_as(m.weight.data)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias, 0.0)
    for net in networks:
        net.apply(init_weights)


if __name__ == "__main__":
    opt = config.read_arguments(train=True)
    print(np.random.randint(0,5, size=(2,2)))

    # jittor
    opt.dataroot = "./datasets/sample_images"

    opt.num_epochs = 2
    opt.batch_size = 1
    opt.freq_fid = 4
    opt.freq_print = 1
    opt.gpu_ids = "0"

    opt.norm_mod = True
    opt.no_3dnoise = True
    opt.no_labelmix = True
    # opt.no_spectral_norm = True


    dataloader, dataloader_val = dataloaders.get_dataloaders(opt)

    netG = generators.OASIS_Generator(opt)
    netD = discriminators.OASIS_Discriminator(opt)
    netG.cuda()
    netD.cuda()
    init_networks([netG,netD])

    for i, data_i in enumerate(dataloader):
        image, label = models.preprocess_input(opt, data_i)
        fake = netG(label)
        print(fake.max(), fake.min())

