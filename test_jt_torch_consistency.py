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
                m.weight.data = weight.expand_as(m.weight.data).clone()
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias, 0.0)
        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            data = get_numpy_weight("xavier_normal", m.weight.shape[2], m.weight.shape[3])
            weight = torch.cuda.FloatTensor(data[0].astype(np.float32))
            # 必须要加上clone，具体原因请查看：https://discuss.pytorch.org/t/about-error-more-than-one-element-of-the-written-to-tensor-refers-to-a-single-memory-location/85526
            m.weight.data = weight.expand_as(m.weight.data).clone()
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
    losses_computer = losses.losses_computer(opt)

    netG = generators.OASIS_Generator(opt)
    netD = discriminators.OASIS_Discriminator(opt)
    netG.cuda()
    netD.cuda()
    init_networks([netG,netD])

    optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, opt.beta2))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, opt.beta2))

    for i, data_i in enumerate(dataloader):
        # forward
        image, label = models.preprocess_input(opt, data_i)
        fake = netG(label)
        print(fake.max().detach().cpu().data, fake.min().detach().cpu().data)
        output_D = netD(fake)
        print(output_D.max().detach().cpu().data, output_D.min().detach().cpu().data)
        # loss
        loss_G = 0
        loss_G_adv = losses_computer.loss(output_D, label, for_real=True)
        loss_G += loss_G_adv
        loss_G_vgg = None
        loss_G, losses_G_list = loss_G.mean(), [loss.mean() if loss is not None else None for loss in [loss_G_adv, loss_G_vgg]]
        print(losses_G_list)
        netG.zero_grad()
        loss_G.backward()
        optimizerG.step()
        print("stop here to see weight changes")

