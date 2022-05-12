import random
import torch
from torchvision import transforms as TR
import os
import os.path as osp
from PIL import Image


# 专用于官方测试数据
class LandscapeTestDataset(torch.utils.data.Dataset):
    # 原图大小是768 * 1024
    # load_size=512相当于读入低分辨率的原图像；load_size=1024相当于读入正常分辨率的图像
    def __init__(self, opt, for_metrics):
        opt.load_size = 256
        opt.crop_size = 256
        # 默认29个类
        opt.label_nc = 29
        # TODO 先改为False，如果有需要再改回来
        opt.contain_dontcare_label = False
        # TODO contain_dontcare_label为False，semantic_nc就不用加一
        opt.semantic_nc = 29 # label_nc + unknown
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        # 宽高比：4:3
        opt.aspect_ratio = 1024 / 768

        self.opt = opt
        self.for_metrics = for_metrics
        self.images, self.labels, self.paths = self.list_images()

    def __len__(self,):
        return len(self.labels)

    def __getitem__(self, idx):
        # 用torch.zeros作为原图像
        image = torch.zeros(1)
        label = Image.open(os.path.join(self.paths[1], self.labels[idx])).convert('L')
        image, label = self.transforms(image, label)
        label = label * 255
        return {"image": image, "label": label, "name": self.labels[idx]}
        
    # 测试集没有原图，
    def list_images(self):
        # for_metrics的意义暂不明确，可以推理的是它的作用与 opt.phase=='test' 有联系
        mode = "test"
            
        labels = []
        path_lab = os.path.join(self.opt.dataroot, mode)
        for item in sorted(os.listdir(path_lab)):
            labels.append(item)
                
        return None, labels, (None, path_lab)

    def transforms(self, image, label):
        # resize (load_size不等于原图的大小，crop_size是在load_size基础上进行裁剪的)
        new_width, new_height = (int(self.opt.load_size / self.opt.aspect_ratio), self.opt.load_size)
        label = TR.functional.resize(label, (new_width, new_height), Image.NEAREST)
        # flip（test阶段、for_metrics不翻转，显式指定no_flip也不翻转）
        if not (self.opt.phase == "test" or self.opt.no_flip or self.for_metrics):
            if random.random() < 0.5:
                label = TR.functional.hflip(label)
        # to tensor（像RGB mode下的PIL.Image对象，会转换为FloatTensor，同时会缩放到[0,1.0]，所以后续label会乘上255；注意RGBA mode的Image对象，会直接转换为IntTensor或int32类型）
        label = TR.functional.to_tensor(label)
        return image, label

