import random
import torch
from torchvision import transforms as TR
import os
from PIL import Image

class CityscapesDataset(torch.utils.data.Dataset):
    # 原图大小是1024 * 2048，所以load_size=512相当于读入低分辨率的原图像
    def __init__(self, opt, for_metrics):
        opt.load_size = 512
        opt.crop_size = 512
        opt.label_nc = 34
        opt.contain_dontcare_label = True
        opt.semantic_nc = 35 # label_nc + unknown
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 2.0

        self.opt = opt
        self.for_metrics = for_metrics
        self.images, self.labels, self.paths = self.list_images()

    def __len__(self,):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.paths[0], self.images[idx])).convert('RGB')
        label = Image.open(os.path.join(self.paths[1], self.labels[idx]))
        image, label = self.transforms(image, label)
        label = label * 255
        return {"image": image, "label": label, "name": self.images[idx]}

    def list_images(self):
        # for_metrics的意义暂不明确，可以推理的是它的作用与 opt.phase=='test' 有联系
        mode = "val" if self.opt.phase == "test" or self.for_metrics else "train"
        images = []
        path_img = os.path.join(self.opt.dataroot, "leftImg8bit", mode)
        for city_folder in sorted(os.listdir(path_img)):
            cur_folder = os.path.join(path_img, city_folder)
            for item in sorted(os.listdir(cur_folder)):
                images.append(os.path.join(city_folder, item))
        labels = []
        path_lab = os.path.join(self.opt.dataroot, "gtFine", mode)
        for city_folder in sorted(os.listdir(path_lab)):
            cur_folder = os.path.join(path_lab, city_folder)
            for item in sorted(os.listdir(cur_folder)):
                if item.find("labelIds") != -1:
                    labels.append(os.path.join(city_folder, item))
        assert len(images)  == len(labels), "different len of images and labels %s - %s" % (len(images), len(labels))
        for i in range(len(images)):
            assert images[i].replace("_leftImg8bit.png", "") == labels[i].replace("_gtFine_labelIds.png", ""),\
                '%s and %s are not matching' % (images[i], labels[i])
        return images, labels, (path_img, path_lab)

    def transforms(self, image, label):
        assert image.size == label.size
        # resize (load_size不等于原图的大小，crop_size是在load_size基础上进行裁剪的)
        new_width, new_height = (int(self.opt.load_size / self.opt.aspect_ratio), self.opt.load_size)
        image = TR.functional.resize(image, (new_width, new_height), Image.BICUBIC)
        label = TR.functional.resize(label, (new_width, new_height), Image.NEAREST)
        # flip（test阶段、for_metrics不翻转，显式指定no_flip也不翻转）
        if not (self.opt.phase == "test" or self.opt.no_flip or self.for_metrics):
            if random.random() < 0.5:
                image = TR.functional.hflip(image)
                label = TR.functional.hflip(label)
        # to tensor（像RGB mode下的PIL.Image对象，会转换为FloatTensor，同时会缩放到[0,1.0]，所以后续label会乘上255）
        image = TR.functional.to_tensor(image)
        label = TR.functional.to_tensor(label)
        # normalize（只有图片要规范化）
        image = TR.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return image, label
