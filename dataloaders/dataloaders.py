import torch


def get_dataset_name(mode):
    if mode == "ade20k":
        return "Ade20kDataset"
    if mode == "cityscapes":
        return "CityscapesDataset"
    if mode == "coco":
        return "CocoStuffDataset"
    if mode == "landscape":
        return "LandscapeDataset"
    if mode == "landscapetest":
        return "LandscapeTestDataset"
    else:
        return ValueError("There is no such dataset regime as %s" % mode)


def get_dataloaders(opt):
    dataset_name   = get_dataset_name(opt.dataset_mode)
    file = __import__("dataloaders."+dataset_name)

    # test_only代表使用官方测试集
    if opt.phase=="test" and opt.test_only:
        dataset_test = file.__dict__[dataset_name].__dict__[dataset_name](opt, for_metrics=True)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size = opt.batch_size, shuffle = False, drop_last = False, num_workers=4)
        # 保持接口统一性。加上None后，在test.py中就不需要修改接收的返回值数量了
        return None, dataloader_test
    else:
        dataset_train = file.__dict__[dataset_name].__dict__[dataset_name](opt, for_metrics=False)
        dataset_val   = file.__dict__[dataset_name].__dict__[dataset_name](opt, for_metrics=True)
        print("Created %s, size train: %d, size val: %d" % (dataset_name, len(dataset_train), len(dataset_val)))

        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size = opt.batch_size, shuffle = True, drop_last=True, num_workers=4)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size = opt.batch_size, shuffle = False, drop_last=False, num_workers=4)

        return dataloader_train, dataloader_val

