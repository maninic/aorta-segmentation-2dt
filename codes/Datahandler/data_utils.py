import monai
from Options.train_options import TrainOptions
import json

options = TrainOptions()
info = options.gather_options()

def set_transforms():
    transforms = monai.transforms.Compose([
        monai.transforms.LoadImageD(keys=['magnitude','mask','velx','vely','velz']),
        monai.transforms.SqueezeDimd(keys=['magnitude','mask','velx','vely','velz'], dim=3),
        monai.transforms.SqueezeDimd(keys=['magnitude','mask','velx','vely','velz'], dim=2),
        monai.transforms.SqueezeDimd(keys=['magnitude','mask','velx','vely','velz'], dim=-1),
        monai.transforms.AddChanneld(keys=['magnitude','mask','velx','vely','velz']),
        monai.transforms.NormalizeIntensityD(keys=['magnitude']), 
        monai.transforms.NormalizeIntensityD(keys=['velx','vely','velz'], channel_wise=False),
        monai.transforms.RandRotated(keys=["magnitude","mask",'velx','vely','velz'], prob=1, range_z=[-1.57,1.57], mode=["bilinear", "nearest", "bilinear", "bilinear", "bilinear"]),
        monai.transforms.Resized(keys=["magnitude","mask",'velx','vely','velz'], spatial_size = [-1,-1,info.Tresample], size_mode='all', mode= "nearest", allow_missing_keys=False),
        monai.transforms.Spacingd(keys=["magnitude","mask",'velx','vely','velz'], pixdim = [info.sp_res[0], info.sp_res[1], -1], mode=  ["bilinear", "nearest", "bilinear", "bilinear", "bilinear"]),
        monai.transforms.SpatialPadd(
                    keys=["magnitude","mask",'velx','vely','velz'], spatial_size=[info.patch_size[0],info.patch_size[1],info.Tresample], allow_missing_keys=False
                ),
        monai.transforms.RandFlipd(keys=["magnitude","mask",'velx','vely','velz'], spatial_axis=[0,1], prob=0.5),  # Random flip along x y axis
        monai.transforms.RandWeightedCropd(keys=["magnitude","mask",'velx','vely','velz'], w_key="mask", spatial_size=[info.patch_size[0],info.patch_size[1],info.Tresample], num_samples=1),
        monai.transforms.ConcatItemsd(keys=["magnitude", "velx", "vely", "velz"], name="inputs"),
        monai.transforms.EnsureTyped(keys=['inputs','mask'])
    ])
    return transforms

def set_transforms_aug():
    transforms = monai.transforms.Compose([
        monai.transforms.LoadImageD(keys=['magnitude','mask','velx','vely','velz']),
        monai.transforms.RandFlipd(keys=["magnitude","mask",'velx','vely','velz'], spatial_axis=2, prob=0.5),  # Random flip along z-axis
        monai.transforms.SqueezeDimd(keys=['magnitude','mask','velx','vely','velz'], dim=3),
        monai.transforms.SqueezeDimd(keys=['magnitude','mask','velx','vely','velz'], dim=2),
        monai.transforms.SqueezeDimd(keys=['magnitude','mask','velx','vely','velz'], dim=-1),
        monai.transforms.AddChanneld(keys=['magnitude','mask','velx','vely','velz']),
        monai.transforms.NormalizeIntensityD(keys=['magnitude']), 
        monai.transforms.NormalizeIntensityD(keys=['velx','vely','velz'], channel_wise=False),
        monai.transforms.Resized(keys=["magnitude","mask",'velx','vely','velz'], spatial_size = [-1,-1,info.Tresample], size_mode='all', mode= "nearest", allow_missing_keys=False),
        monai.transforms.Spacingd(keys=["magnitude","mask",'velx','vely','velz'], pixdim = [info.sp_res[0], info.sp_res[1], -1], mode=  ["bilinear", "nearest", "bilinear", "bilinear", "bilinear"]),
        monai.transforms.SpatialPadd(
            keys=["magnitude","mask",'velx','vely','velz'], spatial_size=[info.patch_size[0],info.patch_size[1],-1], allow_missing_keys=False
        ),
        monai.transforms.RandWeightedCropd(keys=["magnitude","mask",'velx','vely','velz'], w_key="mask", spatial_size=[info.patch_size[0],info.patch_size[1],-1], num_samples=1),
        monai.transforms.ConcatItemsd(keys=["magnitude", "velx", "vely", "velz"], name="inputs"),
        monai.transforms.EnsureTyped(keys=['inputs','mask'])
    ])
    return transforms

def set_val_transf():
    val_transforms = monai.transforms.Compose([
        monai.transforms.LoadImageD(keys=['magnitude','mask','velx','vely','velz']),
        monai.transforms.SqueezeDimd(keys=['magnitude','mask','velx','vely','velz'], dim=3),
        monai.transforms.SqueezeDimd(keys=['magnitude','mask','velx','vely','velz'], dim=2),
        monai.transforms.SqueezeDimd(keys=['magnitude','mask','velx','vely','velz'], dim=-1),
        monai.transforms.AddChanneld(keys=['magnitude','mask','velx','vely','velz']),
        monai.transforms.NormalizeIntensityD(keys=['magnitude']), 
        monai.transforms.NormalizeIntensityD(keys=['velx','vely','velz'], channel_wise=False), 
        monai.transforms.Resized(keys=["magnitude","mask",'velx','vely','velz'], spatial_size = [-1,-1,info.Tresample], size_mode='all', mode= "nearest", allow_missing_keys=False),
        monai.transforms.Spacingd(keys=["magnitude","mask",'velx','vely','velz'], pixdim = [info.sp_res[0], info.sp_res[1], -1], mode= ["bilinear", "nearest", "bilinear", "bilinear", "bilinear"]),
        monai.transforms.SpatialPadd(
                keys=["magnitude","mask",'velx','vely','velz'], spatial_size=[info.patch_size[0],info.patch_size[1],info.Tresample], allow_missing_keys=False
                ),
        monai.transforms.CenterSpatialCropd(keys=["magnitude","mask",'velx','vely','velz'], roi_size=[info.patch_size[0],info.patch_size[1],info.Tresample]),
        monai.transforms.ConcatItemsd(keys=["magnitude", "velx", "vely", "velz"], name="inputs"),
        monai.transforms.EnsureTyped(keys=['inputs','mask'])
    ])
    return val_transforms

def data_from_dict(d, phase):
    #phase should be training validation or test
    #MAKE SURE
    #pointing the correct model folder
    #dict json files created
    #need the transformations
    #return dictionary, dataset and dataloader
    file = d / f'{phase}_dict.json'
    if phase == "training":
        t = set_transforms()
    else:
        t = set_val_transf()
    with open(file) as json_file:
        data = json.load(json_file)
    dataset = monai.data.Dataset(data = data, transform = t)
    dataloader = monai.data.DataLoader(dataset, batch_size=info.batch_size, num_workers=info.workers)
    return data, dataset, dataloader