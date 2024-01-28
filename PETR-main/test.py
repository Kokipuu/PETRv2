import torch
import torch.nn as nn
import torchvision.transforms as transforms

import model
from library.mmdet.datasets import build_dataloader
from library.mmdet3d.models import build_model

# set up the path
DATASET_PATH = './dataset/'
WEIGHT_PATH = './weight/  .pth'

# loading the datasets
dataset_type = 'CustomNuScenesDataset'
data_root = '/data/Dataset/nuScenes/'  # ToDo: update dataset later
input_modality = input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadMultiViewImageFromMultiSweepsFiles', sweeps_num=1, to_float32=True, pad_empty_sweeps=True, sweep_range=[3,27]),
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img'],
            meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'intrinsics', 'extrinsics',
                'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
                'img_norm_cfg', 'sample_idx', 'timestamp'))
        ])
]

# ToDo: update .pkl later
cfg_test = dict(type=dataset_type, pipeline=test_pipeline, ann_file=data_root + 'mmdet3d_nuscenes_30f_infos_val.pkl', classes=class_names, modality=input_modality)
# dataset = build_dataset(cfg_test)
dataset = dataset_type  # ToDo: program of build_dataset(mmdet3d(or mmdet)==builder.py, mmcv==resister.py, mmdet==builder.py)
data_loader = build_dataloader(
    dataset,
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=cfg.data.workers_per_gpu,
    dist=distributed,
    shuffle=False)



# loading model and weight
model = model.CNN()
model.load_state_dict(torch.load(weight_path))
model.eval()

# evaluate and output the result
with torch.no_grad():
    outputs = model(img)
    predict = torch.sigmoid(outputs)

print(predict)
