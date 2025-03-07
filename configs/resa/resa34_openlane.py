net = dict(
    type='Detector',
)

backbone = dict(
    type='ResNetWrapper',
    resnet='resnet34',
    pretrained=True,
    replace_stride_with_dilation=[False, True, True],
    out_conv=True,
)
featuremap_out_channel = 128
featuremap_out_stride = 8
sample_y = range(1270, 490, -10)

aggregator = dict(
    type='RESA',
    direction=['d', 'u', 'r', 'l'],
    alpha=2.0,
    iter=4,
    conv_stride=9,
)

heads = dict( 
    type='LaneSeg',
    decoder=dict(type='BUSD'),
    thr=0.6,
    sample_y=sample_y,
)

trainer = dict(
    type='RESA'
)

evaluator = dict(
    type='CULane',        
)

optimizer = dict(
  type = 'SGD',
  lr = 0.030,
  weight_decay = 1e-4,
  momentum = 0.9
)

epochs = 15
batch_size = 8
total_iter = (157807 // batch_size) * epochs
import math
scheduler = dict(
    type = 'LambdaLR',
    lr_lambda = lambda _iter : math.pow(1 - _iter/total_iter, 0.9)
)

seg_loss_weight = 1.0
eval_ep = 2
save_ep = 2

bg_weight = 0.4

img_norm = dict(
    mean=[103.939, 116.779, 123.68],
    std=[1., 1., 1.]
)

img_height = 560
img_width = 840
cut_height = 500
ori_img_h = 1280
ori_img_w = 1920

train_process = [
    dict(type='RandomRotation'),
    dict(type='RandomHorizontalFlip'),
    dict(type='Resize', size=(img_width, img_height)),
    dict(type='Normalize', img_norm=img_norm),
    dict(type='ToTensor',),
]

val_process = [
    dict(type='Resize', size=(img_width, img_height)),
    dict(type='Normalize', img_norm=img_norm),
    dict(type='ToTensor', keys=['img']),
]

dataset_path = '/home/liang/Datasets/OpenLane/'
dataset = dict(
    train=dict(
        type='OpenLane',
        data_root=dataset_path,
        split='train',
        processes=train_process,
    ),
    val=dict(
        type='OpenLane',
        data_root=dataset_path,
        split='test',
        processes=val_process,
    ),
    test=dict(
        type='OpenLane',
        data_root=dataset_path,
        split='val',
        processes=val_process,
    )
)


workers = 12
num_classes = 14 + 1
ignore_label = 255
log_interval = 1000

lr_update_by_epoch = False
