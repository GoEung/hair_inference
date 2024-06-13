pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'
model = dict(
    type='HybridTaskCascade',
    backbone=dict(
        type='SwinTransformer',
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        # ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        # use_checkpoint=False
    ),
    # neck=dict(
    #     type='DyHead',
    #     in_channels=[128, 256, 512, 1024],
    #     out_channels=256,
    #     # num_outs=5
    # ),


    neck=[
            # dict(type='FPN',
            #     # in_channels=[256, 512, 1024, 2048],
            #     in_channels=[128, 256, 512, 1024], out_channels=256, num_outs=5),
            dict(type='HRFPN', in_channels=[128, 256, 512, 1024], out_channels=256),
            # dict(type='HRFPN_COM', in_channels=[128, 256, 512, 1024], out_channels=256),
            # dict(type='DyHead', in_channels=256, out_channels=256, num_blocks=6,
            #      zero_init_offset=False)
            # dict(type='HRFPN_CBAM',in_channels=[128, 256, 512, 1024], out_channels=256)
        ],


    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8, 12],
            ratios=[0.3, 0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='HybridTaskCascadeRoIHead',
        interleaved=True,
        mask_info_flow=True,
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ],
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=[
            dict(
                type='HTCMaskHead',
                with_conv_res=False,
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=1,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
            dict(
                type='HTCMaskHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=1,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
            dict(
                type='HTCMaskHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=1,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))
        ]),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict( 
        rpn=dict(   #bbox 설정
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='soft_nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(  #마스크 설정
            score_thr=0.001, #mask 수행하기 전 걸러지는 bbox confidence score 
            nms=dict(type='soft_nms', iou_threshold=0.5),
            max_per_img=1000,
            mask_thr_binary=0.5)))  #0.5 이상이면 1, 0.5 이하면 0으로 masking.



img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

classes = ("hair",)


albu_train_transforms = [
    dict(
        type='Rotate',
        limit=30,
        p=0.6),
    dict(
        type='RandomBrightnessContrast',
        p=0.5),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.2
    ),

    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=2010,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.15
    ),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    # dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    # dict(type='Resize', img_scale=(2700, 2000), keep_ratio=True),
    dict(type='Resize', img_scale=(1120,1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),

    #########################################
    #                   v2                  #
    #########################################
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True
        ),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes',
            'gt_masks': 'masks'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    #########################################

    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(512, 512),
        # img_scale=(2700, 2000),
        img_scale=(1120,1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(512, 512),
        # img_scale=(2700, 2000),
        img_scale=(1120,1024),
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

dataset_type = 'CocoDataset'
# data_root = './dataset/random_dataset_1/'
# data_root = './dataset/original/'
data_root = './dataset/slicing/'
# data_root_original = '../dataset/original/'
# data_root_gray = '../dataset_grayscale/'
# data_root_clahe = '/tmp/pycharm_project_189/dataset/slicing/'

# img => color/grayscale 바꾸는 위치 >> mmdetection-swin\mmdet\datasets\pipelines\loading.py 69th.
## the way to change code from loading.py >> img = mmcv.imfrombytes(img_bytes, flag=self.color_type)

data = dict(
    samples_per_gpu=1,      # GPU 카드 한장당 batch_size
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train/annotations/added_train_dataset.json',
        # ann_file=data_root + 'train/annotations/random_train_annotation.json',
        # ann_file=data_root + 'train/annotations/1-30_annotation.json',
        img_prefix=data_root + 'train/images/',
        # img_prefix=data_root_original + 'train/',
        # img_prefix=r'D:\PycharmProjects\mmdetection_226\dataset\slicing\train\images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val/annotations/instances_default.json',
        # ann_file=data_root + 'val/annotations/random_val_annotation.json',
        img_prefix=data_root + 'val/images/',
        # img_prefix=data_root_gray + 'val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # ann_file=data_root + 'test/annotations/new_modified_test_annotation.json',        # 데이터 25장
        ## only_swin     ==>  (mAP)   0.3:   , 0.5:
        ##               ==> (recall) 0.3:   ,
        ## only_dyhead   ==>  (mAP)   0.3: 0.8788498,   0.5: 0.8242467
        ##               ==> (recall) 0.3: 0.892833217,  0.5: 0.8456128
        ## only_hrfpn    ==>  (mAP)   0.3: 0.91554837 ,  0.5: 0.8799984
        ##               ==> (recall) 0.3: 0.941058,  0.5: 0.9065639
        ## hrfpn+dyhead  ==>  (mAP)   0.3: 0.9091498  , 0.5: 0.8648438
        ##               ==> (recall) 0.3: 0.9340254  , 0.5: 0.88847957

        ann_file=data_root + 'test/annotations/test_modified_annotation.json',      # 데이터 35장
        ## DetectoRS     ==>  (mAP)   0.3:    , 0.5:
        ##               ==> (recall) 0.3:   , 0.5:
        ## only_swin     ==>  (mAP)   0.3:     , 0.5:
        ##               ==> (recall) 0.3:     , 0.5:
        ## only_dyhead   ==>  (mAP)   0.3: 0.842801,   0.5: 0.7678307
        ##               ==> (recall) 0.3: 0.8600713,  0.5: 0.795231729
        ## only_hrfpn    ==>  (mAP)   0.3: 0.8946229,  0.5: 0.8341140
        ##               ==> (recall) 0.3: 0.92647058, 0.5: 0.86163101
        ## hrfpn+dyhead  ==>  (mAP)   0.3:  0.851317,  0.5: 0.7833702
        ##               ==> (recall) 0.3:  0.88948,   0.5: 0.8195187
        ## HRNET         ==>  (mAP)   0.3: 0.7518659  ,0.5: 0.7406233
        ##               ==> (recall) 0.3: 0.7542335  ,0.5: 0.7435383

        # ann_file=data_root + 'test/annotations/new_test_instances_default.json',
        # ann_file=data_root + 'test/annotations/instances_default.json',      # 데이터 35장 --> 0.758 mAP
        # ann_file=data_root + 'test/annotations/random_test_annotation.json',
        # # sample
        # ann_file=data_root + '/sample_22.json',
        img_prefix=data_root + 'test/images/',
        pipeline=test_pipeline))

evaluation = dict(metric=['bbox', 'segm'])


## this is original
# evaluation = dict(
#     # interval=1,
#     metric='bbox',)

# evaluation = dict(interval=1, metric='bbox', iou_thrs=[0.3])
# evaluation = dict(interval=1, metric=['bbox','segm'], iou_thrs=[0.5])


### changed
# evaluation = dict(type='CustomCocoEval', interval=1, metric='bbox', iou_thr=0.3)
# evaluation = dict(interval=1, metric='custom_evaluate')
# custom_iou_thr = 0.3




# Typically, a good starting point is to set the learning rate between 0.001 and 0.1.
# If the network is not learning well, the learning rate might be too high, and reducing it could help.
# If the training process is slow, the learning rate might be too low, and increasing it could speed up convergence.

# optimizer = dict(type='AdamW', lr=4e-5, betas=(0.9, 0.999), weight_decay=0.05)\
# optimizer = dict(type='AdamW', lr=0.00016 , betas=(0.9, 0.999), weight_decay=0.05)\
# optimizer = dict(type='AdamW', lr=0.005 , betas=(0.9, 0.999), weight_decay=0.05)\
optimizer = dict(type='AdamW', lr=0.00016 , betas=(0.9, 0.999), weight_decay=0.05)\
# Weight decay is a regularization technique that adds a penalty term to the loss function during training
# to encourage the model to use smaller weights. The penalty term is proportional to the L2 norm of the model's weights,
# and it helps to prevent overfitting by reducing the complexity of the model. In the mmdetection framework, weight decay can be applied by setting the weight_decay parameter in the optimizer.


                 # paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                 #                                 'relative_position_bias_table': dict(decay_mult=0.),
                 #                                 'norm': dict(decay_mult=0.)}))
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0/10,
    min_lr_ratio=4e-6
)

runner = dict(type='EpochBasedRunner', max_epochs=2000)




checkpoint_config = dict(interval=20)
# yapf:disable
log_config = dict(
    # interval=50,
    # interval=187,
    interval=197,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'