LEARNING_RATE = 0.0005
MK = "mk001a"
data_root = "/rds/user/ajg206/rds-meerkat-Y4bixlqNojM/pose-datasets/torso-3/"
ann_file = f"{data_root}/jsons/torso_annotations_{MK}.json"
val_file = f"{data_root}/jsons/torso_annotations_{MK}_bgr_validation.json"
img_prefix = f"{data_root}/archives/"

checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type="TextLoggerHook")])
log_level = "INFO"
load_from = None
resume_from = None
dist_params = dict(backend="nccl")
workflow = [("train", 1)]
opencv_num_threads = 0
mp_start_method = "fork"
dataset_info = dict(
    dataset_name="coco",
    paper_info=dict(
        author="Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Dollar, Piotr and Zitnick, C Lawrence",
        title="Microsoft coco: Common objects in context",
        container="European conference on computer vision",
        year="2014",
        homepage="http://cocodataset.org/",
    ),
    keypoint_info=dict(
        {
            0: dict(
                name="left_shoulder",
                id=5,
                color=[0, 255, 0],
                type="upper",
                swap="right_shoulder",
            ),
            1: dict(
                name="right_shoulder",
                id=6,
                color=[255, 128, 0],
                type="upper",
                swap="left_shoulder",
            ),
            2: dict(
                name="left_hip",
                id=11,
                color=[0, 255, 0],
                type="lower",
                swap="right_hip",
            ),
            3: dict(
                name="right_hip",
                id=12,
                color=[255, 128, 0],
                type="lower",
                swap="left_hip",
            ),
        }
    ),
    skeleton_info=dict(
        {
            0: dict(link=("left_hip", "right_hip"), id=4, color=[51, 153, 255]),
            1: dict(link=("left_shoulder", "left_hip"), id=5, color=[51, 153, 255]),
            2: dict(link=("right_shoulder", "right_hip"), id=6, color=[51, 153, 255]),
            3: dict(
                link=("left_shoulder", "right_shoulder"), id=7, color=[51, 153, 255]
            ),
        }
    ),
    joint_weights=[1.0, 1.0, 1.0, 1.0],
    sigmas=[0.079, 0.079, 0.107, 0.107],
)
evaluation = dict(interval=1, metric="mAP", save_best="AP")
optimizer = dict(type="Adam", lr=LEARNING_RATE)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[6, 9]
)
total_epochs = 12
target_type = "GaussianHeatmap"
channel_cfg = dict(
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[[5, 6, 11, 12]],
    inference_channel=[5, 6, 11, 12],
)
model = dict(
    type="TopDown",
    pretrained=None,
    # pretrained="/rds/user/ajg206/rds-meerkat-Y4bixlqNojM/pose-work/pretrained/td_torso_model.pth",
    backbone=dict(
        type="HRNet",
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block="BOTTLENECK",
                num_blocks=(4,),
                num_channels=(64,),
            ),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block="BASIC",
                num_blocks=(4, 4),
                num_channels=(32, 64),
            ),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block="BASIC",
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128),
            ),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block="BASIC",
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256),
            ),
        ),
    ),
    keypoint_head=dict(
        type="TopdownHeatmapSimpleHead",
        in_channels=32,
        out_channels=4,
        num_deconv_layers=0,
        extra=dict(final_conv_kernel=1),
        loss_keypoint=dict(type="JointsMSELoss", use_target_weight=True),
    ),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=False,
        post_process="default",
        shift_heatmap=False,
        target_type="GaussianHeatmap",
        modulate_kernel=17,
        use_udp=True,
    ),
)
data_cfg = dict(
    image_size=[224, 288],
    heatmap_size=[56, 72],
    num_output_channels=4,
    num_joints=4,
    dataset_channel=[[5, 6, 11, 12]],
    inference_channel=[5, 6, 11, 12],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    use_nms=False,
    bbox_file="/Users/alex/dev/topdown/jsons/example-json.json",
)
train_pipeline = [
    dict(type="LoadImageFromMeerkat"),
    dict(type="TopDownMakeBboxFullImage", padding=1.25),
    dict(type="TopDownRandomShiftBboxCenter", shift_factor=0.16, prob=0.3),
    dict(type="TopDownRandomFlip", flip_prob=0.5),
    dict(type="TopDownHalfBodyTransform", num_joints_half_body=4, prob_half_body=0.3),
    dict(type="TopDownGetRandomScaleRotation", rot_factor=40, scale_factor=0.5),
    dict(type="TopDownAffine", use_udp=True),
    dict(type="ToTensor"),
    dict(type="NormalizeTensor", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(
        type="TopDownGenerateTarget",
        sigma=3,
        encoding="UDP",
        target_type="GaussianHeatmap",
    ),
    dict(
        type="Collect",
        keys=["img", "target", "target_weight"],
        meta_keys=[
            "image_file",
            "joints_3d",
            "joints_3d_visible",
            "center",
            "scale",
            "rotation",
            "bbox_score",
            "flip_pairs",
        ],
    ),
]
val_pipeline = [
    dict(type="LoadImageFromMeerkat"),
    dict(type="TopDownMakeBboxFullImage", padding=1.25),
    dict(type="TopDownAffine", use_udp=True),
    dict(type="ToTensor"),
    dict(type="NormalizeTensor", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(
        type="Collect",
        keys=["img"],
        meta_keys=[
            "image_file",
            "center",
            "scale",
            "rotation",
            "bbox_score",
            "flip_pairs",
        ],
    ),
]
test_pipeline = [
    dict(type="LoadImageFromMeerkat"),
    dict(type="TopDownGetBboxCenterScale", padding=1.25),
    dict(type="TopDownMakeBboxFullImage", padding=1.25),
    dict(type="TopDownAffine", use_udp=True),
    dict(type="ToTensor"),
    dict(type="NormalizeTensor", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(
        type="Collect",
        keys=["img"],
        meta_keys=[
            "image_file",
            "center",
            "scale",
            "rotation",
            "bbox_score",
            "flip_pairs",
        ],
    ),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    val_dataloader=dict(samples_per_gpu=24),
    test_dataloader=dict(samples_per_gpu=24),
    train=dict(
        type="TopDownCocoDataset",
        ann_file=ann_file,
        img_prefix=img_prefix,
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info=dataset_info,
    ),
    val=dict(
        type="TopDownCocoDataset",
        ann_file=val_file,
        img_prefix=img_prefix,
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info=dataset_info,
    ),
    test=dict(
        type="TopDownCocoDataset",
        ann_file=val_file,
        img_prefix=img_prefix,
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info=dataset_info,
    ),
)
