_base_ = [
    "../_base_/datasets/dg_gta_1024x1024.py",
    "../_base_/default_runtime.py",
    "../_base_/models/dinov2_mask2former.py",
]

crop_size = (1024, 1024)
rank = 16

model = dict(
    type="GeCoPEFTBackboneEncoderDecoder",
    enable_geco=True,
    backbone=dict(
        img_size=1024,
        init_cfg=dict(
            checkpoint="checkpoints/dinov2_geco_converted_1024x1024.pth",
        ),
        lora_cfg=dict(
            type="geco_adapter",
            r=rank,
            first_eigen=False,
            lora_alpha=rank,
            lora_dropout=0.0,
            lora_weight_init="geco_adapter",
            target_modules=["q", "k", "v", "proj", "fc1", "fc2"],
            start_lora_idx=8,
        ),
    ),
    data_preprocessor=dict(size=crop_size),
    test_cfg=dict(
        crop_size=(1024, 1024),
        stride=(683, 683),
    ),
    geco_regularizer=dict(
        type="GeCoRegularizer",
        alpha=0.1,
        beta=0.2,
        lambda_geo=0.1,
        num_neighbors=8,
        tangent_dim=4,
        curvature_threshold=None,
        curvature_quantile=0.5,
        max_tokens=1024,
        eps=1e-6,
        detach_clean=True,
        perturb_levels=(-1,),
        warmup_iters=1000,
        prototype_momentum=0.99,
        use_prototype_bank=False,
        num_classes=19,
        prototypes_per_class=4,
    ),
)

train_dataloader = dict(batch_size=4)

embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    constructor="PEFTOptimWrapperConstructor",
    optimizer=dict(
        type="AdamW", lr=0.0001, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999)
    ),
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.5, decay_mult=1.0),
            "query_embed": embed_multi,
            "query_feat": embed_multi,
            "level_embed": embed_multi,
            "norm": dict(decay_mult=0.0),
        },
        norm_decay_mult=0.0,
    ),
)
param_scheduler = [
    dict(type="PolyLR", eta_min=0, power=0.9, begin=0, end=40000, by_epoch=False),
    dict(
        type="CosineAnnealingParamScheduler",
        param_name="weight_decay",
        eta_min=0.00001,
        by_epoch=False,
        begin=0,
        end=40000,
    ),
]

train_cfg = dict(type="IterBasedTrainLoop", max_iters=40000, val_interval=4000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=4000,
        max_keep_ckpts=6,
        save_best="mean_mIoU",
        rule="greater",
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)
custom_hooks = [
    dict(type="WeightDecayLoggingHook"),
    dict(type="EMAHook", begin_iter=16000, priority="NORMAL"),
]

find_unused_parameters = True
