_base_ = "./geco_dinov2-L_mask2former_gta.py"

test_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset={{_base_.val_cityscapes}},
)
val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset={{_base_.val_cityscapes}},
)

test_evaluator = dict(_delete_=True, type="ShapeAlignedIoUMetric", iou_metrics=["mIoU"])
val_evaluator = dict(_delete_=True, type="ShapeAlignedIoUMetric", iou_metrics=["mIoU"])
