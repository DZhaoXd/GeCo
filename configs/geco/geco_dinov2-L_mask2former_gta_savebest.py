_base_ = "./geco_dinov2-L_mask2former_gta.py"

default_hooks = dict(
    checkpoint=dict(
        save_best="mean_mIoU",
        rule="greater",
    ),
)
