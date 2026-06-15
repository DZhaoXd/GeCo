_base_ = "./geco_dinov2-L_mask2former.py"

model = dict(
    type="GeCoPEFTBackboneEncoderDecoder",
    enable_geco=True,
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
