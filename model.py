import segmentation_models_pytorch as smp

def get_model(backbone='resnet34', num_classes=20, pretrained=True):
    model = smp.Unet(
        encoder_name = backbone,
        encoder_weights = 'imagenet' if pretrained else None,
        in_channels = 3,
        classes = num_classes,
    )
    return model
