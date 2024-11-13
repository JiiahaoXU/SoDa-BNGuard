from resnet import ResNet18, ResNet34


def get_model(data):

    if data == 'cifar10':
        model = ResNet18(num_classes=10)
    elif data == 'cifar100':
        model = ResNet34(num_classes=100)
    return model
