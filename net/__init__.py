import net.resnet3d18
import net.Model_07


def resnet3d_18(in_channels=3, num_classes=20, layers=[2, 2, 2, 2]):
    return resnet3d18.Model(in_channels, layers, num_classes)


def gen_Model_07(*args, **kwargs):
    return Model_07.Model(*args, **kwargs)
