import torch
import torch.nn as nn

# input_size = (Batch, channels, time, height, width)
class Model(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, feature_dim=32):
        super().__init__()
        self.bn = nn.BatchNorm3d(in_channels)
        self.sigmoid = nn.Sigmoid()

        in_channel = in_channels
        out_channel = 64
        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=(1, 5, 5), padding=(1, 2, 2)),
            nn.BatchNorm3d(out_channel),
            nn.ReLU()
        )

        in_channel = out_channel
        out_channel = 32
        self.layer2 = nn.Sequential(
            nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=(3, 1, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(out_channel),
            nn.ReLU()
        )

        # event_feature_layer
        in_channel = out_channel
        mid_channel = 32
        out1_channel = out_channels
        self.event_feature_layer = nn.Sequential(
            nn.Conv3d(in_channels=in_channel, out_channels=mid_channel, kernel_size=(1, 5, 5), padding=(1, 2, 2)),
            nn.BatchNorm3d(mid_channel),
            nn.ReLU(),
            nn.Conv3d(in_channels=mid_channel, out_channels=out1_channel, kernel_size=(3, 1, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(out1_channel),
        )

        # class_feature_layer
        in_channel = out_channel
        mid_channel = 32
        out2_channel = feature_dim
        self.class_feature_layer = nn.Sequential(
            nn.Conv3d(in_channels=in_channel, out_channels=mid_channel, kernel_size=(1, 5, 5), padding=(1, 2, 2)),
            nn.BatchNorm3d(mid_channel),
            nn.ReLU(),
            nn.Conv3d(in_channels=mid_channel, out_channels=out2_channel, kernel_size=(3, 1, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(out2_channel),
        )

        self.layer_norm = nn.LayerNorm(3 * 30)

    def forward(self, x):
        # x = batch, channel, time, height, width

        x = x.permute(0, 3, 4, 1, 2)  # x = batch, height, width, channel, time
        batch, height, width, channel, time = x.shape

        x = x.reshape(batch, height, width, channel * time)
        x = self.layer_norm(x)
        x = x.reshape(batch, height, width, channel, time)

        x = x.permute(0, 3, 4, 1, 2)  # x = batch, channel, time, height, width

        x = self.layer1(x)
        x = self.layer2(x)
        event_feature = self.event_feature_layer(x)
        event_feature = self.sigmoid(event_feature)

        cls_feature = self.class_feature_layer(x)
        return event_feature, cls_feature


def get_model(**kwargs):
    return Model(**kwargs)
