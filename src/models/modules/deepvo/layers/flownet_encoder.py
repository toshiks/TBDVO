import torch
import torch.nn as nn


def _conv_block(
        in_channels: int, out_channels: int, kernel_size: int = 3,
        stride: int = 1, dropout: float = 0, batch_norm: bool = True
) -> nn.Sequential:
    layers = [nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=(kernel_size - 1) // 2,
        bias=not batch_norm
    )]

    if batch_norm:
        layers.append(
            nn.BatchNorm2d(out_channels)
        )

    layers.append(
        nn.LeakyReLU(0.1)
    )
    layers.append(
        nn.Dropout(dropout)
    )

    return nn.Sequential(*layers)


class FlowNetEncoder(nn.Module):
    def __init__(self, in_channels: int, batch_norm: bool = True):
        super().__init__()

        self._batch_norm = batch_norm
        self.conv1 = _conv_block(in_channels, 64, kernel_size=7, stride=2, dropout=0.2, batch_norm=batch_norm)
        self.conv2 = _conv_block(64, 128, kernel_size=5, stride=2, dropout=0.2, batch_norm=batch_norm)

        self.conv3 = _conv_block(128, 256, kernel_size=5, stride=2, dropout=0.2, batch_norm=batch_norm)
        self.conv3_1 = _conv_block(256, 256, kernel_size=3, stride=1, dropout=0.2, batch_norm=batch_norm)

        self.conv4 = _conv_block(256, 512, kernel_size=3, stride=2, dropout=0.2, batch_norm=batch_norm)
        self.conv4_1 = _conv_block(512, 512, kernel_size=3, stride=1, dropout=0.2, batch_norm=batch_norm)

        self.conv5 = _conv_block(512, 512, kernel_size=3, stride=2, dropout=0.2, batch_norm=batch_norm)
        self.conv5_1 = _conv_block(512, 512, kernel_size=3, stride=1, dropout=0.2, batch_norm=batch_norm)
        self.conv6 = _conv_block(512, 1024, kernel_size=3, stride=2, dropout=0.5, batch_norm=batch_norm)

        self._valid_conv_names = [
            f"conv{i}" for i in range(1, 7)
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_conv1 = self.conv1(x)

        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)

        return out_conv6

    def load_pretrained(self, path: str):
        model_dict = torch.load(path, map_location='cpu')
        state_dict = {
            i: j for i, j in model_dict['state_dict'].items() if self._is_layer_valid(i)
        }

        self.load_state_dict(state_dict)

    def _is_layer_valid(self, state: str):
        if state.startswith("conv6_1"):
            return False

        for i in self._valid_conv_names:
            if state.startswith(i):
                return True

        return False
