# -*- coding: utf-8 -*-
# @Author   : WenHan

from torch import nn


class AlexNet1d(nn.Module):
    def __init__(self):
        super(AlexNet1d, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=11, stride=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),

            nn.AdaptiveAvgPool1d(output_size=1)
        )

        self.classifier = nn.Linear(in_features=64, out_features=5)

    def forward(self, x):
        x = self.feature(x)
        x = x.view(-1, 64)
        logits = self.classifier(x)
        return logits


if __name__ == '__main__':
    import torch
    from torchsummary import summary
    model = AlexNet1d()
    summary(model, (8, 2048))
    my_x = torch.randn([1, 8, 2048])
    my_logits = model(my_x)
    print(my_logits)

