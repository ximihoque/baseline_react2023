"""
Using only Audio features
"""

import torchaudio
import torch.nn as nn
bundle = torchaudio.pipelines.HUBERT_LARGE
model = bundle.get_model()
model = model.cuda()
model.train(False)

class HuBERTEncoder(nn.Module):
    def __init__(self, seq_len, feature_dim):
        super(HuBERTEncoder, self).__init__()

        self.conv1d = nn.Conv1d(in_channels=seq_len*2 - 1, out_channels=seq_len, kernel_size=3)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=seq_len, kernel_size=3)
        # emprical
        self.lin = nn.Linear(746, feature_dim)
    def forward(self, x):
        # x = B x F (audio features)
        out = model(x)[0]
        out = self.conv1d(out)
        # out = B x T x 1024
        # out = out.unsqueeze(1)
        # # out = B x C (1) x T x 1024
        # out = self.conv1(out)
        # out = self.pool(out)
        # out = self.conv2(out)
        # out = out.mean(dim=(-1))
        # out = self.lin(out)
        return out 
