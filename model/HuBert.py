"""
Using only Audio features
"""
import torchaudio
import torch.nn as nn
bundle = torchaudio.pipelines.HUBERT_LARGE
model = bundle.get_model()
model = model.cuda()

class HuBERTEncoder(nn.Module):
    def __init__(self, feature_dim=512):
        super(HuBERTEncoder, self).__init__()
        # self.model.train(mode=False)
        # self.model.model.encoder.requires_grad_ = True
        # self.model.model.feature_extractor.requires_grad_= True

        self.conv1 = nn.Conv1d(in_channels=1024, out_channels=feature_dim, kernel_size=3)
     
        self.pool = nn.AdaptiveMaxPool1d(1499)
        self.relu = nn.ReLU()

        
    def forward(self, x):
        # x = B x F (audio features)
        out = model(x)[0]
        out = out.permute(0, 2, 1) # N x features x seq

        out = self.relu(self.conv1(out))

        out = self.pool(out)
        out = out.permute(0, 2, 1) # N x seq x features
        return out 
