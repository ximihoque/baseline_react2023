import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import argparse
from tqdm import tqdm
import json
import torchaudio
import warnings # Ignore warnings
warnings.filterwarnings("ignore")

# Emotions reader for loader
def emotion_reader(emt_path):
    df = pd.read_csv(emt_path)
    return torch.from_numpy(df.to_numpy())

def sample_audio(audio, fps, n_frames, src_sr, target_sr):
    """Explicitly for torch audio ops
    """
    audio = torchaudio.functional.resample(audio, src_sr, target_sr)
    sr = target_sr
    audio = audio.mean(0)
    frame_n_samples = int(sr / fps)
    curr_length = len(audio)
    target_length = frame_n_samples * n_frames
    if curr_length > target_length:
        audio = audio[:target_length]
    elif curr_length < target_length:
        audio = np.pad(audio, [0, target_length - curr_length])
    return audio, frame_n_samples


class MyDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        label = self.labels[idx]
        
        x1, x2, x3 = pair
        x1 = torch.load(x1).float()
        
        try:
            speaker_audio_clip, sr = torchaudio.load(x2)
        except Exception as err:
            return self.__getitem__(idx+1)
        speaker_audio_clip = torchaudio.functional.resample(speaker_audio_clip, sr, 16000)
        speaker_audio_clip, frame_n_samples = sample_audio(speaker_audio_clip, fps=25, src_sr=sr,
                                                           target_sr=16000, n_frames=750)
        
        try:
            x3 = emotion_reader(x3).float()
        except:
            return self.__getitem__(idx + 1)
        return ((x1, speaker_audio_clip), x3), label

bundle = torchaudio.pipelines.HUBERT_LARGE
model = bundle.get_model()
model = model.cuda()

class HuBERTEncoder(nn.Module):
    def __init__(self, feature_dim=512):
        super(HuBERTEncoder, self).__init__()

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

class VideoEncoder(nn.Module):
    def __init__(self, feature_dim=512):
        super(VideoEncoder, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1024, out_channels=feature_dim, kernel_size=3)
        self.pool = nn.AdaptiveMaxPool1d(23)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = x.permute(0, 2, 1) # N x features x seq
        out = self.relu(self.conv1(out))
        out = self.pool(out)
        out = out.permute(0, 2, 1) # N x seq x features
        
        return out

class SpeakerBehaviourEncoder(nn.Module):
    def __init__(self):
        super(SpeakerBehaviourEncoder, self).__init__()

        self.video_encoder = VideoEncoder(feature_dim=512)
        print ("Using HuBERT.")

        self.audio_encoder = HuBERTEncoder(feature_dim=512)
        self.fusion_layer = nn.Linear(512, 256)

        transformer_layer = nn.TransformerEncoderLayer(d_model=256, 
                                                    nhead=4,
                                                    dim_feedforward=512,
                                                    dropout=0.3)
        self.encoder = nn.TransformerEncoder(transformer_layer, num_layers=1)

        self.lin = nn.Linear(256, 128)

    def forward(self, video, audio):
        video_feature = self.video_encoder(video)
        audio_feature = self.audio_encoder(audio)
     
        fused = self.fusion_layer(torch.cat((video_feature, audio_feature), dim=1))
        encoded_feature = self.lin(self.encoder(fused))

        return encoded_feature

class SiameseNetwork(nn.Module):
    def __init__(self, latent_dim=128, seq_len=750):
        super(SiameseNetwork, self).__init__()
     
        self.spk_enc = SpeakerBehaviourEncoder()
        self.conv2 = nn.Conv1d(in_channels=25, out_channels=latent_dim, kernel_size=3)
        
        self.pool1 = nn.AdaptiveMaxPool1d(750)
        self.pool2 = nn.AdaptiveMaxPool1d(750)
        self.relu = nn.ReLU()
       
        
    def forward(self, x1, x2):
        x1_vid, x1_aud = x1
        x1 = self.pool1(self.spk_enc(x1_vid, x1_aud).permute(0, 2, 1))
        x2 = self.pool2(self.relu(self.conv2(x2.permute(0, 2, 1)))) 
        x2 = x2.permute(0, 2, 1)
        x1 = x1.permute(0, 2, 1)

        return x1, x2

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        
        self.margin = margin

    def forward(self, output1, output2, label):
        # Calculate the Euclidean distance between the outputs of Siamese network
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)

        # Calculate the contrastive loss
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                        (label) * torch.pow(torch.clamp(self.margin - 
                                        euclidean_distance, min=0.0), 2))

        return loss_contrastive

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum/ self.count

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', default="../data", type=str, help="dataset path")
    parser.add_argument('--resume', default="", type=str, help="checkpoint path")
    parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--outdir', default="./siamese30jun", type=str, help="result dir")
    parser.add_argument('--epochs', default=10, type=int, help="number of training epochs")
    
    args = parser.parse_args()
    return args


def train(args, model, loader, optimizer, criterion):
    losses = AverageMeter()
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for batch_idx, (((x1_vid, x1_aud), x2), labels) in enumerate(tqdm(loader)):
        x1_vid, x1_aud, x2, labels = x1_vid.to(device), x1_aud.to(device), x2.to(device), labels.unsqueeze(1).float().to(device)
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            output1, output2 = model((x1_vid.cuda(), x1_aud.cuda()), x2.cuda())
            loss_contrastive  = criterion(output1, output2, labels.unsqueeze(1))
            loss_contrastive.backward()
            optimizer.step()

        losses.update(loss_contrastive.item(), output1.size(0))

    return losses.avg

def val(args, model, val_loader, criterion, render, epoch):
    losses = AverageMeter()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for batch_idx, (((x1_vid, x1_aud), x2), labels) in enumerate(tqdm(loader)):
        x1_vid, x1_aud, x2, labels = x1_vid.to(device), x1_aud.to(device), x2.to(device), labels.unsqueeze(1).float().to(device)
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            output1, output2 = model((x1_vid.cuda(), x1_aud.cuda()), x2.cuda())
            loss_contrastive  = criterion(output1, output2, labels.unsqueeze(1))

        losses.update(loss_contrastive.item(), output1.size(0))

    return losses.avg
    

def main(args):
    with open('../data/posTrain1.txt', 'r') as handle:
        pos_train = json.load(handle)

    with open('../data/negTrain1.txt', 'r') as handle:
        neg_train = json.load(handle)

    with open('../data/posVal1.txt', 'r') as handle:
        pos_val = json.load(handle)

    with open('../data/negVal1.txt', 'r') as handle:
        neg_val = json.load(handle)

    pairs_train = []
    pairs_val = []
    labels_train = []
    labels_val = []

    pos_keys = list(pos_train.keys())
    neg_keys = list(neg_train.keys())

    pos_keys1 = list(pos_val.keys())
    neg_keys1 = list(neg_val.keys())

    random.shuffle(pos_keys)
    random.shuffle(neg_keys)

    random.shuffle(pos_keys1)
    random.shuffle(neg_keys1)

    num_samples = 2194
    for k1, k2 in zip(pos_keys, neg_keys):
        sample_pos = ("../data/Emotions/" + pos_train[k1][0] + ".csv").split()
        sample_neg = ("../data/Emotions/" + neg_train[k2][0] + ".csv").split()

        for p in sample_pos:
            k1_p = "../data/Videos/" + k1 + "_marlin.pt"
            k1_a = "../data/Audios/" + k1 + ".wav"
            pairs_train.append((k1_p, k1_a, p))
            labels_train.append(0)

        for n in sample_neg:
            k2_p = "../data/Videos/" + k2 + "_marlin.pt"
            k2_a = "../data/Audios/" + k2 + ".wav"
            pairs_train.append((k2_p, k2_a, n))
            labels_train.append(1)

    for k1, k2 in zip(pos_keys1, neg_keys1):
        sample_pos = ("../data/Emotions/" + pos_val[k1][0] + ".csv").split()
        sample_neg = ("../data/Emotions/" + neg_val[k2][0] + ".csv").split()

        for p in sample_pos:
            k1_p = "../data/Videos/" + k1 + "_marlin.pt"
            k1_a = "../data/Audios/" + k1 + ".wav"
            pairs_val.append((k1_p, k1_a, p))
            labels_val.append(0)

        for n in sample_neg:
            k2_p = "../data/Videos/" + k2 + "_marlin.pt"
            k2_a = "../data/Audios/" + k2 + ".wav"
            pairs_val.append((k2_p, k2_a, n))
            labels_val.append(1)
    
    dataset = MyDataset(pairs_train, labels_train)
    valD = MyDataset(pairs_val, labels_val)

    start_epoch = 0
    sim = SiameseNetwork()
    if args.resume != '':
        checkpoint_path = args.resume
        print("Resume from {}".format(checkpoint_path))
        checkpoints = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        state_dict = checkpoints['state_dict']
        sim.load_state_dict(state_dict)

    # Create or import the model
    
    sim = sim.cuda()
    # Create the optimizer and loss criterion
    criterion = ContrastiveLoss()
    optimizer = optim.AdamW(sim.parameters(), lr = 0.0001)
    
    # Get the train loader
    loader_tr = DataLoader(dataset = dataset,
                    batch_size = 5,
                    shuffle = True,
                    num_workers = 5)

    loader_val = DataLoader(dataset = valD,
                    batch_size = 5,
                    shuffle = True,
                    num_workers = 5)

    # output_dir = 'results/siamese30jun'
    output_dir = args.outdir
    checkpoint_path = os.path.join(output_dir, 'cur_checkpoint.pth')
    for epoch in range(start_epoch, args.epochs):
        loss_train = train(args, sim, loader_tr, optimizer, criterion)
        if epoch % 10 == 0:
            loss_val = val(args, sim, loader_val, optimizer, criterion)
            print(f"Current val loss {loss_val}")
        print(f"Epoch number {epoch}\n Current train loss {loss_train}")
        
        checkpoint = {
            'epoch': epoch+1,
            'state_dict': sim.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': loss_train
        }


        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(checkpoint, os.path.join(output_dir, 'cur_checkpoint.pth'))

if __name__=="__main__":
    torch.multiprocessing.set_start_method('spawn')

    args = parse_arg()
    main(args)
