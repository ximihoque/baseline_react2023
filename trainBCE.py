import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import math
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

class EmotionEncoder(nn.Module):
    def _init_(self, feature_dim=128):
        super(EmotionEncoder, self)._init_()
        self.conv1 = nn.Conv1d(in_channels=25, out_channels=feature_dim, kernel_size=3)

        self.pool = nn.AdaptiveMaxPool1d(750)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = x.permute(0, 2, 1) # N x features x seq
        out = self.pool(self.relu(self.conv1(x.permute(0, 2, 1)))) 
        out = out.permute(0, 2, 1)
        return out

class SiameseNetwork(nn.Module):
    def __init__(self, latent_dim=128, seq_len=750):
        super(SiameseNetwork, self).__init__()
        d_model = latent_dim

        self.encoder = EmotionEncoder()
        
        self.pool1 = nn.AdaptiveMaxPool1d(750)
        self.pool2 = nn.AdaptiveMaxPool1d(750)

        self.relu = nn.ReLU()
        
        self.mlp = nn.Sequential(
                    nn.Linear(d_model, d_model//2),
                    nn.ReLU(),
                    nn.Linear(d_model//2, d_model//4),
                    nn.ReLU(),
                    nn.Linear(d_model//4, 1),
                    nn.Sigmoid()
        )
        
    def forward(self, x1, x2):
        encSpk = self.encoder(x1)
        encList = self.encoder(x2)
        print(encSpk)
        print(encList)
        out = self.mlp(out)

class MyDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        label = self.labels[idx]
        
        x1, x2 = pair

        try:
            x1 = emotion_reader(x1).float()
        except:
            return self.__getitem__(idx + 1)
        
        try:
            x2 = emotion_reader(x2).float()
        except:
            return self.__getitem__(idx + 1)
        return (x1, x2), label

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
    parser.add_argument('--outdir', default="./results", type=str, help="result dir")
    parser.add_argument('--epochs', default=10, type=int, help="number of training epochs")
    
    args = parser.parse_args()
    return args


def train(args, model, loader, optimizer, criterion):
    losses = AverageMeter()
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for batch_idx, ((x1, x2), labels) in enumerate(tqdm(loader)):
        x1, x2, labels = x1.to(device), x2.to(device), labels.unsqueeze(1).float().to(device)
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            output = model(x1.cuda(), x2.cuda())
            loss_bce  = criterion(output, labels.unsqueeze(1))
            loss_bce.backward()
            optimizer.step()

        losses.update(loss_bce.item(), output.size(0))

    return losses.avg
    

def main(args):
    sim = SiameseNetwork()
    sim = sim.cuda()

    with open('../data/posTrain2.txt', 'r') as handle:
        pos_train = json.load(handle)

    with open('../data/negTrain2.txt', 'r') as handle:
        neg_train = json.load(handle)

    pairs_train = []
    labels_train = []


    for k1, k2 in zip(pos_train.keys(), neg_train.keys()):
        sample_pos = pos_train[k1]
        sample_neg = neg_train[k2]

        for p in sample_pos:
            pairs_train.append((k1, p))
            labels_train.append(0)

        for n in sample_neg:
            pairs_train.append((k2, n))
            labels_train.append(1)

    dataset = MyDataset(pairs_train, labels_train)

    start_epoch = 0

    if args.resume != '':
        checkpoint_path = args.resume
        print("Resume from {}".format(checkpoint_path))
        checkpoints = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        state_dict = checkpoints['state_dict']
        sim.load_state_dict(state_dict)

    # Create or import the model

    # Create the optimizer and loss criterion
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(sim.parameters(), lr = 0.0001)
    
    # Get the train loader
    loader_tr = DataLoader(dataset = dataset,
                    batch_size = 32,
                    shuffle = True,
                    num_workers = 4)

    output_dir = 'results/siamese2jul'
    checkpoint_path = os.path.join(output_dir, 'cur_checkpoint.pth')
    for epoch in range(start_epoch, args.epochs):
        loss_train = train(args, sim, loader_tr, optimizer, criterion)
        print(f"Epoch number {epoch}\n Current train loss {loss_train}")
        
        checkpoint = {
            'epoch': epoch+1,
            'state_dict': sim.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': loss_train
        }
        # if epoch % 10 == 0:
        #     loss_val = val(args, sim, loader_val, optimizer, criterion)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(checkpoint, os.path.join(output_dir, 'cur_checkpoint.pth'))

if __name__=="__main__":
    torch.multiprocessing.set_start_method('spawn')

    args = parse_arg()
    main(args)
