from dataset import ReactionDataset
from torch.utils.data import DataLoader
from model import TransformerVAE
import pdb

model = TransformerVAE()
model = model.cuda()
dataset = ReactionDataset('../data/combined', 'data/train.csv')
shuffle = True 
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=shuffle, num_workers=4)
data_point = next(iter(train_loader))

speaker_video_clip, speaker_audio_clip, _, _, _, listener_emotion, listener_3dmm, _ = data_point
# pdb.set_trace();
listener_3dmm_out, listener_emotion_out, distribution = model(speaker_video_clip.cuda(), speaker_audio_clip.cuda())
print ("done.")
# print ('Batch idx: ', batch_idx)
# train_loader = get_dataloader(args, "data/train.csv", load_ref=False, load_video_l=False)