from dataset import ReactionDataset
from torch.utils.data import DataLoader
from model import TransformerVAE
import pdb
from tqdm import tqdm
from model import HuBERTEncoder

audio_flag = True
model = TransformerVAE(use_hubert=audio_flag,
audio_dim=128, # becomes irrelevant in case for hubert
max_seq_len=751, 
seq_len=750)
model = model.cuda()
dataset = ReactionDataset('../data', 'data/small.csv', clip_length=750, use_raw_audio=audio_flag)
shuffle = True 
# sample = dataset[0]
# hubert = HuBERTEncoder(1024)
# hubert = hubert.cuda()

train_loader = DataLoader(dataset=dataset, batch_size=2, shuffle=shuffle, num_workers=1)

# data_point = next(iter(train_loader))
print ("enumeratin")
for batch_idx, (speaker_video_clip, speaker_audio_clip, _, _, _, _, listener_emotion, listener_3dmm, listener_references) in enumerate(tqdm(train_loader)):

    print ('speaker video shape: ', speaker_video_clip.shape)
    print ('speaker audio shape: ', speaker_audio_clip.shape)
    print ("running inference...")
    # out = hubert(speaker_audio_clip.cuda())
    
# # pdb.set_trace();
    listener_3dmm_out, listener_emotion_out, distribution = model(speaker_video_clip.cuda(), speaker_audio_clip.cuda())
    print ('output shape: ', listener_3dmm_out.shape)
# print ("done.")
# print ('Batch idx: ', batch_idx)
# train_loader = get_dataloader(args, "data/train.csv", load_ref=False, load_video_l=False)