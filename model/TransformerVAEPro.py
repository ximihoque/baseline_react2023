import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .BasicBlock import ConvBlock, PositionalEncoding, lengths_to_mask, init_biased_mask
from .HuBert import HuBERTEncoder

class VAEModel(nn.Module):
    def __init__(self,
                 latent_dim: int = 256,
                 device='cuda',
                 **kwargs) -> None:
        super(VAEModel, self).__init__()

        self.latent_dim = latent_dim
        self.device = device 
        seq_trans_encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                             nhead=4,
                                                             dim_feedforward=latent_dim * 2,
                                                             dropout=0.1)

        self.seqTransEncoder = nn.TransformerEncoder(seq_trans_encoder_layer, num_layers=1)
        self.mu_token = nn.Parameter(torch.randn(latent_dim))
        self.logvar_token = nn.Parameter(torch.randn(latent_dim))


    def forward(self, x):
        B, T, D = x.shape
        lengths = [len(item) for item in x]

        mu_token = torch.tile(self.mu_token, (B,)).reshape(B, 1, -1)
        logvar_token = torch.tile(self.logvar_token, (B,)).reshape(B, 1, -1)

        x = torch.cat([mu_token, logvar_token, x], dim=1)

        x = x.permute(1, 0, 2)

        token_mask = torch.ones((B, 2), dtype=bool, device=self.device)
        mask = lengths_to_mask(lengths, device=self.device)
        aug_mask = torch.cat((token_mask, mask), 1)

        x = self.seqTransEncoder(x, src_key_padding_mask=~aug_mask)

        mu = x[0]
        logvar = x[1]
        std = logvar.exp().pow(0.5)
        # print ('mu', mu.shape)
        # print ('logvar', logvar.shape)
        dist = torch.distributions.Normal(mu, std)
        # print ('dist', dist)
        motion_sample = self.sample_from_distribution(dist).to(self.device)

        return motion_sample, dist

    def sample_from_distribution(self, distribution):
         return distribution.rsample()




class Decoder(nn.Module):
    def __init__(self,  output_3dmm_dim = 58, output_emotion_dim = 25, feature_dim = 128, device = 'cpu', 
                        max_seq_len=751, n_head = 4, window_size = 8, online = False):
        super(Decoder, self).__init__()

        self.feature_dim = feature_dim
        self.window_size = window_size
        self.device = device
        
        self.vae_model = VAEModel(feature_dim, device=device)

        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=n_head, dim_feedforward=2*feature_dim, batch_first=True)
        self.listener_reaction_decoder_1 = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.listener_reaction_decoder_2 = nn.TransformerDecoder(decoder_layer, num_layers=1)


        self.biased_mask = init_biased_mask(n_head = n_head, max_seq_len = max_seq_len, period=max_seq_len)

        self.listener_reaction_3dmm_map_layer = nn.Linear(feature_dim, output_3dmm_dim)

        #TODO: create two emotion map layers, 1 - AU (Sigmoid activation), 2 - Emotions 
        self.listener_reaction_emotion_map_layer = nn.Sequential(
            nn.Linear(feature_dim + output_3dmm_dim, feature_dim),
            nn.Linear(feature_dim, output_emotion_dim)
        )
        self.listener_emotion_AU = nn.Sequential(
            nn.Linear(output_emotion_dim, 15),
            nn.ReLU(),
            nn.Sigmoid()
        )
        self.listener_emotion_emotion = nn.Linear(output_emotion_dim, 10)

        self.PE = PositionalEncoding(feature_dim)


    def forward(self, encoded_feature, past_reaction_3dmm = None, past_reaction_emotion = None):
        B = encoded_feature.shape[0]
        TL = 750
        motion_sample, dist = self.vae_model(encoded_feature)
     
        time_queries = torch.zeros(B, TL, self.feature_dim, device=encoded_feature.get_device())
        time_queries = self.PE(time_queries)
        tgt_mask = self.biased_mask[:, :TL, :TL].clone().detach().to(device=self.device).repeat(B,1,1)

        listener_reaction = self.listener_reaction_decoder_1(tgt=time_queries, memory=motion_sample.unsqueeze(1), tgt_mask=tgt_mask)
        listener_reaction = self.listener_reaction_decoder_2(listener_reaction, listener_reaction, tgt_mask=tgt_mask)

       
        listener_3dmm_out = self.listener_reaction_3dmm_map_layer(listener_reaction)


        listener_emotion_out = self.listener_reaction_emotion_map_layer(
            torch.cat((listener_3dmm_out, listener_reaction), dim=-1))

        # listener_emotion_au = self.listener_emotion_AU(listener_emotion_out)
        # listener_emotion_emt = self.listener_emotion_emotion(listener_emotion_out)

        # listener_emotion = torch.cat([listener_emotion_au, listener_emotion_emt], dim=-1)

        return listener_3dmm_out, listener_emotion_out, dist

    def reset_window_size(self, window_size):
        self.window_size = window_size

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
    def __init__(self, img_size=224, audio_dim = 78, feature_dim = 128, use_hubert=False, seq_len=750, device = 'cpu'):
        super(SpeakerBehaviourEncoder, self).__init__()

        self.device = device
        
        # self.video_encoder = VideoEncoder(img_size=img_size, feature_dim=feature_dim, device=device)
        self.video_encoder = VideoEncoder(feature_dim=512)

        self.use_hubert = use_hubert
        # if self.use_hubert:
        print ("Using HuBERT.")
        self.audio_encoder = HuBERTEncoder(feature_dim=512)
        #TODO: use this fusion 
        self.fusion_layer = nn.Linear(512, 256)
        transformer_layer = nn.TransformerEncoderLayer(d_model=256, 
                                                    nhead=4,
                                                    dim_feedforward=512,
                                                    dropout=0.3)
        self.encoder = nn.TransformerEncoder(transformer_layer, num_layers=1)

        self.lin = nn.Linear(256, feature_dim)

    def forward(self, video, audio):
        video_feature = self.video_encoder(video)
        audio_feature = self.audio_encoder(audio)
     
        fused = self.fusion_layer(torch.cat((video_feature, audio_feature), dim=1))
        encoded_feature = self.lin(self.encoder(fused))

        return encoded_feature



class TransformerVAEPro(nn.Module):
    def __init__(self, img_size=224, audio_dim = 78, output_3dmm_dim = 58, output_emotion_dim = 25, feature_dim = 128, seq_len=751, max_seq_len=751, online = False, window_size = 8, use_hubert=False, device = 'cuda'):
        super(TransformerVAEPro, self).__init__()

        self.img_size = img_size
        self.feature_dim = feature_dim
        self.output_3dmm_dim = output_3dmm_dim
        self.output_emotion_dim = output_emotion_dim
        self.seq_len = seq_len
        self.online = online
        self.window_size = window_size
        self.use_hubert = use_hubert
        self.speaker_behaviour_encoder = SpeakerBehaviourEncoder(img_size, audio_dim, feature_dim, use_hubert, seq_len, device)
        self.reaction_decoder = Decoder(output_3dmm_dim = output_3dmm_dim, output_emotion_dim = output_emotion_dim, feature_dim = feature_dim, max_seq_len=max_seq_len, device=device, window_size = self.window_size, online = online)
        # self.fusion = nn.Linear(feature_dim + self.output_3dmm_dim + self.output_emotion_dim, feature_dim)

    def forward(self, speaker_video=None, speaker_audio=None, **kwargs):

        """
        input:
        video: (batch_size, seq_len, 3, img_size, img_size)
        audio: (batch_size, raw_wav)

        output:
        3dmm_vector: (batch_size, seq_len, output_3dmm_dim)
        emotion_vector: (batch_size, seq_len, output_emotion_dim)
        distribution: [dist_1,...,dist_n]
        """

        distribution = []
      
        encoded_feature = self.speaker_behaviour_encoder(speaker_video, speaker_audio)
        listener_3dmm_out, listener_emotion_out, dist = self.reaction_decoder(encoded_feature)
        distribution.append(dist)
        return listener_3dmm_out, listener_emotion_out, distribution


    def reset_window_size(self, window_size):
        self.window_size = window_size
        self.reaction_decoder.reset_window_size(window_size)



if __name__ == "__main__":
    pass
