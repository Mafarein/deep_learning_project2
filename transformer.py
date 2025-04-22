import torch
import torch.nn as nn
import torchaudio

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x

class AudioTransformer(nn.Module):
    def __init__(self, n_classes, n_mels=64, d_model=128, nhead=4, num_layers=4, dropout=0):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=n_mels
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()
        
        self.input_proj = nn.Linear(n_mels, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, waveform):
        with torch.no_grad():
            mel = self.mel_spec(waveform)
            mel_db = self.db_transform(mel)

        mel_db = mel_db.squeeze(1)
        mel_db = mel_db.transpose(1, 2)

        x = self.input_proj(mel_db)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        x = x.mean(dim=1)
        return self.classifier(x)
