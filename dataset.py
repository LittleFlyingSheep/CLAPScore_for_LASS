import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
from librosa.filters import mel as librosa_mel_fn
import numpy as np
import csv
from utils import calculate_sisdr, calculate_sdr, calculate_sdr_tensor

class AudioTextDataset4CLAP(Dataset):
    def __init__(
            self,
            sampling_rate=16000,
            text_queries='text_queries.csv',
            audio_dir='audio_dir',
    ):

        self.sampling_rate = sampling_rate

        with open(text_queries) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            eval_list = [row for row in csv_reader][1:]

        self.eval_list = eval_list
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.eval_list)

    def __getitem__(self, item):

        audio_filename, caption = self.eval_list[item]

        audio_file = Path(self.audio_dir) / audio_filename
        waveform, sampling_rate = torchaudio.load(audio_file)
        if sampling_rate != self.sampling_rate:
            waveform = torchaudio.transforms.Resample(sampling_rate, self.sampling_rate)(waveform)

        return {
            "waveform": waveform,
            "caption": caption,
            "filename": audio_filename
        }

def collate_fn(list_data_dict):
    at_list_data_dict = [data_dict for data_dict in list_data_dict]

    at_data_dict = {}

    if len(at_list_data_dict) > 0:
        for key in at_list_data_dict[0].keys():
            at_data_dict[key] = [at_data_dict[key] for at_data_dict in at_list_data_dict]
            if key in ["source_waveform", "noise_waveform", "mixture_waveform", "separated_waveform", "waveform"]:
                at_data_dict[key] = torch.stack(at_data_dict[key])

    return at_data_dict

def get_dataloader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
)