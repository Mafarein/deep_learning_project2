import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import random

COMMANDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
SILENCE_LABEL = 'silence'
UNKNOWN_LABEL = 'unknown'


class SpeechCommandsDataset(Dataset):
    def __init__(self, data_dir, mode='train', transform=None, silence_percentage=0.05, unknown_percentage=1):

        self.data_dir = data_dir + '/audio'
        self.transform = transform
        self.mode = mode
        self.silence_percentage = silence_percentage
        self.unknown_percentage = unknown_percentage
        # Load the validation and testing file lists (from dataset)
        self.val_list = self._load_list(os.path.join(data_dir, "validation_list.txt"))
        self.test_list = self._load_list(os.path.join(data_dir, "testing_list.txt"))

        self.background_noises = self._load_background_noises()

        # Build the list of samples and their labels
        self.samples = self._build_file_list()

    def _load_list(self, filepath):
        # Load list of files
        if not os.path.exists(filepath):
            return []
        with open(filepath, 'r') as f:
            return set([line.strip() for line in f])

    def _load_background_noises(self):
        # Load background noises
        background_dir = os.path.join(self.data_dir, "_background_noise_")
        if not os.path.exists(background_dir):
            return []
        noise_files = [os.path.join(background_dir, file) for file in os.listdir(background_dir) if
                       file.endswith('.wav')]
        return noise_files

    def _which_set(self, filepath):
        if filepath in self.val_list:
            return 'validation'
        if filepath in self.test_list:
            return 'testing'
        return 'train'

    def _build_file_list(self):
        command_samples = []
        unknown_samples = []

        if self.mode in ['validation', 'testing']:
            for file_path in (self.val_list if self.mode == 'validation' else self.test_list):
                label = self._extract_label_from_path(file_path)
                if label in COMMANDS:
                    command_samples.append((file_path, label))
                else:
                    unknown_samples.append((file_path, UNKNOWN_LABEL))
            silence_samples = [(None, SILENCE_LABEL)] * int(
                len(command_samples + unknown_samples) * self.silence_percentage)
            return command_samples + unknown_samples + silence_samples

        for label in os.listdir(self.data_dir):
            label_dir = os.path.join(self.data_dir, label)
            if not os.path.isdir(label_dir) or label == "_background_noise_":
                continue
            for file_name in os.listdir(label_dir):
                file_path = os.path.join(label, file_name)
                set_type = self._which_set(file_path)
                if set_type != 'train':
                    continue
                if label in COMMANDS:
                    command_samples.append((file_path, label))
                else:
                    unknown_samples.append((file_path, UNKNOWN_LABEL))

        num_unknown = int(len(command_samples) * self.unknown_percentage)
        unknown_samples = random.sample(unknown_samples, min(num_unknown, len(unknown_samples)))

        silence_samples = [(None, SILENCE_LABEL)] * int(
            (len(command_samples) + len(unknown_samples)) * self.silence_percentage)

        all_samples = command_samples + unknown_samples + silence_samples
        random.shuffle(all_samples)
        return all_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]

        if label == SILENCE_LABEL:
            noise_file = random.choice(self.background_noises)
            waveform, sample_rate = torchaudio.load(noise_file)
            start = random.randint(0, waveform.size(1) - 16000)
            waveform = waveform[:, start:start + 16000]
        else:
            waveform, sample_rate = torchaudio.load(os.path.join(self.data_dir, file_path))
            waveform = self._pad_or_trim(waveform)

        if self.transform:
            waveform = self.transform(waveform)

        if label in COMMANDS:
            label_idx = COMMANDS.index(label)
        elif label == SILENCE_LABEL:
            label_idx = len(COMMANDS)
        else:
            label_idx = len(COMMANDS) + 1

        return waveform, label_idx

    def _pad_or_trim(self, waveform):
        # Make sure every audio is 1 second (16000 samples)
        if waveform.shape[1] > 16000:
            waveform = waveform[:, :16000]
        elif waveform.shape[1] < 16000:
            padding = 16000 - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        return waveform

    def _extract_label_from_path(self, file_path):
        return file_path.split('/')[0]


def create_dataloader(data_dir, batch_size=32, mode='train', transform=None):
    dataset = SpeechCommandsDataset(data_dir, mode=mode, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(mode == 'train'))
    return loader

