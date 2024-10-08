import torch
import torchaudio
from supervoice_flow.config import config
from supervoice_flow.audio import load_mono_audio, spectogram
from .audio import do_reverbrate
from pathlib import Path
import random
import os
import stat

def lock_directory(directory_path):
    # 获取当前权限
    current_permissions = os.stat(directory_path).st_mode
    new_permissions = current_permissions & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH
    os.chmod(directory_path, new_permissions)

def unlock_directory(directory_path):
    # 获取当前权限
    current_permissions = os.stat(directory_path).st_mode
    new_permissions = current_permissions | stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    os.chmod(directory_path, new_permissions)

def load_clean_sampler(datasets, duration, return_source = False):

    # Target duration
    frames = int(duration * config.audio.sample_rate)

    # Load the datasets
    # files = []
    # if isinstance(datasets, str):
    #     with open(datasets + "files_all.txt", 'r') as file:
    #         dataset_files = file.read().splitlines()
    #     dataset_files = [datasets + p + ".flac" for p in dataset_files]
    # else:
    #     dataset_files = []
    #     for dataset in datasets:
    #         dataset_files += list(Path(dataset).rglob("*.wav")) + list(Path(dataset).rglob("*.flac"))
    #     dataset_files = [str(p) for p in dataset_files]
    # files += dataset_files
    # print(f"Loaded {len(files)} files")
    # dataset_files = list(Path(datasets).rglob("*.wav"))
    # dataset_files = [str(p) for p in dataset_files]
    # files = dataset_files

    # Sample a single item
    def sample_item():
        # lock_directory(Path(datasets))
        dataset_files = list(Path(datasets).rglob("*.wav"))
        dataset_files = [str(p) for p in dataset_files]
        files = dataset_files
        # Load random audio
        while True:
            try:
                f = random.choice(files)
                audio = load_mono_audio(f, config.audio.sample_rate)
                break
            except Exception as e:
                print(files)
                print(e)
        # unlock_directory(Path(datasets))
        # Pad or trim audio
        if audio.shape[0] < frames:
            padding = frames - audio.shape[0]
            padding_left = random.randint(0, padding)
            padding_right = padding - padding_left
            audio = torch.nn.functional.pad(audio, (padding_left, padding_right), value=0)
        else:
            start = random.randint(0, audio.shape[0] - frames)
            audio = audio[start:start + frames]

        # Spectogram
        spec = spectogram(audio, 
            n_fft = config.audio.n_fft, 
            n_mels = config.audio.n_mels, 
            n_hop = config.audio.hop_size, 
            n_window = config.audio.win_size,  
            mel_norm = config.audio.mel_norm, 
            mel_scale = config.audio.mel_scale, 
            sample_rate = config.audio.sample_rate
        ).transpose(0, 1).to(torch.float16)

        # Return result
        if return_source:
            return spec, audio
        else:
            return spec

    return sample_item

def load_effected_sampler(datasets, effect, duration, return_source = False):

    # Target duration
    frames = int(duration * config.audio.sample_rate)

    # Load the datasets
    # files = []
    # for dataset in datasets:
    #     dataset_files = list(Path(dataset).rglob("*.wav")) + list(Path(dataset).rglob("*.flac"))
    #     dataset_files = [str(p) for p in dataset_files]
    #     files += dataset_files
    dataset_files = list(Path(datasets).rglob("*.wav"))
    dataset_files = [str(p) for p in dataset_files]
    files = dataset_files
    
    # Sample a single item
    def sample_item(file_idx=None, padding_trim_random = True):

        # Load random audio
        if file_idx == None:
            f = random.choice(files)
        else:
            f = files[file_idx]
        audio = load_mono_audio(f, config.audio.sample_rate)

        # Pad or trim audio
        if padding_trim_random:
            if audio.shape[0] < frames:
                padding = frames - audio.shape[0]
                padding_left = random.randint(0, padding)
                padding_right = padding - padding_left
                audio = torch.nn.functional.pad(audio, (padding_left, padding_right), value=0)
            else:
                start = random.randint(0, audio.shape[0] - frames)
                audio = audio[start:start + frames]
        else:
            if audio.shape[0] < frames:
                padding_left = 0
                padding_right = frames - audio.shape[0]
                audio = torch.nn.functional.pad(audio, (padding_left, padding_right), value=0)
            else:
                audio = audio[:frames]

        # Apply effect
        # audio_with_effect = effect(audio)

        # Spectogram
        spec = spectogram(audio, 
            n_fft = config.audio.n_fft, 
            n_mels = config.audio.n_mels, 
            n_hop = config.audio.hop_size, 
            n_window = config.audio.win_size,  
            mel_norm = config.audio.mel_norm, 
            mel_scale = config.audio.mel_scale, 
            sample_rate = config.audio.sample_rate
        ).transpose(0, 1).to(torch.float16)

        # Spectogram with effect
        # spec_with_effect = spectogram(audio_with_effect, 
        #     n_fft = config.audio.n_fft, 
        #     n_mels = config.audio.n_mels, 
        #     n_hop = config.audio.hop_size, 
        #     n_window = config.audio.win_size,  
        #     mel_norm = config.audio.mel_norm, 
        #     mel_scale = config.audio.mel_scale, 
        #     sample_rate = config.audio.sample_rate
        # ).transpose(0, 1).to(torch.float16)

        # Return results
        if return_source:
            return (spec, spec_with_effect, audio, audio_with_effect)
        else:
            return (spec, audio)

    # Return generator
    return sample_item


def load_distorted_sampler(datasets, duration, return_source = False):

    # Codecs for distortion
    codec_probability = 0.3
    codecs = [
        {'format': "wav", 'encoder': "pcm_mulaw"},
        {'format': "g722"},
        {"format": "mp3", "codec_config": torchaudio.io.CodecConfig(bit_rate=8_000)},
        {"format": "mp3", "codec_config": torchaudio.io.CodecConfig(bit_rate=64_000)}
    ]

    # RIRs
    # rir_probability = 0.8
    # rir_files = []
    # with open('./external_datasets/rir-1/files.txt', 'r') as file:
    #     for line in file:
    #         rir_files.append("./external_datasets/rir-1/" + line.strip())

    # Apply RIR effect
    def effect(audio):

        # Pick RIR
        rir = None
        if random.random() < rir_probability:
            rir = random.choice(rir_files)
            rir = load_mono_audio(rir, config.audio.sample_rate)

        # Pick codec
        codec = None
        if random.random() < codec_probability:
            codec = random.choice(codecs)

        # Pick effect
        effect = None

        # Apply RIR
        if rir is not None:
            audio = do_reverbrate(audio, rir)

        # Apply processor
        if codec is not None or effect is not None:
            args = {}
            if effect is not None:
                args['effect'] = effect
            if codec is not None:
                args.update(codec)
            effector = torchaudio.io.AudioEffector(**args)
            audio = effector.apply(audio.unsqueeze(0).T, config.audio.sample_rate).T[0]

        return audio

    # Load sampler
    sampler = load_effected_sampler(datasets, effect, duration, return_source)

    return sampler

def load_distorted_loader(datasets, duration, batch_size, num_workers, return_source = False):

    # Load sampler
    sampler = load_distorted_sampler(datasets, duration, return_source)

    # Load dataset
    class DistortedDataset(torch.utils.data.IterableDataset):
        def __init__(self, sampler):
            self.sampler = sampler
        def generate(self):
            while True:
                yield self.sampler()
        def __iter__(self):
            return iter(self.generate())
    dataset = DistortedDataset(sampler)

    # Load loader
    loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, num_workers = num_workers, pin_memory = True, shuffle=False)

    return loader

def load_clean_loader(datasets, duration, batch_size, num_workers, return_source = False):

    # Load sampler
    sampler = load_clean_sampler(datasets, duration, return_source)

    # Load dataset
    class DistortedDataset(torch.utils.data.IterableDataset):
        def __init__(self, sampler):
            self.sampler = sampler
        def generate(self):
            while True:
                yield self.sampler()
        def __iter__(self):
            return iter(self.generate())
    dataset = DistortedDataset(sampler)

    # Load loader
    loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, num_workers = num_workers, pin_memory = True, shuffle=False)

    return loader