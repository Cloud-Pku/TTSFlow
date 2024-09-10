import os
import json
import subprocess
import torchaudio
import aubio
import numpy as np
import pyworld as pw
from praatio import textgrid
import torch

pitch_threshold = 50
def extract_pitch_and_duration(audio_file, textgrid_file, output_csv):
    sound = torchaudio.load(audio_file)
    samplerate = sound[1]
    hop_size = 128  # 你可以根据实际情况调整这个值

    # 将音频信号转换为 numpy 数组
    samples = np.array(sound[0]).astype('double')
    samples = samples.mean(0)

    # 使用 pyworld 计算音高
    f0, t0 = pw.dio(samples, samplerate, frame_period=(1000 * hop_size) / samplerate)
    f0_raw = torch.tensor(f0)
    f0 = 2595 * torch.log10(1 + torch.tensor(f0) / 700)
    f0 = f0 / (2595 * torch.log10(torch.tensor(1 + 24000 / 700))) * (100 - 1)

    tg = textgrid.openTextgrid(textgrid_file, includeEmptyIntervals=True)
    intervals = tg._tierDict['words'].entries

    with open(output_csv, 'w', encoding='utf-8') as f:
        f.write('phoneme,start_time,end_time,duration,pitch\n')
        for interval in intervals:
            start_time, end_time, label = interval
            # if label:
            duration = end_time - start_time
            start_frame = int(start_time * (samplerate / hop_size))
            end_frame = int(end_time * (samplerate / hop_size))
            pitch_values = f0[start_frame:end_frame]
            # pitch_values = pitch_values[pitch_values > pitch_threshold]
            # if pitch_values.numel() > 0:
            pitch_mean = pitch_values.mean().item()
            # else:
            #     pitch_mean = 0.0
            f.write(f'{label},{start_time},{end_time},{duration},{pitch_mean}\n')


if __name__ == '__main__':

    audio_file = '/mnt/afs/chenyun/TTSFlow/ref_wav/1001.wav'
    textgrid_file = '/mnt/afs/chenyun/TTSFlow/dataset/ref_wav/1001/1001.TextGrid'
    output_csv = '/mnt/afs/chenyun/TTSFlow/dataset/ref_wav/1001/1001.csv'
    json_path = '/mnt/afs/chenyun/TTSFlow/dataset/ref.json'
    ph1_output_path = '/mnt/afs/chenyun/TTSFlow/dataset/ref_wav'
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        name = item["ref_wav_path"].split('.')[0]
        lab_path = os.path.join(ph1_output_path, name)

        extract_pitch_and_duration(audio_file, textgrid_file, output_csv)