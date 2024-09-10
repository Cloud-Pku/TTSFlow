import os
import json
import subprocess
from pydub import AudioSegment
import aubio
import numpy as np

# 定义文件路径
json_file = '/mnt/afs/chenyun/TTSFlow/ref.json'
audio_root_dir = '/mnt/afs/chenyun/TTSFlow/ref_wav'
corpus_directory = '/mnt/afs/chenyun/TTSFlow/corpus'
output_directory = '/mnt/afs/chenyun/TTSFlow/output_directory'

# 使用 MFA 进行音素对齐
def align_audio(audio_file, text, output_dir):
    audio_dir = os.path.join(corpus_directory, os.path.basename(audio_file).replace('.wav', ''))
    os.makedirs(audio_dir, exist_ok=True)
    os.symlink(audio_file, os.path.join(audio_dir, os.path.basename(audio_file)))
    text_file = os.path.join(audio_dir, os.path.basename(audio_file).replace('.wav', '.txt'))
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(text)
    subprocess.run(['mfa', 'align', corpus_directory, 'tools/mandarin_pinyin_to_mfa_lty.dict', 'tools/mandarin_mfa.zip', output_dir])

# 解析 TextGrid 文件并提取音高信息
def extract_pitch_and_duration(audio_file, textgrid_file, output_csv):
    sound = AudioSegment.from_wav(audio_file)
    samplerate = sound.frame_rate
    win_s = 1024  # FFT size
    hop_s = win_s // 4  # Hop size

    pitch_o = aubio.pitch("yin", win_s, hop_s, samplerate)
    pitch_o.set_unit("Hz")
    pitch_o.set_tolerance(0.8)

    with open(textgrid_file, 'r', encoding='utf-8') as f:
        textgrid_content = f.read()

    intervals = []
    for line in textgrid_content.splitlines():
        if 'text = "' in line:
            label = line.split('"')[1]
            start_time = float(prev_line.split('=')[1].strip())
            end_time = float(line.split('=')[1].strip())
            intervals.append((start_time, end_time, label))
        prev_line = line

    with open(output_csv, 'w', encoding='utf-8') as f:
        f.write('phoneme,start_time,end_time,duration,pitch\n')
        for start_time, end_time, label in intervals:
            if label:
                duration = end_time - start_time
                start_sample = int(start_time * samplerate)
                end_sample = int(end_time * samplerate)
                samples = np.array(sound.get_array_of_samples()[start_sample:end_sample])
                pitch_values = []
                for i in range(0, len(samples), hop_s):
                    sample = samples[i:i + hop_s]
                    if len(sample) < hop_s:
                        break
                    pitch = pitch_o(sample)[0]
                    if pitch > 0:
                        pitch_values.append(pitch)
                if pitch_values:
                    pitch_mean = sum(pitch_values) / len(pitch_values)
                else:
                    pitch_mean = 0.0
                f.write(f'{label},{start_time},{end_time},{duration},{pitch_mean}\n')

# 主函数
def main():
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        audio_file = os.path.join(audio_root_dir, item['ref_wav_path'])
        text = item['prompt_text']
        # output_dir = os.path.join(output_directory, os.path.basename(item['ref_wav_path']).replace('.wav', ''))
        output_dir = output_directory
        os.makedirs(output_dir, exist_ok=True)

        # 对齐音频
        print('output_dir:', output_dir)
        align_audio(audio_file, text, output_dir)

        # 提取音高和时长
        textgrid_file = os.path.join(output_dir, os.path.basename(item['ref_wav_path']).replace('.wav', ''), os.path.basename(audio_file).replace('.wav', '.TextGrid'))
        output_csv = os.path.join(output_dir, 'output.csv')
        extract_pitch_and_duration(audio_file, textgrid_file, output_csv)

if __name__ == '__main__':
    main()