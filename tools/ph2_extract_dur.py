import os
import json
import subprocess


def align_audio(audio_file, text, output_dir):
    audio_dir = os.path.join(corpus_directory, os.path.basename(audio_file).replace('.wav', ''))
    os.makedirs(audio_dir, exist_ok=True)
    os.symlink(audio_file, os.path.join(audio_dir, os.path.basename(audio_file)))
    text_file = os.path.join(audio_dir, os.path.basename(audio_file).replace('.wav', '.txt'))
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(text)
    subprocess.run(['mfa', 'align', corpus_directory, 'tools/mandarin_pinyin_to_mfa_lty.dict', 'tools/mandarin_mfa.zip', output_dir])

def align_audio(lab_path, textgrid_path, tmp_path):
    subprocess.run(['mfa', 'align', lab_path, 'tools/mandarin_pinyin_to_mfa_lty.dict', 'tools/mandarin_mfa.zip', textgrid_path, '--clean', '-t', tmp_path, '--single_speaker', '--use_mp'])


if __name__ == '__main__':
    json_path = '/mnt/afs/chenyun/TTSFlow/dataset/ref.json'
    ph1_output_path = '/mnt/afs/chenyun/TTSFlow/dataset/ref_wav'
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        name = item["ref_wav_path"].split('.')[0]
        lab_path = os.path.join(ph1_output_path, name)
        textgrid_path = lab_path
        tmp_path = os.path.join(lab_path, 'tmp')
        align_audio(lab_path, textgrid_path, tmp_path)
