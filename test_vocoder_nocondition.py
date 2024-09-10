import torch
import torchaudio
from supervoice_flow.audio import resampler, spectogram
from supervoice_flow.config import config
from supervoice_flow.model import AudioFlow
import pyworld as pw
from training.dataset import load_distorted_sampler
import matplotlib.pyplot as plt
import io
import soundfile as sf
device = "cpu"

#
# Vocoder
#

vocoder = torch.hub.load(repo_or_dir='ex3ndr/supervoice-vocoder', model='bigvsan')
vocoder = vocoder.to(device)
def do_vocoder(src):
    with torch.no_grad():
        return vocoder.generate(src.T.to(torch.float32)).squeeze(0)

#
# Tools
#

def do_spectogram(src):
    return spectogram(src, config.audio.n_fft, config.audio.n_mels, config.audio.hop_size, config.audio.win_size, config.audio.mel_norm, config.audio.mel_scale, config.audio.sample_rate)

def plot_audio(audio):
    # 提取音频数据和采样率
    waveform = audio.data
    sample_rate = config.audio.sample_rate
    with io.BytesIO() as audio_file:
        sf.write(audio_file, audio, sample_rate, format='WAV')
        audio_file.seek(0)  # 重置文件指针
        audio_array, _ = sf.read(audio_file, dtype='float32')
        sf.write("audio.wav", audio_array, sample_rate)

    # Pitch detector
    f0, t0 = pw.dio(waveform.cpu().squeeze(0).numpy().astype('double'), config.audio.sample_rate, frame_period=(1000 * config.audio.hop_size)/config.audio.sample_rate)
    f0 = 2595 * torch.log10(1 + torch.tensor(f0) / 700)
    f0 = f0 / (2595 * torch.log10(torch.tensor(1 + 24000 / 700))) * (100 - 1)

    # Plot
    _, axis = plt.subplots(1, 1, figsize=(20, 5))
    axis.imshow(spec.cpu(), cmap="viridis", vmin=-10, vmax=0, origin="lower", aspect="auto")
    axis.plot(f0, color="white")
    # axis.set_title(title)
    plt.tight_layout()
    plt.savefig('audio.png')



def plot_debug(waveform, file_name, wave=True):

    # Preprocess
    if wave:
        spec = do_spectogram(waveform)
    else:
        spec = waveform


    # Pitch detector
    # f0, t0 = pw.dio(np.ascontiguousarray(waveform.cpu().squeeze(0).numpy().astype('double')), config.audio.sample_rate, frame_period=(1000 * config.audio.hop_size)/config.audio.sample_rate)
    # f0 = 2595 * torch.log10(1 + torch.tensor(f0) / 700)
    # f0 = f0 / (2595 * torch.log10(torch.tensor(1 + 24000 / 700))) * (100 - 1)

    # Plot
    _, axis = plt.subplots(1, 1, figsize=(20, 5))
    axis.imshow(spec.cpu(), cmap="viridis", vmin=-10, vmax=0, origin="lower", aspect="auto")
    # axis.plot(f0, color="white")
    # axis.set_title(title)
    plt.tight_layout()
    plt.savefig(file_name)

#
# Dataset
#

sampler = load_distorted_sampler("/mnt/afs/chenyun/TTSFlow/ref_wav_tiny", 15, False)

# AudioFlow
predictor = AudioFlow(config)
checkpoint = torch.load(f'/mnt/afs/chenyun/TTSFlow/tinydata/large-03.16000.pt', map_location="cpu")
predictor.load_state_dict(checkpoint['model'])
print("Flow at ", checkpoint['step'])

def do_effect(src, steps = 8):
    src = (src - config.audio.norm_mean) / config.audio.norm_std
    pr, _ = predictor.sample(audio = src.to(torch.float32), steps = steps)
    return ((pr * config.audio.norm_std) + config.audio.norm_mean).to(torch.float32)

spec, audio = sampler()
spec_processed = do_effect(spec, steps = 4)


# waveform, sample_rate = torchaudio.load('/mnt/afs/chenyun/TTSFlow/ref_wav/192090.wav')

# stft_params = {
#   'n_fft': 2048,
#   'hop_length': 256,
# }
# y = waveform
# y_np = y.numpy()
# mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, **stft_params, power=1)(y) # mel_scale(S**x)
# spec1 = torchaudio.transforms.InverseMelScale(n_stft=stft_params['n_fft'] // 2 + 1, sample_rate=sample_rate)(mel)  # S**x
# torch_y = torchaudio.transforms.GriffinLim(**stft_params, power=1)(spec1)
# torchaudio.save('reconstructed_origin_spec_torch.wav', torch_y, sample_rate)

# stft_params = {
#   'n_fft': 1024,
#   'hop_length': 256,
# }
# x=1
# torch_y = torchaudio.transforms.GriffinLim(**stft_params, power=x)(spec.transpose(0, 1).unsqueeze(0).to(torch.float32))
# torchaudio.save('reconstructed_origin_spec_torch.wav', torch_y, config.audio.sample_rate)

# torch_y = torchaudio.transforms.GriffinLim(**stft_params, power=x)(spec_processed.detach().transpose(0, 1))
# torchaudio.save('reconstructed_flow_spec_torch.wav', torch_y, config.audio.sample_rate)


# wav_aug_2 = do_vocoder(spec)
wav_aug_3 = do_vocoder(spec_processed)

raw_spec_after_vocoder = do_vocoder(spec)


torchaudio.save("sample.wav", wav_aug_3.unsqueeze(0), config.audio.sample_rate)
torchaudio.save("origin.wav", audio.unsqueeze(0), config.audio.sample_rate)
torchaudio.save("raw_spec_after_vocoder.wav", raw_spec_after_vocoder.unsqueeze(0), config.audio.sample_rate)


plot_debug(wav_aug_3, 'sample.png')
plot_debug(audio, 'origin.png')
plot_debug(spec_processed.detach().transpose(0, 1), 'before_vocoder.png', False)


# plot_debug(wav)
# plot_debug(wav_aug)
# plot_debug(wav_aug_2)
# plot_debug(wav_aug_3)
# plot_audio(audio)