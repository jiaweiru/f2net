import torch
import torchaudio
import speechbrain
from speechbrain.processing import features,signal_processing
import matplotlib.pyplot as plt


def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c + 1}")
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=False)


def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None, ylim=None, winl=256, hopl=128, mode='psd'):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate, NFFT=winl, noverlap=hopl, mode=mode)
        # win_length=256, hop_length=128, psd, window_hanning
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c + 1}")
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=False)


if __name__ == "__main__":
    path = "./data/LibriSpeech/dev-clean-2/2035/147961/2035-147961-0025.flac"

    """
    origin spec
    """
    # waveform, sr = torchaudio.load(path)
    # plot_specgram(waveform, sr, winl=400, hopl=160, title="origin")
    # plot_specgram(waveform, sr, winl=400, hopl=160, mode='phase')

    """
    downsaple
    plot spec
    """
    # downsample = torchaudio.transforms.Resample(16000, 8000)
    # wave_down = downsample(waveform)
    # plot_specgram(wave_down, sr // 2, winl=200, hopl=80, ylim=[0, 8000], title="down sample")
    # plot_specgram(wave_dowm, sr // 2, ylim=[0, 8000], winl=200, hopl=80, mode='phase')

    """
    upsample 
    plot spec
    """
    # upsample = torchaudio.transforms.Resample(8000, 16000)
    # wave_up = upsample(wave_down)
    # plot_specgram(wave_up, sr, winl=400, hopl=160, title="up sample")
    # plot_specgram(wave_up, sr, winl=400, hopl=160, mode='phase')

    """
    save specs of clean, downsample, upsample
    """
    # torchaudio.save("./test_clean.flac", waveform, 16000)
    # torchaudio.save("./test_up.flac", wave_up, 16000)
    # torchaudio.save("./test_down.flac", wave_down, 8000)

    """
    about infer
    """
    wav_input, sr = torchaudio.load("./test_result/up10.flac")
    wav_infer, _ = torchaudio.load("./test_result/infer10.flac")
    wav_clean, _ = torchaudio.load("./test_result/clean10.flac")
    plot_waveform(wav_input, 16000, title="up")
    plot_waveform(wav_clean, 16000, title="clean")
    plot_waveform(wav_infer, 16000, title="infer")
    plot_specgram(wav_input, sr, winl=400, hopl=160, title="up")
    plot_specgram(wav_clean, sr, winl=400, hopl=160, title="clean")
    plot_specgram(wav_infer, sr, winl=400, hopl=160, title="infer")


