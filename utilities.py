import librosa
from librosa import display
import matplotlib.pyplot as plt
import scipy
import numpy as np
import soundfile as sf

def load_audio(filepath):
    """
    params :
    -> filepath : location to filepath
    returns : 
    -> samples : array containing audio data in time domain
    -> sampling_rate : sampling rate of original audio
    """
    samples, sampling_rate = librosa.load(filepath, sr = None, mono = True, offset = 0.0, duration = None)
    return samples, sampling_rate


def plot_waveshow_td_representation(audio, sampling_rate):
    """
    plots audio in time domain
    params : 
    -> audio : td represented audio data
    -> sampling_rate : sampling_rate of original audio
    """    
    plt.figure(figsize=(15,7))
    display.waveshow(y = audio, sr = sampling_rate)
    plt.xlabel("Time (in seconds)")
    plt.ylabel("Amplitude")
    return plt.show()


def perform_fft(audio, sampling_rate):
    """
    performs fast fourier transform on time domain audio data,
    params : 
    -> audio : td represented audio data
    -> sampling_rate : sampling_rate of original audio
    returns :
    -> frequency : frequency domain audio data
    -> magnitude : range of magnitude of audio
    """
    n = len(audio)
    T = 1/sampling_rate
    frequency = scipy.fft.fft(audio)
    magnitude = np.linspace(0.0, 1.0/(2.0*T), n//2)
    return frequency, magnitude

def plot_fd_representation(frequency_data, magnitude):
    """
    plots frequency domain data
    params :
    -> frequency : frequency domain audio data
    -> magnitude : range of magnitude of audio
    """
    plt.rcParams["figure.figsize"] = [15, 7]
    n = len(frequency_data)
    fig, ax = plt.subplots()
    ax.plot(magnitude, 2.0/n * np.abs(frequency_data[:n//2]))
    plt.grid()
    plt.xlabel("Frequency ->")
    plt.ylabel("Magnitude")
    return plt.show()


def extract_audio(filename, audio, sampling_rate):
    """
    params :
    -> filename : output file
    -> audio : time domain audio data
    -> sampling_rate : sampling rate of output data
    """
    sf.write(filename, audio, samplerate=sampling_rate)
    print("Extracted Successfully")


def spectrogram(samples, sample_rate, stride_ms = 10.0, 
                          window_ms = 20.0, max_freq = None, eps = 1e-14):
    """
    returns :
    -> specgram : spectogram data of audio
    """
    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)

    # Extract strided windows
    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[:len(samples) - truncate_size]
    nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(samples, 
                                          shape = nshape, strides = nstrides)
    
    assert np.all(windows[:, 1] == samples[stride_size:(stride_size + window_size)])

    # Window weighting, squared Fast Fourier Transform (fft), scaling
    weighting = np.hanning(window_size)[:, None]
    
    fft = np.fft.rfft(windows * weighting, axis=0)
    fft = np.absolute(fft)
    fft = fft**2
    
    scale = np.sum(weighting**2) * sample_rate
    fft[1:-1, :] *= (2.0 / scale)
    fft[(0, -1), :] /= scale
    
    # Prepare fft frequency list
    freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])
    
    # Compute spectrogram feature
    ind = np.where(freqs <= max_freq)[0][-1] + 1
    specgram = np.log(fft[:ind, :] + eps)
    return specgram


def plot_specgram(specgram, audio_len, sampling_rate):
    """
    params :
    -> specgram : spectrogram data
    -> audio_len : length of audio in seconds
    -> sampling_rate : sampling_rate of audio
    """
    plt.imshow(np.transpose(specgram), extent=[0,audio_len,0,sampling_rate//2], vmin=-30, vmax=0, cmap='magma', origin='lower', aspect='auto')
    # extent :
    #   x-axis limits : 0 to 5 seconds (time)
    #   y-axis limits : 0 to sampling rate (frequency)
    # vmin, vmax : -30db to 0db
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar()
    return plt.show()



if __name__ == "__main__":
    file_path = "./data/sample1/source.wav"
    samples, sampling_rate = load_audio(file_path)
    # plot_waveshow_td_representation(samples, sampling_rate)

    freq_data, mag = perform_fft(samples, sampling_rate)
    plot_fd_representation(freq_data, mag)


    specgramData = spectrogram(samples, sampling_rate, max_freq=max(freq_data))
    # plot_specgram(specgramData, len(samples)/sampling_rate, sampling_rate)    


