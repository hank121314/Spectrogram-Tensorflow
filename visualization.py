from scipy.io import wavfile
import numpy as np
from scipy import signal
from scipy.fftpack import fft


class Visualization():

    def __init__(self,path):
        self.path=path

    def readWAVToRate(self):
        self.rate,self.data=wavfile.read(self.path)
        return self.rate,self.data

    def log_specgram(self, data, sample_rate, window_size=20, step_size=10, eps=1e-10):
        nperseg = int(round(window_size * sample_rate / 1e3))
        noverlap = int(round(step_size * sample_rate / 1e3))
        freqs, times, spec = signal.spectrogram(data,
                                            fs=sample_rate,
                                            window='hann',
                                            nperseg=nperseg,
                                            noverlap=noverlap,
                                            detrend=False)
        return freqs, times, np.log(spec.T.astype(np.float32) + eps)

    def custom_fft(self, data, fs):
        T = 1.0 / fs
        N = data.shape[0]
        yf = fft(data)
        xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
        # FFT is simmetrical, so we take just the first half
        # FFT is also complex, to we take just the real part (abs)
        vals = 2.0/N * np.abs(yf[0:N//2])
        return xf, vals