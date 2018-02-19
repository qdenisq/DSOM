import numpy as np
import scipy.io.wavfile
from scipy.io.wavfile import read
from scipy import signal


def extract_spectrogram(fname, nperseg=512, noverlap=384):
    sample_rate, samples = read(fname)
    frequencies, times, spectogram = signal.spectrogram(samples, sample_rate, nperseg=nperseg, noverlap=noverlap)
    dBS = 20 * np.log10(spectogram)
    dBS = scipy.stats.threshold(dBS, threshmin=0., newval=0.)
    return dBS
