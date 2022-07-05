import numpy as np
from scipy import signal


def circ_mean(a, w=1):
    """
    Calculating a circular mean

    Args:
        a: phases
        w: magnitudes
    
    Returns:
        phase of summed vector
        magnitude of summed vector
    """
    sum_sin = np.sum(np.sin(a) * w)
    sum_cos = np.sum(np.cos(a) * w)
    return np.arctan2(sum_sin, sum_cos), np.linalg.norm([sum_sin, sum_cos])


def wrap(x):
    """
    Wraps x between -pi and pi
    """
    return np.arctan2(np.sin(x), np.cos(x))


def complex_wavelet(timestep, freq, cycles, kernel_length=5):
    """
    create wavelets
    args:
        timestep: simulation timestep in seconds
        freq: frequency of the wavelet
        cycles: number of oscillations of wavelet
        kernel_length: adapted per frequency
    note:
        normalisation as in: https://www.frontiersin.org/articles/10.3389/fnhum.2010.00198/full#B22
        retains signal energy, irrespective of freq, sum of the length of the wavelet is 1
    """

    gauss_sd = cycles/(2 * np.pi * freq)
    t = np.arange(0, kernel_length * gauss_sd, timestep)
    t = np.r_[-t[::-1], t[1:]]
    gauss = (1/(np.sqrt(2*np.pi)*gauss_sd)) * np.exp(-(t ** 2) / (2 * gauss_sd**2))
    sine = np.exp(2j * np.pi * freq * t)
    wavelet = gauss*sine*timestep

    return wavelet


def inst_phase(sign, kernel, t, f, ref_phase=True, mode = "same"):

    """
    Extract intstantaneous phase

    Args:
        sign: signal
        kernel: e.g. wavelet
        t: time in s
        f: frequency to use in Hz
        ref_phase: whether to use reference phase
        mode: e.g. padding for np.convolve
    Returns:
        phase: instantaneous phase
        amp: instantaneous amplitude
    
    """
    # TO DO:
    # use FFT + multiplication?

    conv = np.convolve(sign, kernel, mode=mode)

    # cut off more in case kernel is too long
    if len(conv) > len(sign):
        st = (len(conv) - len(sign)) // 2
        conv = conv[st : st + len(sign)]
    amp = np.abs(conv)
    arg = np.angle(conv)
    if ref_phase:
        ref = wrap(2 * np.pi * f * t)
        phase = wrap(arg - ref)
    else:
        phase = arg
    return phase, amp


def scalogram(sign, cycles, t, timestep, freqs, kernel_length=5):
    """
    Creates a scalogram of a signal
    
    Args: 
        sign: signal
        cycles: cycles parameter for wavelet
        t: time in s
        timestep: timestep in s
        freqs: frequencies to use
        kernel_length: wavelet kernel length
    
    Returns:
        phasegram, phases per time and frequency
        ampgram, amplitudes per time and frequency

    
    """
    phasegram = np.zeros((len(freqs), len(t)))
    ampgram = np.zeros((len(freqs), len(t)))
    for i, f in enumerate(freqs):
        kernel = complex_wavelet(timestep, f, cycles, kernel_length)
        phase, amp = inst_phase(sign, kernel, t, f)
        phasegram[i] = phase
        ampgram[i] = amp*np.sqrt(2)
    return phasegram, ampgram
