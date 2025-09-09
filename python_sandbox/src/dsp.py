import numpy as np

def hann(N: int): return np.hanning(N)

def fft_mag(x, fs):
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1/fs)
    return freqs, np.abs(X)

def rms(x): return float(np.sqrt(np.mean(x**2)))

def real_power(v, i): return float(np.mean(v*i))

def apparent_power(vrms, irms): return float(vrms * irms)

def power_factor(P, S): return float(P / max(S, 1e-12))

def thd(mag, freqs, f0=60.0, harmonics=(120,180,240)):
    fund = float(np.interp(f0, freqs, mag))
    harms = [float(np.interp(h, freqs, mag)) for h in harmonics]
    return float(np.sqrt(np.sum(np.square(harms))) / max(fund, 1e-12))

def dft_phasor(x: np.ndarray, fs: float, f0: float):
    """
    Single-bin DFT to estimate the fundamental complex phasor at f0.
    Returns complex value whose magnitude ~peak amplitude (not RMS).
    """
    N = len(x)
    n = np.arange(N, dtype=float)
    w = np.exp(-1j * 2*np.pi * f0 * n / fs)
    X = (2.0 / N) * np.sum(x * w)  # ~peak amplitude for a pure tone
    # Convert peak -> RMS for convenience
    rms = np.abs(X) / np.sqrt(2.0)
    phase_rad = np.angle(X)
    return rms, phase_rad  # RMS magnitude and phase (rad)

def crest_factor(x: np.ndarray) -> float:
    rms = np.sqrt(np.mean(x**2)) + 1e-12
    peak = np.max(np.abs(x))
    return float(peak / rms)

def form_factor(x: np.ndarray) -> float:
    rms = np.sqrt(np.mean(x**2)) + 1e-12
    avg_rect = np.mean(np.abs(x)) + 1e-12
    return float(rms / avg_rect)

def harmonic_ratios(mag: np.ndarray, freqs: np.ndarray, f0: float, kmax: int = 5):
    """
    Return dict of normalized harmonic magnitudes (current-focused).
    Values are magnitude ratios vs the fundamental magnitude.
    """
    out = {}
    fund = float(np.interp(f0, freqs, mag)) + 1e-12
    for k in range(2, kmax + 1):
        fk = k * f0
        out[str(int(fk))] = float(np.interp(fk, freqs, mag) / fund)
    return out