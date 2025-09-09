import numpy as np

def sine(f, fs, N, amplitude=1.0, phase=0.0):
    t = np.arange(N) / fs
    return amplitude * np.sin(2*np.pi*f*t + phase)

def vi_test_signals(fs, N, vrms, irms, f0=60.0, phase_deg=30.0, h2_amp=0.0):
    t = np.arange(N) / fs
    Vpk = vrms * np.sqrt(2.0)
    Ipk = irms * np.sqrt(2.0)

    phase_rad = np.deg2rad(phase_deg)

    v = Vpk * np.sin(2*np.pi*f0*t)  # reference: voltage phase = 0
    # CHANGE HERE: use -phase_rad so current *lags* voltage
    i = Ipk * np.sin(2*np.pi*f0*t - phase_rad)

    # optional 2nd harmonic on current
    i += h2_amp * Ipk * np.sin(2*np.pi*(2*f0)*t - phase_rad)

    return v, i