# src/simulate.py
import time, os, json
import numpy as np
from . import dsp, signals, telemetry

# ---- Spec constants (match your team doc) ----
FS = 4096            # Hz
N  = 1024            # samples per frame
F0 = 60.0            # line freq (Hz)
WINDOW_NAME = "hann"
FRAME_PERIOD = 0.25  # seconds (~4 fps)

# ---- Device pattern (OFF↔ON) ----
ON_DURATION   = 5.0     # s
OFF_DURATION  = 5.0     # s
I_RMS_OFF     = 3.0     # A baseline
I_RMS_ON      = 6.0     # A when device ON
PHASE_DEG     = 30.0    # desired lag angle (current lags voltage by +30°)
H2_OFF        = 0.05    # 5% distortion (OFF)
H2_ON         = 0.12    # 12% distortion (ON)
V_RMS_TARGET  = 120.0   # Vrms

# ---- Output log (NDJSON) ----
LOG_PATH = "data/fixtures/sim.ndjson"

def z(x: float, eps: float = 1e-9) -> float:
    """Zero-out tiny values to avoid -0.0 and numerical fuzz."""
    return 0.0 if abs(x) < eps else x

def emit_frame(vrms_target: float, irms_target: float, h2_amp: float, frame_id: int,
               prev_irms: float | None, prev_p: float | None, state: str):
    # --- Generate V & I for one frame ---
    # IMPORTANT: pass -PHASE_DEG so current *lags* voltage by +PHASE_DEG
    v, i = signals.vi_test_signals(
        fs=FS, N=N, vrms=vrms_target, irms=irms_target,
        f0=F0, phase_deg=-PHASE_DEG, h2_amp=h2_amp
    )

    # --- Spectrum/THD from current (windowed) ---
    window = dsp.hann(N)
    freqs, mag_i = dsp.fft_mag(i * window, FS)

    # --- Core time-domain power metrics ---
    Vrms = dsp.rms(v)
    Irms = dsp.rms(i)
    P    = dsp.real_power(v, i)                 # mean(v*i)
    S    = dsp.apparent_power(Vrms, Irms)       # Vrms*Irms
    PF   = dsp.power_factor(P, S)               # true PF (includes distortion)

    # --- Fundamental (phasor) metrics for complex power (displacement PF) ---
    v1_rms, v1_phase = dsp.dft_phasor(v, FS, F0)
    i1_rms, i1_phase = dsp.dft_phasor(i, FS, F0)
    # Convention: phi = angle(V1) - angle(I1); positive => current lags
    phi = (v1_phase - i1_phase)
    phi_deg = float(np.degrees(phi))
    S1 = float(v1_rms * i1_rms)                 # fundamental apparent power
    P1 = float(S1 * np.cos(phi))                # fundamental real power
    Q1 = float(S1 * np.sin(phi))                # fundamental reactive power
    PF_disp = float(np.cos(phi))                # displacement PF (cos phi)

    # --- Distortion features for ML ---
    THD_i = dsp.thd(mag_i, freqs, f0=F0, harmonics=(2*F0, 3*F0, 4*F0, 5*F0))
    THD_v = 0.01  # voltage is clean in sim; compute like current if you add v FFT
    h_i   = dsp.harmonic_ratios(mag_i, freqs, f0=F0, kmax=5)  # { "120": 0.12, ... } relative to I1

    # odd/even sums based on harmonic order (2nd=even, 3rd=odd, etc.)
    odd_sum_i  = float(sum(v for k, v in h_i.items() if (int(k)//int(F0)) % 2 == 1))
    even_sum_i = float(sum(v for k, v in h_i.items() if (int(k)//int(F0)) % 2 == 0))

    crest_i = dsp.crest_factor(i)
    form_i  = dsp.form_factor(i)

    # --- Timestamp once; use for both frame and any events ---
    now_ms = int(time.time() * 1000)

    # --- Event helpers / deltas ---
    d_irms = float(Irms - (prev_irms if prev_irms is not None else Irms))
    d_p    = float(P    - (prev_p    if prev_p    is not None else P))
    event = None
    if abs(d_irms) > 1.0:
        event = {"type": "on" if d_irms > 0 else "off", "t": now_ms}  # event time == frame time

    # --- Build a rich JSON frame ---
    frame = {
        # Versioning & conventions
        "schema": 1,
        "delta_convention": "current_minus_previous",
        "h_i_units": "relative_to_fundamental",

        # Timing / frame
        "t": now_ms,
        "frame_id": frame_id,
        "fs": FS, "N": N, "window": WINDOW_NAME,
        "freq_hz": round(F0, 3),

        # Core (true) power
        "rms_v": round(Vrms, 3),
        "rms_i": round(Irms, 3),
        "p": round(P, 3),
        "s": round(S, 3),
        "pf_true": round(PF, 3),

        # Fundamental/complex power (displacement PF)
        "v1_rms": round(v1_rms, 3),
        "i1_rms": round(i1_rms, 3),
        "phi_deg": round(phi_deg, 2),
        "phi_convention": "phi = angle(V1) - angle(I1); positive = current lags",
        "p1": round(P1, 3),
        "q1": round(Q1, 3),
        "s1": round(S1, 3),
        "pf_disp": round(PF_disp, 3),

        # Distortion features (current-focused)
        "thd_i": round(THD_i, 3),
        "thd_v": round(THD_v, 3),
        "h_i": {k: round(v, 3) for k, v in h_i.items()},
        "odd_sum_i": round(z(odd_sum_i), 3),
        "even_sum_i": round(z(even_sum_i), 3),
        "crest_i": round(crest_i, 3),
        "form_i": round(form_i, 3),

        # State & events
        "state": state,
        "d_irms": round(z(d_irms), 3),
        "d_p": round(z(d_p), 3),
        "events": ([event] if event else []),

        # Health flags (simple placeholders in sim)
        "fft_ok": True,
        "sync_ok": True,
        "adc_ok": True,

        # Metadata
        "fw": "sandbox-0.2.1",
        "cal_id": "sim-default"
    }

    # NOTE: unified harmonics → keep only `h_i` (relative). Removed legacy `harmonics` map.

    return json.dumps(frame)

def main():
    print("Starting live sandbox… Ctrl+C to stop.")
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    frame_id = 0
    next_tick = time.time()
    state_on = False
    switched_at = time.time()
    prev_irms = None
    prev_p = None
    t_last_event = None

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        while True:
            now = time.time()
            # toggle ON/OFF by durations
            elapsed = now - switched_at
            if state_on and elapsed >= ON_DURATION:
                state_on = False; switched_at = now
            elif (not state_on) and elapsed >= OFF_DURATION:
                state_on = True; switched_at = now

            # targets for this state
            irms = I_RMS_ON if state_on else I_RMS_OFF
            h2   = H2_ON if state_on else H2_OFF
            state = "on" if state_on else "off"

            line = emit_frame(
                vrms_target=V_RMS_TARGET,
                irms_target=irms,
                h2_amp=h2,
                frame_id=frame_id,
                prev_irms=prev_irms,
                prev_p=prev_p,
                state=state
            )
            print(line)
            f.write(line + "\n"); f.flush()

            # update prevs (parse minimal fields without re-computing)
            obj = json.loads(line)
            prev_irms = obj["rms_i"]
            prev_p    = obj["p"]
            if obj["events"]:
                t_last_event = obj["events"][0]["t"]
                obj["t_since_event_ms"] = 0
            elif t_last_event is not None:
                obj["t_since_event_ms"] = int(time.time()*1000) - t_last_event

            frame_id += 1
            next_tick += FRAME_PERIOD
            time.sleep(max(0.0, next_tick - time.time()))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
