import json, time

def pack_frame(vrms, irms, P, S, PF, THD, harmonics=None):
    return json.dumps({
        "t": int(time.time()*1000),
        "rms_v": round(vrms, 3),
        "rms_i": round(irms, 3),
        "p": round(P, 3),
        "s": round(S, 3),
        "pf": round(PF, 3),
        "thd": round(THD, 3),
        "harmonics": {str(k): round(v, 3) for k, v in (harmonics or {}).items()}
    })