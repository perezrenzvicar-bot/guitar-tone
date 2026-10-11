#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================
  GUITAR TONE MATCHER  v3  (made for Pydroid 3)
=====================================================
 * Search songs ONLINE (30-second previews from Deezer
   or iTunes) or use files from your phone.
 * Shows how to set up the M-VAVE CUBE BABY pedal
   (TYPE, GAIN, TONE, MOD, delay, reverb, IR CAB...)
   for the song's guitar tone - RHYTHM or LEAD.
 * Every gear choice is optional (skip anything).
 * Can also apply the tone to your own guitar WAV.

Install in Pydroid 3 > Pip:  numpy  scipy
Online MP3 needs ONE decoder (try in this order):
   miniaudio    or    soundfile    or    av
Local WAV files work with no extra install.
"""

import os
import sys
import math
import time
import wave
import json
import queue
import threading
import tempfile
import urllib.request
import urllib.parse

try:
    import numpy as np
    from scipy import signal
    from scipy.io import wavfile
    from scipy.ndimage import median_filter
except ImportError:
    print("Missing libraries. In Pydroid 3 open the menu")
    print("(three lines) > Pip, then install:")
    print("   numpy   and   scipy")
    sys.exit(1)

try:
    import tkinter as tk
    from tkinter import font as tkfont
except ImportError:
    print("tkinter is missing. Use the official Pydroid 3 app.")
    sys.exit(1)

ASR = 22050  # analysis sample rate
AUDIO_EXT = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

# Typical octave-band energy of a guitar inside a mix (dB).
REF_OCTAVES = np.array([-2.0, 0.0, 0.0, -2.5, -6.0, -10.5, -16.0])
OCT_EDGES = [80, 160, 320, 640, 1280, 2560, 5120, 10000]


# ----------------------------------------------------
#  small helpers
# ----------------------------------------------------
def clip(v, lo=0.0, hi=10.0):
    return float(min(hi, max(lo, v)))


def scale01(v, lo, hi):
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


def bar(v, vmax=10.0, width=10):
    n = int(round(width * clip(v, 0, vmax) / vmax))
    return "[" + "#" * n + "." * (width - n) + "]"


def fmt_time(sec):
    sec = int(sec)
    return "%02d:%02d" % (sec // 60, sec % 60)


def parse_time(text, default):
    text = text.strip()
    if not text:
        return default
    try:
        if ":" in text:
            m, s = text.split(":")
            return int(m) * 60 + float(s)
        return float(text)
    except ValueError:
        return default


# ----------------------------------------------------
#  audio loading (WAV always; MP3/M4A need a decoder)
# ----------------------------------------------------
def _read_wav_24bit(path):
    with wave.open(path, "rb") as w:
        ch, sw, sr, n = (w.getnchannels(), w.getsampwidth(),
                         w.getframerate(), w.getnframes())
        raw = w.readframes(n)
    if sw != 3:
        raise ValueError("unsupported WAV format")
    a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
    v = a[:, 0] | (a[:, 1] << 8) | (a[:, 2] << 16)
    v = np.where(v >= 2 ** 23, v - 2 ** 24, v).astype(np.float32) / 2 ** 23
    return v.reshape(-1, ch), sr


def _to_float(d):
    if d.dtype == np.int16:
        d = d.astype(np.float32) / 32768.0
    elif d.dtype == np.int32:
        d = d.astype(np.float32) / 2147483648.0
    elif d.dtype == np.uint8:
        d = (d.astype(np.float32) - 128.0) / 128.0
    else:
        d = d.astype(np.float32)
    if d.ndim == 1:
        d = d[:, None]
    return d


def _dec_wav(path):
    try:
        sr, d = wavfile.read(path)
        return _to_float(d), sr
    except Exception:
        return _read_wav_24bit(path)


def _dec_miniaudio(path):
    import miniaudio
    d = miniaudio.decode_file(path)
    arr = np.array(d.samples, dtype=np.int16).reshape(-1, d.nchannels)
    return arr.astype(np.float32) / 32768.0, d.sample_rate


def _dec_soundfile(path):
    import soundfile as sf
    d, sr = sf.read(path, always_2d=True, dtype="float32")
    return d, sr


def _dec_av(path):
    import av
    cont = av.open(path)
    st = cont.streams.audio[0]
    res = av.AudioResampler(format="s16", layout="stereo", rate=44100)
    chunks = []

    def grab(frames):
        if frames is None:
            return
        if not isinstance(frames, (list, tuple)):
            frames = [frames]
        for fr in frames:
            chunks.append(fr.to_ndarray().reshape(-1))

    for frame in cont.decode(st):
        grab(res.resample(frame))
    try:
        grab(res.resample(None))
    except Exception:
        pass
    cont.close()
    arr = np.concatenate(chunks).astype(np.int16).reshape(-1, 2)
    return arr.astype(np.float32) / 32768.0, 44100


def _dec_pydub(path):
    from pydub import AudioSegment
    seg = AudioSegment.from_file(path)
    d = np.array(seg.get_array_of_samples()).reshape(-1, seg.channels)
    d = d.astype(np.float32) / float(2 ** (8 * seg.sample_width - 1))
    return d, seg.frame_rate


def decoder_status():
    """Returns (summary text, mp3_ok, m4a_ok)."""
    have = []
    for name in ("miniaudio", "soundfile", "av", "pydub"):
        try:
            mod = __import__(name)
            if name == "soundfile":
                if "MP3" not in mod.available_formats():
                    continue
            have.append(name)
        except Exception:
            pass
    mp3 = bool(have)
    m4a = "av" in have or "pydub" in have
    txt = "WAV: yes | MP3: %s | M4A: %s" % (
        "yes" if mp3 else "NO", "yes" if m4a else "no")
    if have:
        txt += "  (" + ", ".join(have) + ")"
    else:
        txt += "\nFor online songs install 'miniaudio' (Pip menu)."
    return txt, mp3, m4a


def load_audio(path):
    """Return (data[n, channels] float32 in -1..1, samplerate)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        return _dec_wav(path)
    errors = []
    for fn in (_dec_miniaudio, _dec_soundfile, _dec_av, _dec_pydub):
        try:
            d, sr = fn(path)
            if d is not None and len(d) > 0:
                return d.astype(np.float32), sr
        except ImportError:
            continue
        except Exception as ex:
            errors.append("%s: %s" % (fn.__name__[5:], ex))
    msg = ("Can't decode %s files on this phone yet.\n"
           "Install 'miniaudio' (Pydroid menu > Pip) and try again, "
           "or use a WAV file." % ext)
    if errors:
        msg += "\n(" + "; ".join(errors)[:120] + ")"
    raise RuntimeError(msg)


def resample(y, sr_in, sr_out):
    if sr_in == sr_out:
        return y
    g = math.gcd(int(sr_in), int(sr_out))
    return signal.resample_poly(y, int(sr_out) // g, int(sr_in) // g)


# ----------------------------------------------------
#  ANALYSIS
# ----------------------------------------------------
def third_octave_profile(freqs, pmean):
    """Mean-removed third-octave profile (dB) 125 Hz..~5 kHz."""
    centers = 125.0 * 2.0 ** (np.arange(0, 17) / 3.0)
    out = []
    for c in centers:
        lo, hi = c * 2 ** (-1 / 6.0), c * 2 ** (1 / 6.0)
        idx = np.where((freqs >= lo) & (freqs <= hi))[0]
        if len(idx) == 0:
            idx = [int(np.argmin(np.abs(freqs - c)))]
        out.append(10 * np.log10(np.mean(pmean[idx]) + 1e-12))
    out = np.array(out)
    return centers, out - out.mean()


def estimate_tempo(y, sr):
    f, t, Z = signal.stft(y, sr, nperseg=1024, noverlap=768)
    P = np.log1p(np.abs(Z) * 10.0)
    flux = np.maximum(0, np.diff(P, axis=1)).sum(axis=0)
    flux = flux - flux.mean()
    n = len(flux)
    if n < 64:
        return 120.0
    spec = np.fft.rfft(flux, 2 * n)
    ac = np.fft.irfft(spec * np.conj(spec))[:n]
    hop = 256.0 / sr
    lo, hi = int(60.0 / 180.0 / hop), int(60.0 / 70.0 / hop)
    hi = min(hi, n - 2)
    lags = np.arange(lo, hi)
    bpms = 60.0 / (lags * hop)
    w = np.exp(-0.5 * (np.log2(bpms / 115.0) / 0.9) ** 2)
    sc = ac[lo:hi] * w
    k = int(np.argmax(sc))
    lag = lags[k]
    if 0 < k < len(sc) - 1:  # parabolic refinement
        a, b, c = sc[k - 1], sc[k], sc[k + 1]
        den = a - 2 * b + c
        if den != 0:
            lag = lag + 0.5 * (a - c) / den
    bpm = 60.0 / (lag * hop)
    while bpm < 80:
        bpm *= 2
    while bpm > 165:
        bpm /= 2
    return float(bpm)


def analyze_song(path, start, dur, log=print):
    log("Loading audio...")
    data, sr = load_audio(path)
    total = len(data) / float(sr)
    if start >= total - 3:
        start = max(0.0, total - dur)
    a, b = int(start * sr), int(min(total, start + dur) * sr)
    seg = data[a:b]
    if len(seg) < sr * 3:
        raise RuntimeError("Section too short (need at least 3 seconds).")
    chans = [resample(seg[:, c], sr, ASR) for c in range(seg.shape[1])]
    mono = np.mean(chans, axis=0).astype(np.float32)
    mono /= (np.max(np.abs(mono)) + 1e-9)

    log("Isolating guitar-ish (harmonic) content...")
    f, t, Z = signal.stft(mono, ASR, nperseg=2048, noverlap=1024)
    mag = np.abs(Z).astype(np.float32)
    Hm = median_filter(mag, size=(1, 15), mode="nearest")
    Pm = median_filter(mag, size=(15, 1), mode="nearest")
    mask = Hm ** 2 / (Hm ** 2 + Pm ** 2 + 1e-12)
    Zh = Z * mask
    _, yh = signal.istft(Zh, ASR, nperseg=2048, noverlap=1024)
    Hmag = np.abs(Zh)

    log("Measuring tone...")
    gb = (f >= 80) & (f <= 8000)
    e = (Hmag[gb] ** 2).sum(axis=0)
    active = e > np.percentile(e, 40)
    if active.sum() < 5:
        active = np.ones_like(e, dtype=bool)
    P = Hmag ** 2

    # spectral flatness 1k-6k (dense harmonics => distortion)
    fb = (f >= 1000) & (f <= 6000)
    pw = P[fb][:, active] + 1e-12
    fl = np.exp(np.mean(np.log(pw), axis=0)) / np.mean(pw, axis=0)
    flat = float(10 * np.log10(np.median(fl) + 1e-12))  # dB, higher = denser

    # octave energy deviations
    pmean = P[:, active].mean(axis=1)
    lv = []
    for lo, hi in zip(OCT_EDGES[:-1], OCT_EDGES[1:]):
        m = (f >= lo) & (f < hi)
        lv.append(10 * np.log10(pmean[m].sum() + 1e-12))
    dev = np.array(lv) - REF_OCTAVES
    dev -= dev.mean()
    d_bass = float(np.mean(dev[0:2]))
    d_mid = float(np.mean(dev[2:4]))
    d_treb = float(np.mean(dev[4:6]))
    d_pres = float(np.mean(dev[5:7]))

    # brightness
    ps = pmean[gb]
    centroid = float(np.sum(f[gb] * ps) / (np.sum(ps) + 1e-12))

    # crest factor of harmonic guitar band
    sos = signal.butter(2, [80, 5000], "band", fs=ASR, output="sos")
    yb = signal.sosfilt(sos, yh)
    blk = 1024
    nb = len(yb) // blk
    bl = yb[:nb * blk].reshape(nb, blk)
    rms = np.sqrt((bl ** 2).mean(axis=1)) + 1e-9
    pk = np.abs(bl).max(axis=1)
    act = rms > np.percentile(rms, 40)
    crest = float(np.median(20 * np.log10(pk[act] / rms[act])))

    # sustain / ambience from decay rate
    blk2 = 512
    nb2 = len(yb) // blk2
    r2 = np.sqrt((yb[:nb2 * blk2].reshape(nb2, blk2) ** 2).mean(axis=1)) + 1e-9
    env = 20 * np.log10(r2)
    d = np.diff(env) / (blk2 / float(ASR))
    ok = (d < 0) & (env[:-1] > np.percentile(env, 40))
    rate = float(np.percentile(-d[ok], 40)) if ok.sum() > 10 else 25.0
    ambience = float(np.clip((35.0 - rate) / 30.0, 0, 1))

    bpm = estimate_tempo(mono, ASR)

    width = 0.0
    if len(chans) >= 2:
        sosw = signal.butter(2, [200, 4000], "band", fs=ASR, output="sos")
        L = signal.sosfilt(sosw, chans[0])
        R = signal.sosfilt(sosw, chans[1])
        M, S = (L + R) / 2, (L - R) / 2
        width = float(np.sqrt(np.mean(S ** 2)) / (np.sqrt(np.mean(M ** 2)) + 1e-9))

    centers, prof = third_octave_profile(f, pmean)
    return dict(file=os.path.basename(path), start=start,
                end=min(total, start + dur), flat=flat, crest=crest,
                d_bass=d_bass, d_mid=d_mid, d_treb=d_treb, d_pres=d_pres,
                centroid=centroid, ambience=ambience, decay_rate=rate,
                bpm=bpm, width=width, prof_centers=centers, prof=prof)


# ----------------------------------------------------
#  TONE SETTINGS  (matched tone + RHYTHM / LEAD tailoring)
# ----------------------------------------------------
def amp_name(gain):
    if gain < 2.5:
        return "Clean - Fender-style (sparkly)"
    if gain < 5.0:
        return "Crunch - Vox/Plexi-style"
    if gain < 7.5:
        return "Hot Rod - JCM/Plexi cranked"
    return "High gain - Mesa/5150-style"


def make_settings(f, mode):
    """mode: 'rhythm' or 'lead'. Returns dict of settings."""
    g = 10 * (0.5 * scale01(f["flat"], -50.0, -18.0) +
              0.5 * scale01(17.0 - f["crest"], 0.0, 8.0))
    bass = 5 + f["d_bass"] / 2.5
    mid = 5 + f["d_mid"] / 2.5
    treb = 5 + f["d_treb"] / 2.5
    pres = 5 + f["d_pres"] / 2.5
    reverb = 5 + 25 * f["ambience"]           # % mix
    comp = 2 + 5 * scale01(f["crest"], 9, 18) * (1 - g / 12.0)
    delay_mix, notes, pedals = 0.0, [], []

    quarter = 60000.0 / f["bpm"]
    dotted8 = quarter * 0.75

    if mode == "rhythm":
        g = g * 0.88 - 0.4
        bass -= 0.6 if g > 5.5 else 0.0
        mid -= 0.6
        treb += 0.3
        pres += 0.5
        reverb *= 0.6
        comp = max(comp - 1, 0)
        gate = 3 + 0.45 * g
        delay_ms = quarter / 2
        if g > 6.5:
            pedals.append("Boost (Tube-Screamer type): drive 0,")
            pedals.append("  tone 6, level 7 -> tightens lows")
        if f["width"] > 0.35:
            pedals.append("Double-track / light chorus (mix 20%)")
        pickup = ("Bridge humbucker" if g > 5 else
                  "Middle/Bridge single-coil" if f["centroid"] > 2200
                  else "Neck/Middle pickup")
        notes = ["Palm-mute chugs close to the bridge",
                 "Pick hard, play tight & on the beat",
                 "Keep volume knob at 9-10 for chords"]
    else:  # lead
        g = g + 1.5
        bass -= 1.0
        mid += 1.2
        pres -= 0.2
        reverb = reverb * 1.3 + 1
        comp = comp + 2
        gate = 2 + 0.3 * g
        delay_mix = 20.0
        delay_ms = dotted8
        if g < 6:
            pedals.append("Overdrive/boost in front: drive 4,")
            pedals.append("  tone 5, level 7 -> more sustain")
        pedals.append("Delay: %d ms (dotted 1/8), mix 20%%" % round(dotted8))
        if f["width"] > 0.35:
            pedals.append("Subtle chorus for width (depth 3)")
        pickup = ("Neck humbucker (smooth, singing)"
                  if f["centroid"] < 1900 else "Bridge pickup (cutting)")
        notes = ["Use vibrato & bends to make notes sing",
                 "Roll tone knob to ~7 for a thicker lead",
                 "Step on boost/delay only for solos"]

    if comp >= 4:
        pedals.append("Compressor: sustain 5, level 6")
    g = clip(g)
    return dict(mode=mode, gain=g, bass=clip(bass), mid=clip(mid),
                treble=clip(treb), presence=clip(pres),
                reverb=clip(reverb, 0, 60), delay_mix=delay_mix,
                delay_ms=float(delay_ms), comp=clip(comp), gate=clip(gate),
                amp=amp_name(g), pickup=pickup, pedals=pedals,
                notes=notes, bpm=f["bpm"])


def brightness_label(c):
    if c < 1500:
        return "Dark/warm"
    if c < 2200:
        return "Balanced"
    if c < 3000:
        return "Bright"
    return "Very bright"


def build_report(f, s):
    L = []
    w = 44
    L.append("=" * w)
    L.append("  TONE MATCH  >>  %s TONE" % s["mode"].upper())
    L.append("=" * w)
    L.append(" Song    : %s" % f["file"][:32])
    L.append(" Section : %s - %s" % (fmt_time(f["start"]), fmt_time(f["end"])))
    L.append(" Tempo   : ~%d BPM" % round(f["bpm"]))
    L.append(" Color   : %s (%d Hz)" % (brightness_label(f["centroid"]),
                                         round(f["centroid"])))
    L.append("-" * w)
    L.append(" AMP : %s" % s["amp"])
    L.append(" PICKUP: %s" % s["pickup"])
    L.append("-" * w)
    for name, key in (("Gain    ", "gain"), ("Bass    ", "bass"),
                      ("Mid     ", "mid"), ("Treble  ", "treble"),
                      ("Presence", "presence")):
        L.append(" %s %s %4.1f" % (name, bar(s[key]), s[key]))
    L.append("-" * w)
    L.append(" Reverb  %s %3d%%" % (bar(s["reverb"], 60), round(s["reverb"])))
    if s["delay_mix"] > 0:
        L.append(" Delay   %s %3d%% @%dms" % (bar(s["delay_mix"], 60),
                                              round(s["delay_mix"]),
                                              round(s["delay_ms"])))
    else:
        L.append(" Delay   off")
    L.append(" Comp    %s %4.1f" % (bar(s["comp"]), s["comp"]))
    L.append(" N-Gate  %s %4.1f" % (bar(s["gate"]), s["gate"]))
    if s["pedals"]:
        L.append("-" * w)
        L.append(" PEDALBOARD:")
        for p in s["pedals"]:
            L.append("  * " + p if not p.startswith("  ") else "   " + p.strip())
    L.append("-" * w)
    L.append(" TIPS:")
    for n in s["notes"]:
        L.append("  - " + n)
    L.append("=" * w)
    L.append(" Estimates from audio analysis. Fine-tune")
    L.append(" by ear - every guitar/amp is different!")
    return "\n".join(L)


# ----------------------------------------------------
#  DSP: amp simulator for YOUR guitar recording
# ----------------------------------------------------
def biquad_sos(kind, f0, gain_db, Q, fs):
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * f0 / fs
    al = math.sin(w0) / (2 * Q)
    cw = math.cos(w0)
    sA = math.sqrt(A)
    if kind == "peak":
        b0, b1, b2 = 1 + al * A, -2 * cw, 1 - al * A
        a0, a1, a2 = 1 + al / A, -2 * cw, 1 - al / A
    elif kind == "low":
        b0 = A * ((A + 1) - (A - 1) * cw + 2 * sA * al)
        b1 = 2 * A * ((A - 1) - (A + 1) * cw)
        b2 = A * ((A + 1) - (A - 1) * cw - 2 * sA * al)
        a0 = (A + 1) + (A - 1) * cw + 2 * sA * al
        a1 = -2 * ((A - 1) + (A + 1) * cw)
        a2 = (A + 1) + (A - 1) * cw - 2 * sA * al
    else:  # high shelf
        b0 = A * ((A + 1) + (A - 1) * cw + 2 * sA * al)
        b1 = -2 * A * ((A - 1) + (A + 1) * cw)
        b2 = A * ((A + 1) + (A - 1) * cw - 2 * sA * al)
        a0 = (A + 1) - (A - 1) * cw + 2 * sA * al
        a1 = 2 * ((A - 1) - (A + 1) * cw)
        a2 = (A + 1) - (A - 1) * cw - 2 * sA * al
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def smooth_env(x, a):
    return signal.lfilter([a], [1, -(1 - a)], x)


def block_rms_db(x, blk):
    n = int(math.ceil(len(x) / float(blk)))
    xp = np.zeros(n * blk)
    xp[:len(x)] = x
    r = np.sqrt((xp.reshape(n, blk) ** 2).mean(axis=1)) + 1e-9
    centers = (np.arange(n) + 0.5) * blk
    return 20 * np.log10(r), centers


def compressor(x, sr, amount):
    if amount < 0.5:
        return x
    blk = max(1, int(sr * 0.005))
    env, ctr = block_rms_db(x, blk)
    thr = np.max(env) - 14 - 1.2 * amount
    ratio = 1 + amount * 0.45
    gr = np.minimum(0.0, (thr - env)) * (1 - 1 / ratio)
    gr = smooth_env(gr, 0.08)
    makeup = -(thr - np.max(env)) * (1 - 1 / ratio) * 0.5
    g = 10 ** ((gr + makeup) / 20.0)
    return x * np.interp(np.arange(len(x)), ctr, g)


def noise_gate(x, sr, amount):
    if amount < 0.5:
        return x
    blk = max(1, int(sr * 0.005))
    env, ctr = block_rms_db(x, blk)
    thr = np.max(env) - (62 - amount * 3.2)
    opn = smooth_env((env > thr).astype(float), 0.15)
    return x * np.interp(np.arange(len(x)), ctr, opn)


def distort(x, sr, gain):
    g = gain / 10.0
    drive = 1.0 + 70.0 * g ** 2.2
    bias = 0.12 * g
    up = signal.resample_poly(x, 2, 1)
    y = np.tanh(drive * up + bias) - np.tanh(bias)
    if gain > 5:  # second gain stage with interstage tightening
        sos = signal.butter(1, 130, "high", fs=2 * sr, output="sos")
        y = signal.sosfilt(sos, y)
        y = np.tanh((1.0 + 2.5 * (g - 0.5)) * y)
    y = signal.resample_poly(y, 1, 2)
    return y[:len(x)]


def reverb_ir(sr, seconds=1.6):
    n = int(sr * seconds)
    t = np.arange(n) / float(sr)
    rng = np.random.RandomState(7)
    ir = rng.standard_normal(n) * np.exp(-6.9 * t / seconds)
    sos = signal.butter(1, 4200, "low", fs=sr, output="sos")
    ir = signal.sosfilt(sos, ir)
    ir = np.concatenate([np.zeros(int(0.012 * sr)), ir])
    return ir / (np.sqrt(np.sum(ir ** 2)) + 1e-9)


def match_eq(y, sr, target_centers, target_prof, strength=0.5):
    if sr / 2.0 <= 7000:
        return y
    f, t, Z = signal.stft(y, sr, nperseg=4096)
    P = np.abs(Z) ** 2
    e = P.sum(axis=0)
    act = e > np.percentile(e, 50)
    if act.sum() < 3:
        return y
    pm = P[:, act].mean(axis=1)
    _, prof = third_octave_profile(f, pm)
    corr = np.clip((target_prof - prof) * strength, -6, 6)
    corr = np.convolve(np.pad(corr, 1, mode="edge"),
                       [0.25, 0.5, 0.25], mode="valid")
    nyq = sr / 2.0
    fr = np.concatenate([[0, 100], target_centers, [6500, nyq]])
    gn = np.concatenate([[1, 1], 10 ** (corr / 20.0), [1, 1]])
    h = signal.firwin2(1025, fr / nyq, gn)
    return signal.fftconvolve(y, h, mode="same")


def process_guitar(x, sr, s, feat, use_match=True, log=print):
    x = x.astype(np.float64)
    x -= np.mean(x)
    x /= (np.max(np.abs(x)) + 1e-9)
    x *= 0.35

    log("  low-cut / tightening...")
    hp = 60 + 12 * s["gain"] + (20 if s["mode"] == "rhythm" else 0)
    x = signal.sosfilt(signal.butter(2, hp, "high", fs=sr, output="sos"), x)

    log("  compressor & gate...")
    x = compressor(x, sr, s["comp"])
    x = noise_gate(x, sr, s["gate"])

    log("  amp gain stage...")
    x = distort(x, sr, s["gain"])
    x /= (np.max(np.abs(x)) + 1e-9)

    log("  tone stack (bass/mid/treble/presence)...")
    sos = np.vstack([
        biquad_sos("low", 120, (s["bass"] - 5) * 2.6, 0.7, sr),
        biquad_sos("peak", 650, (s["mid"] - 5) * 2.6, 0.8, sr),
        biquad_sos("high", 3500, (s["treble"] - 5) * 2.6, 0.7, sr),
        biquad_sos("high", 5800, (s["presence"] - 5) * 2.6, 0.7, sr),
    ])
    x = signal.sosfilt(sos, x)

    log("  speaker cabinet sim...")
    fc = min(sr / 2.0 * 0.45, 6200 - 160 * s["gain"])
    cab = np.vstack([
        signal.butter(4, fc, "low", fs=sr, output="sos"),
        signal.butter(2, 75, "high", fs=sr, output="sos"),
        biquad_sos("peak", 2300, 2.0, 1.0, sr),
    ])
    x = signal.sosfilt(cab, x)

    if use_match:
        log("  auto match-EQ toward the song...")
        x = match_eq(x, sr, feat["prof_centers"], feat["prof"], 0.5)

    x /= (np.max(np.abs(x)) + 1e-9)

    if s["delay_mix"] > 0 or s["reverb"] > 1:
        x = np.concatenate([x, np.zeros(int(2.5 * sr))])
    if s["delay_mix"] > 0:
        log("  delay...")
        d = max(1, int(s["delay_ms"] / 1000.0 * sr))
        ir = np.zeros(d * 6 + 1)
        ir[0] = 1.0
        for k in range(1, 7):
            ir[k * d] = (s["delay_mix"] / 100.0) * (0.38 ** (k - 1))
        x = signal.fftconvolve(x, ir)[:len(x)]
    if s["reverb"] > 1:
        log("  reverb...")
        ir = reverb_ir(sr)
        wet = signal.fftconvolve(x, ir)[:len(x)]
        x = x + wet * (s["reverb"] / 100.0) * 1.4

    x = np.tanh(x * 0.9 / (np.max(np.abs(x)) + 1e-9) * 1.1)  # soft limit
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.89
    return x.astype(np.float32)


# ----------------------------------------------------
#  ONLINE SONGS (30-second previews)
# ----------------------------------------------------
UA = "Mozilla/5.0 (Linux; Android 12) ToneMatcher/3.0"


def http_get(url, timeout=25, limit=40 * 1024 * 1024):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    buf, total = [], 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        while True:
            chunk = r.read(65536)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise RuntimeError("File is too large.")
            buf.append(chunk)
    return b"".join(buf)


def search_online(query, source):
    """Return a list of dicts: kind,title,artist,url,ext."""
    q = urllib.parse.quote(query.strip())
    out = []
    if source == "itunes":
        url = ("https://itunes.apple.com/search?term=%s&media=music"
               "&entity=song&limit=25" % q)
        data = json.loads(http_get(url).decode("utf-8", "replace"))
        for r in data.get("results", []):
            if r.get("previewUrl"):
                out.append(dict(kind="online",
                                title=r.get("trackName", "?"),
                                artist=r.get("artistName", "?"),
                                url=r["previewUrl"], ext=".m4a"))
    else:
        url = "https://api.deezer.com/search?q=%s&limit=25" % q
        data = json.loads(http_get(url).decode("utf-8", "replace"))
        for r in data.get("data", []):
            if r.get("preview"):
                out.append(dict(kind="online",
                                title=r.get("title", "?"),
                                artist=(r.get("artist") or {}).get("name", "?"),
                                url=r["preview"], ext=".mp3"))
    return out


def cache_dir():
    bases = []
    try:
        bases.append(os.path.dirname(os.path.abspath(__file__)))
    except Exception:
        pass
    bases += [os.getcwd(), tempfile.gettempdir()]
    for b in bases:
        p = os.path.join(b, "tone_cache")
        try:
            os.makedirs(p, exist_ok=True)
            t = os.path.join(p, ".t")
            with open(t, "w") as fh:
                fh.write("x")
            os.remove(t)
            return p
        except Exception:
            continue
    return tempfile.gettempdir()


def safe_name(s):
    keep = "".join(c if c.isalnum() else "_" for c in s)[:50]
    return keep or "song"


def download_audio(src):
    ext = src.get("ext") or os.path.splitext(
        urllib.parse.urlparse(src["url"]).path)[1] or ".mp3"
    name = safe_name("%s_%s" % (src.get("artist", ""), src.get("title", "")))
    path = os.path.join(cache_dir(), name + ext)
    if not os.path.isfile(path) or os.path.getsize(path) < 1000:
        data = http_get(src["url"])
        with open(path, "wb") as fh:
            fh.write(data)
    return path


# ----------------------------------------------------
#  YOUR GEAR  (everything optional)
# ----------------------------------------------------
GUITAR_SKIP = "Skip / not sure"
GUITARS = {
    GUITAR_SKIP: dict(kind="any", bass=0.0, mid=0.0, treb=0.0, gain=0.0),
    "Strat-style (single-coil)": dict(kind="sc", bass=0.3, mid=0.2,
                                      treb=-0.8, gain=0.7),
    "Telecaster-style": dict(kind="tele", bass=0.4, mid=0.0,
                             treb=-1.0, gain=0.8),
    "Les Paul-style (humbucker)": dict(kind="hb", bass=-0.3, mid=0.3,
                                       treb=0.6, gain=-0.5),
    "Superstrat / high-output HB": dict(kind="ss", bass=-0.3, mid=0.0,
                                        treb=0.2, gain=-1.0),
    "Semi-hollow (335-style)": dict(kind="semi", bass=-0.4, mid=0.0,
                                    treb=0.3, gain=-0.3),
    "P90 guitar": dict(kind="p90", bass=0.0, mid=0.3, treb=-0.2, gain=0.2),
}
TONE_KNOB = {"any": (10, 8), "sc": (10, 9), "tele": (10, 9), "hb": (10, 7),
             "ss": (10, 8), "semi": (9, 7), "p90": (10, 8)}

PEDAL_CUBE = "cube"
PEDAL_NONE = "none"

OUT_SKIP = "Skip / not sure"
OUTPUTS = {
    OUT_SKIP: dict(cab=True, tip="Using headphones, PA or an interface? "
                                 "Keep IR CAB on."),
    "Headphones / phone / audio interface": dict(
        cab=True, tip="Headphones, phone or interface: IR CAB must be ON, "
                      "it replaces the speaker."),
    "Guitar amp (clean channel)": dict(
        cab=False, tip="Set the amp CLEAN, EQ flat, amp reverb off. Plug "
                       "Cube Baby OUT into the amp input."),
    "PA / powered speaker / mixer": dict(
        cab=True, tip="PA or mixer: keep IR CAB on and set VOLUME so the "
                      "channel doesn't clip."),
}

# From the M-VAVE Cube Baby manual: 9 preamp TYPE slots.
CUBE_TYPES = {
    1: ("Power-Zone Clean", "Clean"),
    2: ("US Gold 100 Clean", "Clean"),
    3: ("Two Stone Coral OD", "Overdrive"),
    4: ("Doctor3 B", "Overdrive"),
    5: ("Cali JP A", "Overdrive"),
    6: ("Day Tripper OD", "Distortion"),
    7: ("Shittcow Dist", "Distortion"),
    8: ("Wo Stone Coral OD", "Distortion"),
    9: ("Mr Smith Dist", "Distortion"),
}
# (need_lo, need_hi, slot)
CUBE_BANDS = [(0.0, 2.0, 2), (2.0, 3.6, 3), (3.6, 5.0, 4), (5.0, 6.4, 5),
              (6.4, 7.3, 6), (7.3, 8.2, 7), (8.2, 9.0, 8), (9.0, 10.01, 9)]


def cube_type(need):
    for lo, hi, slot in CUBE_BANDS:
        if need < hi:
            if slot == 2:                       # clean: gain acts as level
                return slot, clip(6.0 + need * 1.5)
            return slot, clip(2.5 + 7.0 * (need - lo) / (hi - lo), 1, 10)
    return 9, 10.0


def clock(v):
    """Knob value 0-10 -> 'o'clock' text (7 o'clock = min, 5 = max)."""
    r = round((7.0 + v) * 2) / 2.0
    if r >= 13:
        r -= 12.0
    hh = int(r)
    return ("%d o'clock" % hh) if r == hh else ("%d:30" % hh)


def pickup_advice(kind, mode, gain, centroid, fallback):
    dark = centroid < 1900
    if kind == "sc":
        if mode == "rhythm":
            return "Bridge (pos 1)" if gain > 5 else "Middle combos (pos 2/4)"
        return "Neck (pos 5)" if dark else "Bridge (pos 1)"
    if kind == "tele":
        if mode == "rhythm":
            return "Bridge" if gain > 3 else "Bridge + a little tone roll-off"
        return "Neck" if dark else "Bridge"
    if kind == "hb":
        if mode == "rhythm":
            return "Bridge" if gain > 5 else "Neck (warm clean)"
        return "Neck (smooth, singing)" if dark else "Bridge (cutting)"
    if kind == "ss":
        return "Bridge" if (mode == "rhythm" or not dark) else "Neck"
    if kind == "semi":
        return ("Bridge (crunch)" if (mode == "rhythm" and gain > 4)
                else "Neck")
    if kind == "p90":
        return "Bridge" if (mode == "rhythm" or not dark) else "Neck"
    return fallback


def _clean_notes(s):
    return [n for n in s["notes"] if not n.startswith("Roll tone")]


def adapt_cube(s, feat, gear):
    """Map the ideal tone onto the M-VAVE Cube Baby's real controls."""
    g = GUITARS.get(gear.get("guitar"), GUITARS[GUITAR_SKIP])
    o = OUTPUTS.get(gear.get("output"), OUTPUTS[OUT_SKIP])
    mode = s["mode"]
    bass = s["bass"] + g["bass"]
    mid = s["mid"] + g["mid"]
    treb = s["treble"] + g["treb"]
    pres = s["presence"]
    need = clip(s["gain"] + g["gain"])
    fit = 100.0
    skipped, tips, warn = [], [], []

    slot, gain_knob = cube_type(need)
    tname, tcat = CUBE_TYPES[slot]
    tone = clip(5 + 0.55 * (treb - 5) + 0.35 * (pres - 5) - 0.2 * (bass - 5))

    # ---- IR cabinet (shapes the lows/mids since there is no EQ)
    if not o["cab"]:
        ir_slot = 1
        ir_line = "IR CAB OFF (first slot): your amp's speaker is the cab."
    elif need < 3.5:
        ir_slot = 5
        ir_line = ("Slot 5 (VHT Deliverance 2x12), or a Fender Deluxe-"
                   "style IR if you loaded one.")
    elif mid < 4.3 and need >= 7.5:
        ir_slot = 4
        ir_line = ("Slot 4 (Marshall 1960A 4x12). For a scooped modern "
                   "sound try a Mesa 4x12 IR if installed.")
    elif mid > 6.3 or mode == "lead":
        ir_slot = 5
        ir_line = "Slot 5 (VHT Deliverance 2x12): focused mids, great for lead."
    else:
        ir_slot = 4
        ir_line = "Slot 4 (Marshall 1960A 4x12): full lows, classic rock."
    ir_val = (ir_slot - 1) / 8.0 * 10

    # ---- MOD (chorus/phaser)
    mod_on = feat.get("width", 0) > 0.35
    mod_v = 3.0 if mod_on else 0.0

    # ---- delay
    delay_on = s["delay_mix"] > 0
    if delay_on:
        t_v = clip(10.0 * (s["delay_ms"] - 50.0) / 750.0)
        fb_v = 3.0
        mix_v = clip(s["delay_mix"] / 6.0)
    else:
        t_v = fb_v = mix_v = 0.0

    # ---- reverb
    rv = s["reverb"]
    rev_v = clip(rv / 7.0) if rv >= 3 else 0.0

    vol_v = 6.5

    def cap(v):
        return "OFF" if v <= 0.05 else clock(v)

    knobs = [("TYPE", (slot - 1) / 8.0 * 10, "slot %d" % slot),
             ("GAIN", gain_knob, clock(gain_knob)),
             ("TONE", tone, clock(tone)),
             ("MOD", mod_v, cap(mod_v)),
             ("TIME", t_v, cap(t_v)),
             ("FB", fb_v, cap(fb_v)),
             ("MIX", mix_v, cap(mix_v)),
             ("REVERB", rev_v, cap(rev_v)),
             ("IR CAB", ir_val, "OFF" if ir_slot == 1 else "slot %d" % ir_slot),
             ("VOLUME", vol_v, clock(vol_v))]

    switches = [("Footswitch A (IR CAB + REVERB)",
                 "ON" if (o["cab"] or rev_v > 0) else "OFF"),
                ("Footswitch B (DELAY + MOD)",
                 "ON" if (delay_on or mod_on) else "OFF")]

    # ---- things the Cube Baby can't do
    if s["comp"] >= 4:
        skipped.append("Compressor: the Cube Baby has none. Pick evenly "
                       "and keep GAIN a bit lower for clear notes.")
        fit -= 4
    if s["gate"] >= 5 and need > 6:
        skipped.append("Noise gate: none built in. Mute the strings "
                       "between phrases.")
        fit -= 2
    if abs(bass - 5) > 2.2:
        fit -= 3
        tips.append("No Bass knob: " + (
            "use the 4x12 cab (slot 4) and a neck pickup for more lows."
            if bass > 5 else
            "use the 2x12 cab (slot 5) and roll your guitar tone down "
            "a little for fewer lows."))
    if abs(mid - 5) > 2.2:
        fit -= 3
        tips.append("No Mid knob: " + (
            "the 2x12 cab (slot 5) pushes the mids."
            if mid > 5 else
            "the 4x12 cab (slot 4) or a scooped distortion TYPE helps."))

    alt = slot + 1 if slot < 9 else slot - 1
    tips.insert(0, "Different flavour? Try TYPE %d (%s) at the same GAIN."
                % (alt, CUBE_TYPES[alt][0]))
    tips.insert(1, o["tip"])
    tips.insert(2, "Dial it in on LIVE mode (knobs control the sound), "
                   "then set VOLUME to match the song.")
    tips.extend(_clean_notes(s))

    kind = g["kind"]
    return dict(
        kind="cube", guitar=gear.get("guitar") or GUITAR_SKIP,
        pickup=pickup_advice(kind, mode, need, feat["centroid"],
                             s["pickup"]),
        volume=10, tone_knob=TONE_KNOB[kind][0 if mode == "rhythm" else 1],
        knobs=knobs, switches=switches,
        type_line="TYPE %d: %s (%s)" % (slot, tname, tcat),
        ir_line=ir_line, warnings=warn, skipped=skipped, tips=tips,
        effects=[], fit=int(clip(fit, 0, 100)))


def adapt_generic(s, feat, gear):
    """No pedal chosen: show ideal, gear-independent knob settings."""
    g = GUITARS.get(gear.get("guitar"), GUITARS[GUITAR_SKIP])
    kind = g["kind"]
    mode = s["mode"]
    need = clip(s["gain"] + g["gain"])
    vals = [("Gain", need), ("Bass", s["bass"] + g["bass"]),
            ("Mid", s["mid"] + g["mid"]), ("Treble", s["treble"] + g["treb"]),
            ("Presence", s["presence"]), ("Reverb", clip(s["reverb"] / 6.0))]
    knobs = [(n, clip(v), "%.1f" % clip(v)) for n, v in vals]
    eff = [p.strip() for p in s["pedals"] if p.strip()]
    return dict(
        kind="generic", guitar=gear.get("guitar") or GUITAR_SKIP,
        pickup=pickup_advice(kind, mode, need, feat["centroid"],
                             s["pickup"]),
        volume=10, tone_knob=TONE_KNOB[kind][0 if mode == "rhythm" else 1],
        knobs=knobs, switches=[], type_line="Amp: " + s["amp"], ir_line="",
        warnings=[], skipped=[], tips=_clean_notes(s), effects=eff, fit=None)


def adapt(s, feat, gear):
    if gear.get("pedal") == PEDAL_CUBE:
        return adapt_cube(s, feat, gear)
    return adapt_generic(s, feat, gear)


def rig_report(feat, s, rig):
    w = 44
    L = ["=" * w, "  TONE MATCH >> %s TONE" % s["mode"].upper(), "=" * w,
         " Song    : %s" % feat["file"][:32],
         " Tempo   : ~%d BPM" % round(feat["bpm"]),
         " Color   : %s" % brightness_label(feat["centroid"])]
    if rig["fit"] is not None:
        L.append(" Gear fit: %d%%" % rig["fit"])
    L += ["-" * w, " GUITAR : %s" % rig["guitar"],
          "  Pickup : %s" % rig["pickup"],
          "  Volume %d / Tone %d" % (rig["volume"], rig["tone_knob"]),
          "-" * w,
          " CUBE BABY" if rig["kind"] == "cube" else " AMP / EFFECTS"]
    if rig["type_line"]:
        L.append("  " + rig["type_line"])
    for n, v, c in rig["knobs"]:
        L.append("  %-7s %s %s" % (n, bar(v), c))
    for n, st in rig["switches"]:
        L.append("  %s: %s" % (n, st))
    if rig["ir_line"]:
        L.append("  IR: " + rig["ir_line"])
    for e in rig["effects"]:
        L.append("  * " + e)
    for x in rig["skipped"]:
        L.append("  - " + x)
    L.append("-" * w)
    L.append(" TIPS:")
    for x in rig["tips"]:
        L.append("  - " + x)
    L.append("=" * w)
    return "\n".join(L)


def gear_file():
    try:
        base = os.path.dirname(os.path.abspath(__file__))
    except Exception:
        base = os.getcwd()
    return os.path.join(base, "tone_gear.json")


def default_gear():
    return dict(guitar=GUITAR_SKIP, pedal=PEDAL_CUBE, output=OUT_SKIP)


def load_gear():
    d = default_gear()
    try:
        with open(gear_file()) as fh:
            raw = json.load(fh)
        if raw.get("guitar") in GUITARS:
            d["guitar"] = raw["guitar"]
        if raw.get("output") in OUTPUTS:
            d["output"] = raw["output"]
        if raw.get("pedal") in (PEDAL_CUBE, PEDAL_NONE):
            d["pedal"] = raw["pedal"]
    except Exception:
        pass
    return d


def save_gear(d):
    try:
        with open(gear_file(), "w") as fh:
            json.dump(d, fh)
    except Exception:
        pass


# ----------------------------------------------------
#  TOUCH INTERFACE (tkinter - works in Pydroid 3)
# ----------------------------------------------------
BG = "#14151a"
PANEL = "#1e2027"
PANEL2 = "#2a2d37"
ACC = "#ff8a00"
TXT = "#f1f1f1"
MUT = "#9aa1ad"
OKC = "#3ddc84"
BAD = "#ff5c5c"
BLUE = "#4aa3ff"
SKIP_TYPES = (tk.Button, tk.Entry, tk.Checkbutton, tk.Menubutton,
              tk.Scrollbar, tk.Listbox)


class ScrollFrame(tk.Frame):
    """Vertical scroll area: scrollbar + finger-drag scrolling."""

    def __init__(self, parent, sbw):
        tk.Frame.__init__(self, parent, bg=BG)
        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0,
                                yscrollincrement=6)
        self.sb = tk.Scrollbar(self, orient="vertical", width=sbw,
                               command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.sb.set)
        self.sb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = tk.Frame(self.canvas, bg=BG)
        self.win = self.canvas.create_window((0, 0), window=self.inner,
                                             anchor="nw")
        self.inner.bind("<Configure>", self._cfg)
        self.canvas.bind("<Configure>", self._ccfg)
        self._y = 0
        self.drag(self.canvas)
        self.drag(self.inner)
        for w in (self.canvas, self.inner):
            w.bind("<MouseWheel>", self._wheel)
            w.bind("<Button-4>",
                   lambda e: self.canvas.yview_scroll(-3, "units"))
            w.bind("<Button-5>",
                   lambda e: self.canvas.yview_scroll(3, "units"))

    def _cfg(self, e):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _ccfg(self, e):
        self.canvas.itemconfig(self.win, width=e.width)

    def _wheel(self, e):
        self.canvas.yview_scroll(int(-e.delta / 40), "units")

    def _press(self, e):
        self._y = e.y_root

    def _move(self, e):
        n = int((self._y - e.y_root) / 6)
        if n:
            self.canvas.yview_scroll(n, "units")
            self._y = e.y_root

    def drag(self, w):
        if not isinstance(w, SKIP_TYPES):
            w.bind("<ButtonPress-1>", self._press, add="+")
            w.bind("<B1-Motion>", self._move, add="+")
        for c in w.winfo_children():
            self.drag(c)

    def top(self):
        self.canvas.yview_moveto(0)


class TouchList(tk.Frame):
    """Listbox you can scroll by dragging; tap an row to pick it."""

    def __init__(self, parent, font, sbw, on_pick):
        tk.Frame.__init__(self, parent, bg=BG)
        self.on_pick = on_pick
        self.rowpx = int(font.metrics("linespace")) + 4
        self.sb = tk.Scrollbar(self, width=sbw)
        self.lb = tk.Listbox(self, font=font, bg=PANEL, fg=TXT, bd=0,
                             highlightthickness=0, activestyle="none",
                             selectbackground=ACC, selectforeground="black",
                             yscrollcommand=self.sb.set)
        self.sb.config(command=self.lb.yview)
        self.sb.pack(side="right", fill="y")
        self.lb.pack(side="left", fill="both", expand=True)
        self.lb.bind("<ButtonPress-1>", self._press)
        self.lb.bind("<B1-Motion>", self._move)
        self.lb.bind("<ButtonRelease-1>", self._release)
        self._y0 = self._y = 0
        self._moved = False
        self._idx = None
        self.count = 0

    def set_items(self, items):
        self.lb.delete(0, "end")
        for it in items:
            self.lb.insert("end", it)
        self.count = len(items)

    def _press(self, e):
        self._y0 = self._y = e.y
        self._moved = False
        self._idx = self.lb.nearest(e.y)
        return "break"

    def _move(self, e):
        if abs(e.y - self._y0) > self.rowpx * 0.4:
            self._moved = True
        n = int((self._y - e.y) / float(self.rowpx))
        if n:
            self.lb.yview_scroll(n, "units")
            self._y = e.y
        return "break"

    def _release(self, e):
        if self._moved or self._idx is None or self._idx >= self.count:
            return "break"
        self.lb.selection_clear(0, "end")
        self.lb.selection_set(self._idx)
        self.on_pick(self._idx)
        return "break"


class App:
    def __init__(self, root):
        self.root = root
        root.title("Guitar Tone Matcher")
        root.configure(bg=BG)
        sw = root.winfo_screenwidth()
        if sw > 1400:                       # desktop test window
            self.WIN = 560
            root.geometry("560x900")
            self.fs = 11
            self.u = 10.0
        else:                               # phone
            self.WIN = sw
            self.fs = int(max(9, min(14, round(sw / 105.0))))
            self.u = sw / 60.0
        fs = self.fs
        self.pad = max(6, int(self.u * 0.8))
        self.sbw = max(22, int(self.u * 1.7))
        self.wrap = self.WIN - self.sbw - 6 * self.pad
        fam = "Helvetica"
        self.f = tkfont.Font(family=fam, size=fs)
        self.fb = tkfont.Font(family=fam, size=fs, weight="bold")
        self.fbig = tkfont.Font(family=fam, size=int(fs * 1.5),
                                weight="bold")
        self.fsm = tkfont.Font(family=fam, size=max(8, fs - 2))
        self.fkn = tkfont.Font(family=fam, size=max(8, fs - 2),
                               weight="bold")

        # state
        self.song_src = None          # dict(kind,title,path/url,...)
        self.guitar_path = None
        self.mode = "rhythm"
        self.feat = None
        self.settings = {}
        self.rig = None
        self.gear = load_gear()
        self.busy = False
        self.q = queue.Queue()
        self.mode_btns = []
        self.status_labels = []
        self.last_dir = "/storage/emulated/0"
        self.current = None
        self.browse_cb = None
        self.browse_back = "song"
        self.entries = []
        self.online_results = []
        self.source = "deezer"
        self.rtab = "settings"

        # bottom nav + body
        self.nav = tk.Frame(root, bg=PANEL)
        self.nav.pack(side="bottom", fill="x")
        self.body = tk.Frame(root, bg=BG)
        self.body.pack(side="top", fill="both", expand=True)
        self.navbtn = {}
        for i, (key, label) in enumerate((("song", "SONG"), ("gear", "GEAR"),
                                          ("result", "TONE"),
                                          ("apply", "APPLY"))):
            b = tk.Button(self.nav, text=label, font=self.fb, bd=0,
                          bg=PANEL, fg=MUT, activebackground=ACC,
                          pady=self.pad // 2 + 3,
                          command=lambda k=key: self.goto(k))
            b.grid(row=0, column=i, sticky="ew")
            self.nav.grid_columnconfigure(i, weight=1, uniform="nav")
            self.navbtn[key] = b

        self.scr = {}
        self.build_song()
        self.build_gear()
        self.build_result()
        self.build_apply()
        self.build_browse()
        self.build_online()
        self.goto("song")
        self.root.after(150, self.poll)

    # ---------- generic helpers ----------
    def button(self, parent, text, cmd, bg=PANEL2, fg=TXT, font=None,
               big=False):
        return tk.Button(parent, text=text, command=cmd,
                         font=font or (self.fb), bg=bg, fg=fg,
                         activebackground=ACC, activeforeground="black",
                         bd=0, pady=self.pad // 2 + (self.pad // 2 if big
                                                     else 0),
                         padx=self.pad // 2)

    def label(self, parent, text, color=TXT, font=None, anchor="w",
              bg=None, wrap=None):
        return tk.Label(parent, text=text, fg=color, bg=bg or parent["bg"],
                        font=font or self.f, anchor=anchor, justify="left",
                        wraplength=wrap or self.wrap)

    def card(self, parent, title):
        outer = tk.Frame(parent, bg=PANEL)
        outer.pack(fill="x", padx=self.pad, pady=self.pad // 3 + 1)
        if title:
            tk.Label(outer, text=title, fg=ACC, bg=PANEL, font=self.fb,
                     anchor="w").pack(fill="x", padx=self.pad,
                                      pady=(self.pad // 2, 0))
        inner = tk.Frame(outer, bg=PANEL)
        inner.pack(fill="x", padx=self.pad, pady=self.pad // 2)
        return inner

    def make_screen(self, name):
        sf = ScrollFrame(self.body, self.sbw)
        self.scr[name] = sf
        return sf

    def goto(self, name):
        if name == "result":
            self.render_result()
        if name == "apply":
            self.refresh_apply()
        self.show(name)

    def show(self, name):
        if self.current:
            self.scr[self.current].pack_forget()
        self.scr[name].pack(fill="both", expand=True)
        self.current = name
        hl = {"online": "song", "browse": self.browse_back}.get(name, name)
        for k, b in self.navbtn.items():
            b.config(bg=ACC if k == hl else PANEL,
                     fg="black" if k == hl else MUT)

    def set_mode(self, m):
        self.mode = m
        self.refresh_modes()
        if self.current == "result":
            self.render_result()
        if self.current == "apply":
            self.refresh_apply()

    def toggle_pair(self, parent, text1, cmd1, text2, cmd2):
        """Two equal-width toggle buttons. Returns (frame, b1, b2)."""
        f = tk.Frame(parent, bg=parent["bg"])
        b1 = self.button(f, text1, cmd1, big=True)
        b2 = self.button(f, text2, cmd2, big=True)
        b1.grid(row=0, column=0, sticky="ew", padx=2)
        b2.grid(row=0, column=1, sticky="ew", padx=2)
        f.grid_columnconfigure(0, weight=1, uniform="tg")
        f.grid_columnconfigure(1, weight=1, uniform="tg")
        return f, b1, b2

    def mode_toggle(self, parent):
        f, b1, b2 = self.toggle_pair(parent, "RHYTHM",
                                     lambda: self.set_mode("rhythm"),
                                     "LEAD",
                                     lambda: self.set_mode("lead"))
        self.mode_btns.append((b1, b2))
        self.refresh_modes()
        return f

    @staticmethod
    def _style(b, on):
        b.config(bg=ACC if on else PANEL2, fg="black" if on else TXT)

    def refresh_modes(self):
        alive = []
        for b1, b2 in self.mode_btns:
            try:
                self._style(b1, self.mode == "rhythm")
                self._style(b2, self.mode == "lead")
                alive.append((b1, b2))
            except tk.TclError:
                pass
        self.mode_btns = alive

    def set_status(self, text, color=MUT):
        alive = []
        for lb in self.status_labels:
            try:
                lb.config(text=text, fg=color)
                alive.append(lb)
            except tk.TclError:
                pass
        self.status_labels = alive

    def status_label(self, parent):
        lb = self.label(parent, "", MUT, self.fsm)
        self.status_labels.append(lb)
        return lb

    # ---------- background work ----------
    def run_bg(self, fn, done):
        def t():
            try:
                res = fn()
            except Exception as ex:
                self.q.put(("err", "Error: %s" % ex, None))
                return
            self.q.put(("done", done, res))
        threading.Thread(target=t, daemon=True).start()

    def poll(self):
        try:
            while True:
                kind, a, b = self.q.get_nowait()
                if kind == "log":
                    self.set_status(a, ACC)
                elif kind == "done":
                    a(b)
                elif kind == "err":
                    self.busy = False
                    self.set_status(a, BAD)
        except queue.Empty:
            pass
        self.root.after(150, self.poll)

    def qlog(self, t):
        self.q.put(("log", t.strip(), None))

    # ---------- SONG screen ----------
    def build_song(self):
        sf = self.make_screen("song")
        p = sf.inner
        self.label(p, "GUITAR TONE MATCHER", ACC, self.fbig,
                   anchor="center").pack(fill="x", pady=(self.pad, 0))
        self.label(p, "Copy the guitar tone of any song", MUT, self.fsm,
                   anchor="center").pack(fill="x")

        c = self.card(p, "1. SONG")
        self.song_lbl = self.label(c, "No song selected", MUT)
        self.song_lbl.pack(fill="x", pady=(0, self.pad // 2))
        row = tk.Frame(c, bg=PANEL)
        row.pack(fill="x")
        b1 = self.button(row, "Search online", self.open_online,
                         bg=ACC, fg="black", big=True)
        b2 = self.button(row, "Phone file", self.pick_song, big=True)
        b1.grid(row=0, column=0, sticky="ew", padx=2)
        b2.grid(row=0, column=1, sticky="ew", padx=2)
        row.grid_columnconfigure(0, weight=1, uniform="s")
        row.grid_columnconfigure(1, weight=1, uniform="s")

        c = self.card(p, "2. SECTION (phone files)")
        self.sect_note = self.label(c, "Online previews are ~30 s, so the "
                                       "whole clip is used.", MUT, self.fsm)
        self.sect_note.pack(fill="x", pady=(0, self.pad // 2))
        self.start_var = tk.StringVar(value="0:30")
        self.len_var = tk.StringVar(value="25")
        for text, var, step in (("Start", self.start_var, 10),
                                ("Length (s)", self.len_var, 5)):
            row = tk.Frame(c, bg=PANEL)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=text, fg=MUT, bg=PANEL, font=self.f,
                     width=10, anchor="w").pack(side="left")
            self.button(row, "-", lambda v=var, s=step: self.nudge(v, -s)
                        ).pack(side="left")
            tk.Entry(row, textvariable=var, font=self.fb, justify="center",
                     bg=PANEL2, fg=TXT, insertbackground=TXT, bd=0,
                     width=6).pack(side="left", expand=True, fill="x",
                                   padx=self.pad // 2,
                                   ipady=self.pad // 3)
            self.button(row, "+", lambda v=var, s=step: self.nudge(v, s)
                        ).pack(side="left")

        c = self.card(p, "3. TONE TYPE")
        self.mode_toggle(c).pack(fill="x")
        self.label(c, "Rhythm = tight & punchy.  Lead = more gain, sustain "
                      "& delay.", MUT, self.fsm).pack(
            fill="x", pady=(self.pad // 2, 0))

        c = self.card(p, "4. MY GEAR (optional)")
        self.gear_lbl = self.label(c, "", TXT)
        self.gear_lbl.pack(fill="x", pady=(0, self.pad // 2))
        self.button(c, "Choose gear", lambda: self.goto("gear")
                    ).pack(fill="x")
        self.update_gear_summary()

        self.button(p, "MATCH TONE", self.do_match, bg=ACC, fg="black",
                    big=True).pack(fill="x", padx=self.pad, pady=self.pad)
        self.status_label(p).pack(fill="x", padx=self.pad)
        tk.Frame(p, bg=BG, height=self.pad * 2).pack()
        sf.drag(p)

    def nudge(self, var, delta):
        v = max(0.0, parse_time(var.get(), 0.0) + delta)
        if var is self.start_var:
            var.set("%d:%02d" % (int(v) // 60, int(v) % 60))
        else:
            var.set(str(int(max(5, min(60, v)))))

    def pick_song(self):
        self.open_browser("Choose a song", self.song_chosen, "song")

    def song_chosen(self, path):
        self.song_src = dict(kind="local", path=path,
                             title=os.path.basename(path))
        self.song_lbl.config(text=os.path.basename(path), fg=OKC)

    def update_gear_summary(self):
        g = self.gear
        gtr = "any" if g["guitar"] == GUITAR_SKIP else g["guitar"]
        ped = ("M-VAVE Cube Baby" if g["pedal"] == PEDAL_CUBE
               else "none (generic)")
        out = "any" if g["output"] == OUT_SKIP else g["output"]
        self.gear_lbl.config(text="Guitar: %s\nPedal: %s\nOutput: %s"
                                  % (gtr, ped, out))

    def do_match(self):
        if self.busy:
            return
        src = self.song_src
        if not src:
            self.set_status("Choose a song first.", BAD)
            return
        st = parse_time(self.start_var.get(), 30.0)
        du = float(np.clip(parse_time(self.len_var.get(), 25.0), 5, 60))
        self.busy = True
        self.set_status("Starting... (can take 10-40 s)", ACC)

        def work():
            if src["kind"] == "local":
                path, s0, d0 = src["path"], st, du
            else:
                self.qlog("Downloading preview...")
                path, s0, d0 = download_audio(src), 0.0, 30.0
            feat = analyze_song(path, s0, d0, log=self.qlog)
            feat["file"] = src["title"]
            return feat
        self.run_bg(work, self.match_done)

    def match_done(self, feat):
        self.busy = False
        self.feat = feat
        self.settings = {m: make_settings(feat, m)
                         for m in ("rhythm", "lead")}
        self.set_status("Done! Tone matched.", OKC)
        self.goto("result")

    # ---------- ONLINE search screen ----------
    def build_online(self):
        fr = tk.Frame(self.body, bg=BG)
        self.scr["online"] = fr
        pad = self.pad
        self.label(fr, "FIND A SONG ONLINE", ACC, self.fbig,
                   anchor="center").pack(fill="x", pady=(pad, 2))
        row = tk.Frame(fr, bg=BG)
        row.pack(fill="x", padx=pad)
        self.q_var = tk.StringVar()
        tk.Entry(row, textvariable=self.q_var, font=self.f, bg=PANEL2,
                 fg=TXT, insertbackground=TXT, bd=0).pack(
            side="left", expand=True, fill="x", ipady=pad // 2)
        self.button(row, "Search", self.do_search, bg=ACC, fg="black"
                    ).pack(side="left", padx=(pad // 2, 0))
        sub = tk.Frame(fr, bg=BG)
        sub.pack(fill="x", padx=pad, pady=(pad // 2, 0))
        f, self.src_b1, self.src_b2 = self.toggle_pair(
            sub, "Deezer (MP3)", lambda: self.set_source("deezer"),
            "iTunes (M4A)", lambda: self.set_source("itunes"))
        f.pack(fill="x")
        self.set_source("deezer")
        self.online_status = self.label(fr, "Type a song or artist.", MUT,
                                        self.fsm)
        self.online_status.pack(fill="x", padx=pad, pady=(pad // 2, 0))
        self.button(fr, "Back", lambda: self.show("song")
                    ).pack(side="bottom", fill="x", padx=pad, pady=pad // 2)
        row = tk.Frame(fr, bg=BG)
        row.pack(side="bottom", fill="x", padx=pad)
        self.url_var = tk.StringVar()
        tk.Entry(row, textvariable=self.url_var, font=self.fsm, bg=PANEL2,
                 fg=TXT, insertbackground=TXT, bd=0).pack(
            side="left", expand=True, fill="x", ipady=pad // 3)
        self.button(row, "Use link", self.use_link, font=self.fsm
                    ).pack(side="left", padx=(pad // 2, 0))
        self.label(fr, "Or paste a direct audio link (mp3/wav...):", MUT,
                   self.fsm).pack(side="bottom", fill="x", padx=pad)
        self.dec_lbl = self.label(fr, "", MUT, self.fsm)
        self.dec_lbl.pack(side="bottom", fill="x", padx=pad)
        self.res_list = TouchList(fr, self.f, self.sbw, self.online_pick)
        self.res_list.pack(fill="both", expand=True, padx=pad,
                           pady=pad // 2)

    def set_source(self, src):
        self.source = src
        self._style(self.src_b1, src == "deezer")
        self._style(self.src_b2, src == "itunes")

    def open_online(self):
        txt, mp3, m4a = decoder_status()
        self.dec_lbl.config(text="Decoders - " + txt,
                            fg=OKC if mp3 else BAD)
        self.show("online")

    def do_search(self):
        q = self.q_var.get().strip()
        if not q:
            self.online_status.config(text="Type something first.", fg=BAD)
            return
        src = self.source
        self.online_status.config(text="Searching...", fg=ACC)

        def work():
            return search_online(q, src)

        def done(res):
            self.online_results = res
            self.res_list.set_items(["%s - %s" % (r["title"], r["artist"])
                                     for r in res])
            self.online_status.config(
                text=("%d results. Tap one." % len(res)) if res
                else "No previews found. Try another search or source.",
                fg=OKC if res else BAD)

        def t():
            try:
                r = work()
            except Exception as ex:
                self.q.put(("done", lambda _: self.online_status.config(
                    text="Search failed: %s (check internet)" % ex,
                    fg=BAD), None))
                return
            self.q.put(("done", done, r))
        threading.Thread(target=t, daemon=True).start()

    def online_pick(self, idx):
        if idx >= len(self.online_results):
            return
        r = self.online_results[idx]
        title = "%s - %s" % (r["title"], r["artist"])
        self.song_src = dict(kind="online", title=title, url=r["url"],
                             ext=r["ext"], artist=r["artist"])
        self.song_lbl.config(text="%s  (30 s preview)" % title, fg=OKC)
        self.show("song")

    def use_link(self):
        url = self.url_var.get().strip()
        if not url.lower().startswith("http"):
            self.online_status.config(text="Paste a link starting with "
                                           "http", fg=BAD)
            return
        path = urllib.parse.urlparse(url).path
        name = os.path.basename(path) or "link"
        self.song_src = dict(kind="url", title=name, url=url,
                             ext=os.path.splitext(path)[1] or ".mp3",
                             artist="")
        self.song_lbl.config(text="Link: %s" % name, fg=OKC)
        self.show("song")

    # ---------- GEAR screen ----------
    def build_gear(self):
        sf = self.make_screen("gear")
        p = sf.inner
        self.label(p, "MY GEAR", ACC, self.fbig, anchor="center").pack(
            fill="x", pady=(self.pad, 0))
        self.label(p, "Everything is optional. Pick only what you have - "
                      "skipped items just get generic settings.", MUT,
                   self.fsm, anchor="center").pack(fill="x", padx=self.pad)

        c = self.card(p, "GUITAR")
        self.guitar_var = tk.StringVar(value=self.gear["guitar"])
        self.option(c, self.guitar_var, list(GUITARS.keys()))

        c = self.card(p, "PEDAL")
        f, self.pd1, self.pd2 = self.toggle_pair(
            c, "CUBE BABY", lambda: self.set_pedal(PEDAL_CUBE),
            "NO PEDAL", lambda: self.set_pedal(PEDAL_NONE))
        f.pack(fill="x")
        self.label(c, "M-VAVE Cube Baby: 9 preamp TYPEs, GAIN, TONE, MOD "
                      "(chorus/phaser), delay (TIME/FB/MIX), REVERB, 8 IR "
                      "cabs, VOLUME. It has no Bass/Mid knob and no "
                      "compressor, so the app shapes those with the TYPE "
                      "and IR CAB.", MUT, self.fsm).pack(
            fill="x", pady=(self.pad // 2, 0))
        self._style(self.pd1, self.gear["pedal"] == PEDAL_CUBE)
        self._style(self.pd2, self.gear["pedal"] == PEDAL_NONE)

        c = self.card(p, "HOW YOU LISTEN / PLAY THROUGH")
        self.out_var = tk.StringVar(value=self.gear["output"])
        self.option(c, self.out_var, list(OUTPUTS.keys()))
        self.label(c, "Decides if the IR cab stays ON (headphones, PA) or "
                      "OFF (into a guitar amp).", MUT, self.fsm).pack(
            fill="x", pady=(self.pad // 2, 0))

        self.button(p, "Reset: skip everything", self.reset_gear
                    ).pack(fill="x", padx=self.pad, pady=self.pad // 2)
        self.button(p, "DONE", lambda: self.goto("song"), bg=ACC, fg="black",
                    big=True).pack(fill="x", padx=self.pad)
        tk.Frame(p, bg=BG, height=self.pad * 2).pack()
        sf.drag(p)

    def option(self, parent, var, values):
        om = tk.OptionMenu(parent, var, *values,
                           command=lambda v: self.gear_changed())
        om.config(font=self.f, bg=PANEL2, fg=TXT, activebackground=ACC,
                  activeforeground="black", highlightthickness=0, bd=0,
                  anchor="w", wraplength=self.wrap - 40, justify="left",
                  padx=self.pad // 2, pady=self.pad // 2)
        om["menu"].config(font=self.f, bg=PANEL2, fg=TXT,
                          activebackground=ACC, activeforeground="black")
        om.pack(fill="x")
        return om

    def set_pedal(self, val):
        self.gear["pedal"] = val
        self._style(self.pd1, val == PEDAL_CUBE)
        self._style(self.pd2, val == PEDAL_NONE)
        self.gear_changed()

    def reset_gear(self):
        self.guitar_var.set(GUITAR_SKIP)
        self.out_var.set(OUT_SKIP)
        self.set_pedal(PEDAL_NONE)

    def gear_changed(self):
        self.gear = dict(guitar=self.guitar_var.get(),
                         output=self.out_var.get(),
                         pedal=self.gear.get("pedal", PEDAL_CUBE))
        save_gear(self.gear)
        self.update_gear_summary()

    # ---------- RESULT screen ----------
    def build_result(self):
        self.make_screen("result")

    def clear(self, frame):
        for w in frame.winfo_children():
            w.destroy()

    def set_rtab(self, t):
        self.rtab = t
        self.render_result()

    def group(self, parent, title):
        pad = self.pad
        f = tk.Frame(parent, bg=PANEL)
        f.pack(fill="x", padx=pad, pady=pad // 3 + 1)
        tk.Label(f, text=title, fg=MUT, bg=PANEL, font=self.fsm,
                 anchor="w").pack(fill="x", padx=pad, pady=(pad // 2, 0))
        inner = tk.Frame(f, bg=PANEL)
        inner.pack(fill="x", padx=pad // 2, pady=(0, pad // 2))
        return inner

    def chips(self, parent, switches):
        pad = self.pad
        row = tk.Frame(parent, bg=BG)
        row.pack(fill="x", padx=pad, pady=pad // 3 + 1)
        for i, (name, st) in enumerate(switches):
            on = st == "ON"
            parts = name.split(" ", 2)
            short = "%s: %s" % (parts[1], parts[2].strip("()")) \
                if len(parts) == 3 else name
            bg = "#1d3a2a" if on else PANEL
            box = tk.Frame(row, bg=bg)
            box.grid(row=0, column=i, sticky="nsew", padx=2)
            row.grid_columnconfigure(i, weight=1, uniform="ch")
            tk.Label(box, text=short, fg=MUT, bg=bg, font=self.fsm,
                     wraplength=self.WIN // 2 - 4 * pad).pack(pady=(pad // 2, 0))
            tk.Label(box, text=st, fg=OKC if on else MUT, bg=bg,
                     font=self.fb).pack(pady=(0, pad // 2))

    def render_result(self):
        sf = self.scr["result"]
        p = sf.inner
        self.clear(p)
        if not self.feat:
            self.label(p, "No tone matched yet.\nGo to SONG, pick a song "
                          "and press MATCH TONE.", MUT).pack(
                fill="x", padx=self.pad, pady=self.pad * 3)
            sf.drag(p)
            return
        s = self.settings[self.mode]
        rig = adapt(s, self.feat, self.gear)
        self.rig = rig
        f = self.feat
        pad = self.pad
        W = self.WIN - self.sbw - 4 * pad

        top = tk.Frame(p, bg=BG)
        top.pack(fill="x", padx=pad, pady=(pad // 2, 0))
        self.mode_toggle(top).pack(fill="x")

        # header: song + one summary line + thin fit bar
        c = self.card(p, None)
        self.label(c, f["file"], TXT, self.fb).pack(fill="x")
        info = "~%d BPM  |  %s" % (round(f["bpm"]),
                                    brightness_label(f["centroid"]))
        if rig["fit"] is not None:
            info += "  |  Gear fit %d%%" % rig["fit"]
        self.label(c, info, MUT, self.fsm).pack(fill="x")
        if rig["fit"] is not None:
            col = (OKC if rig["fit"] >= 80 else ACC if rig["fit"] >= 55
                   else BAD)
            cv = tk.Canvas(c, width=W - pad, height=6, bg=PANEL,
                           highlightthickness=0)
            cv.pack(pady=(pad // 3, 0))
            cv.create_rectangle(0, 0, W - pad, 6, fill=PANEL2, outline="")
            cv.create_rectangle(0, 0, (W - pad) * rig["fit"] / 100.0, 6,
                                fill=col, outline="")

        # tabs
        tabs = tk.Frame(p, bg=BG)
        tabs.pack(fill="x", padx=pad, pady=(pad // 3, pad // 3))
        for i, (k, t) in enumerate((("settings", "SETTINGS"),
                                    ("guitar", "GUITAR"),
                                    ("tips", "TIPS"))):
            b = self.button(tabs, t, lambda k=k: self.set_rtab(k),
                            font=self.fsm)
            self._style(b, self.rtab == k)
            b.grid(row=0, column=i, sticky="ew", padx=2)
            tabs.grid_columnconfigure(i, weight=1, uniform="rt")

        body = tk.Frame(p, bg=BG)
        body.pack(fill="x")
        if self.rtab == "guitar":
            self.tab_guitar(body, rig)
        elif self.rtab == "tips":
            self.tab_tips(body, rig)
        else:
            self.tab_settings(body, rig, W)

        row = tk.Frame(p, bg=BG)
        row.pack(fill="x", padx=pad, pady=pad)
        b1 = self.button(row, "Save report", self.save_report)
        b2 = self.button(row, "Apply to my guitar",
                         lambda: self.goto("apply"), bg=ACC, fg="black")
        b1.grid(row=0, column=0, sticky="ew", padx=2)
        b2.grid(row=0, column=1, sticky="ew", padx=2)
        row.grid_columnconfigure(0, weight=1, uniform="r")
        row.grid_columnconfigure(1, weight=1, uniform="r")
        self.report_lbl = self.label(p, "", MUT, self.fsm)
        self.report_lbl.pack(fill="x", padx=pad)
        tk.Frame(p, bg=BG, height=pad).pack()
        sf.drag(p)
        sf.top()

    def tab_settings(self, body, rig, W):
        pad = self.pad
        k = dict((n, (v, c)) for n, v, c in rig["knobs"])
        Wg = W - pad
        if rig["kind"] == "cube":
            g = self.group(body, "PREAMP")
            self.label(g, rig["type_line"], BLUE, self.fb).pack(fill="x")
            self.draw_knobs(g, [(n, k[n][0], k[n][1]) for n in
                                ("TYPE", "GAIN", "TONE", "VOLUME")], 4, Wg)
            g = self.group(body, "EFFECTS & CAB")
            self.draw_knobs(g, [(n, k[n][0], k[n][1]) for n in
                                ("MOD", "TIME", "FB", "MIX", "REVERB",
                                 "IR CAB")], 3, Wg)
            self.chips(body, rig["switches"])
            if rig["ir_line"]:
                g = self.group(body, "CAB")
                self.label(g, rig["ir_line"], TXT, self.fsm).pack(fill="x")
        else:
            g = self.group(body, "IDEAL KNOBS (any gear)")
            self.label(g, rig["type_line"], BLUE, self.fb).pack(fill="x")
            self.draw_knobs(g, rig["knobs"], 3, Wg)
            if rig["effects"]:
                g = self.group(body, "EFFECTS")
                for e in rig["effects"]:
                    self.label(g, "- " + e, TXT, self.fsm).pack(fill="x")
        if rig["warnings"]:
            g = self.group(body, "HEADS UP")
            for w in rig["warnings"]:
                self.label(g, w, BAD, self.fsm).pack(fill="x")

    def tab_guitar(self, body, rig):
        g = self.group(body, "GUITAR")
        name = "Not specified" if rig["guitar"] == GUITAR_SKIP \
            else rig["guitar"]
        self.label(g, name, MUT, self.fsm).pack(fill="x")
        self.label(g, "Pickup", MUT, self.fsm).pack(fill="x",
                                                    pady=(self.pad // 2, 0))
        self.label(g, rig["pickup"], TXT, self.fb).pack(fill="x")
        self.label(g, "Guitar knobs", MUT, self.fsm).pack(
            fill="x", pady=(self.pad // 2, 0))
        self.label(g, "Volume %d   |   Tone %d" % (rig["volume"],
                                                   rig["tone_knob"]),
                   TXT, self.fb).pack(fill="x")
        if rig["skipped"]:
            g = self.group(body, "NOT POSSIBLE ON THIS GEAR")
            for x in rig["skipped"]:
                self.label(g, x, TXT, self.fsm).pack(
                    fill="x", pady=(0, self.pad // 3))

    def tab_tips(self, body, rig):
        g = self.group(body, "TIPS")
        for x in rig["tips"]:
            self.label(g, x, TXT, self.fsm).pack(
                fill="x", pady=(0, self.pad // 2))

    def draw_knobs(self, parent, knobs, cols, W):
        rows = int(math.ceil(len(knobs) / float(cols)))
        r = min(W / (cols * 2.4), self.u * 3.6)
        l1 = int(self.fkn.metrics("linespace"))
        l2 = int(self.fsm.metrics("linespace"))
        cell_h = int(r * 2 + l1 + l2 + 16)
        cv = tk.Canvas(parent, width=int(W), height=cell_h * rows + 4,
                       bg=PANEL, highlightthickness=0)
        cv.pack()
        for i, (name, val, cap) in enumerate(knobs):
            col, row = i % cols, i // cols
            cx = W / float(cols) * (col + 0.5)
            cy = cell_h * row + r + 4
            self.draw_knob(cv, cx, cy, r, val, name, cap, l1, l2)

    def draw_knob(self, cv, cx, cy, r, val, name, cap, l1, l2):
        wd = max(3, int(r * 0.15))
        a = r * 0.84
        cv.create_oval(cx - r, cy - r, cx + r, cy + r, fill="#2b2e38",
                       outline="#484c59", width=2)
        cv.create_arc(cx - a, cy - a, cx + a, cy + a, start=225,
                      extent=-270, style="arc", outline="#444857", width=wd)
        ext = -270.0 * val / 10.0
        if abs(ext) > 1:
            cv.create_arc(cx - a, cy - a, cx + a, cy + a, start=225,
                          extent=ext, style="arc", outline=ACC, width=wd)
        ang = math.radians(225 + ext)
        cv.create_line(cx, cy, cx + r * 0.62 * math.cos(ang),
                       cy - r * 0.62 * math.sin(ang), fill="white",
                       width=max(2, int(r * 0.12)), capstyle="round")
        y1 = cy + r + 6 + l1 / 2.0
        cv.create_text(cx, y1, text=name, fill=TXT, font=self.fkn)
        cv.create_text(cx, y1 + l1 / 2.0 + l2 / 2.0 + 2, text=cap,
                       fill=ACC, font=self.fsm)

    def save_report(self):
        if not self.rig:
            return
        text = rig_report(self.feat, self.settings[self.mode], self.rig)
        paths = []
        try:
            paths.append(os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "tone_report.txt"))
        except Exception:
            pass
        paths.append(os.path.join(os.getcwd(), "tone_report.txt"))
        for pth in paths:
            try:
                with open(pth, "w") as fh:
                    fh.write(text + "\n")
                self.report_lbl.config(text="Saved: " + pth, fg=OKC)
                return
            except Exception:
                continue
        self.report_lbl.config(text="Could not save report.", fg=BAD)

    # ---------- APPLY screen ----------
    def build_apply(self):
        sf = self.make_screen("apply")
        p = sf.inner
        self.label(p, "APPLY TONE TO MY GUITAR", ACC, self.fbig,
                   anchor="center").pack(fill="x", pady=(self.pad, 0))
        self.label(p, "Runs your own guitar recording (WAV) through the "
                      "matched tone: amp sim, EQ, cab, delay, reverb.",
                   MUT, self.fsm, anchor="center").pack(fill="x",
                                                         padx=self.pad)
        c = self.card(p, "1. TONE")
        self.apply_info = self.label(c, "", TXT)
        self.apply_info.pack(fill="x", pady=(0, self.pad // 2))
        self.mode_toggle(c).pack(fill="x")

        c = self.card(p, "2. YOUR GUITAR RECORDING")
        self.guitar_lbl = self.label(c, "No file selected", MUT)
        self.guitar_lbl.pack(fill="x", pady=(0, self.pad // 2))
        self.button(c, "Choose guitar file", self.pick_guitar, big=True
                    ).pack(fill="x")

        c = self.card(p, "3. OPTIONS")
        row = tk.Frame(c, bg=PANEL)
        row.pack(fill="x")
        tk.Label(row, text="Max seconds", fg=MUT, bg=PANEL,
                 font=self.f).pack(side="left")
        self.max_var = tk.StringVar(value="60")
        tk.Entry(row, textvariable=self.max_var, font=self.fb,
                 justify="center", bg=PANEL2, fg=TXT, bd=0, width=6,
                 insertbackground=TXT).pack(side="left", expand=True,
                                            fill="x", padx=self.pad,
                                            ipady=self.pad // 3)
        self.match_var = tk.BooleanVar(value=True)
        tk.Checkbutton(c, text="Auto match-EQ toward the song",
                       variable=self.match_var, font=self.f, bg=PANEL,
                       fg=TXT, selectcolor=PANEL2, activebackground=PANEL,
                       activeforeground=ACC, anchor="w").pack(fill="x")
        self.label(c, "Preview uses the ideal target tone, not limited by "
                      "your gear.", MUT, self.fsm).pack(fill="x")

        self.button(p, "PROCESS", self.do_apply, bg=ACC, fg="black",
                    big=True).pack(fill="x", padx=self.pad, pady=self.pad)
        self.status_label(p).pack(fill="x", padx=self.pad)
        self.out_lbl = self.label(p, "", OKC, self.fsm)
        self.out_lbl.pack(fill="x", padx=self.pad)
        tk.Frame(p, bg=BG, height=self.pad * 2).pack()
        sf.drag(p)

    def refresh_apply(self):
        if self.feat:
            self.apply_info.config(
                text="Song: %s\nMode: %s tone" % (self.feat["file"],
                                                  self.mode.upper()),
                fg=TXT)
        else:
            self.apply_info.config(text="Match a song first (SONG tab).",
                                   fg=BAD)

    def pick_guitar(self):
        self.open_browser("Choose your guitar file", self.guitar_chosen,
                          "apply")

    def guitar_chosen(self, path):
        self.guitar_path = path
        self.guitar_lbl.config(text=os.path.basename(path), fg=OKC)

    def do_apply(self):
        if self.busy:
            return
        if not self.settings:
            self.set_status("Match a song first.", BAD)
            return
        if not self.guitar_path:
            self.set_status("Choose your guitar recording first.", BAD)
            return
        s = self.settings[self.mode]
        feat = self.feat
        gp = self.guitar_path
        mx = max(5.0, parse_time(self.max_var.get(), 60.0))
        usem = bool(self.match_var.get())
        mode = self.mode
        self.busy = True
        self.out_lbl.config(text="")
        self.set_status("Processing...", ACC)

        def work():
            self.qlog("Loading guitar file...")
            data, sr = load_audio(gp)
            mono = data.mean(axis=1)[:int(mx * sr)]
            if sr > 48000:
                mono = resample(mono, sr, 44100)
                sr = 44100
            out = process_guitar(mono, sr, s, feat, usem, log=self.qlog)
            base = os.path.splitext(os.path.basename(gp))[0]
            name = "%s_%s_tone.wav" % (base, mode)
            pcm = (out * 32767).astype(np.int16)
            outp = os.path.join(os.path.dirname(gp), name)
            try:
                wavfile.write(outp, sr, pcm)
            except Exception:
                outp = os.path.join(os.getcwd(), name)
                wavfile.write(outp, sr, pcm)
            return outp
        self.run_bg(work, self.apply_done)

    def apply_done(self, outp):
        self.busy = False
        self.set_status("Done!", OKC)
        self.out_lbl.config(text="Saved:\n%s\nOpen it with any music "
                                 "player." % outp)

    # ---------- FILE BROWSER ----------
    def build_browse(self):
        fr = tk.Frame(self.body, bg=BG)
        self.scr["browse"] = fr
        pad = self.pad
        self.browse_title = self.label(fr, "Choose a file", ACC, self.fbig,
                                       anchor="center")
        self.browse_title.pack(fill="x", pady=pad // 2)
        row = tk.Frame(fr, bg=BG)
        row.pack(fill="x", padx=pad)
        for i, (text, path) in enumerate((
                ("Download", "/storage/emulated/0/Download"),
                ("Music", "/storage/emulated/0/Music"),
                ("Phone", "/storage/emulated/0"))):
            b = self.button(row, text, lambda pth=path: self.browse_to(pth),
                            font=self.fsm)
            b.grid(row=0, column=i, sticky="ew", padx=2)
            row.grid_columnconfigure(i, weight=1, uniform="q")
        row = tk.Frame(fr, bg=BG)
        row.pack(fill="x", padx=pad, pady=pad // 2)
        self.path_var = tk.StringVar()
        tk.Entry(row, textvariable=self.path_var, font=self.fsm, bg=PANEL2,
                 fg=TXT, insertbackground=TXT, bd=0).pack(
            side="left", expand=True, fill="x", ipady=pad // 3)
        self.button(row, "Go", lambda: self.browse_to(self.path_var.get()),
                    font=self.fsm).pack(side="left", padx=(pad // 2, 0))
        self.browse_msg = self.label(fr, "", BAD, self.fsm)
        self.browse_msg.pack(fill="x", padx=pad)
        self.button(fr, "Cancel", lambda: self.show(self.browse_back)
                    ).pack(side="bottom", fill="x", padx=pad,
                           pady=pad // 2)
        self.blist = TouchList(fr, self.f, self.sbw, self.browse_pick)
        self.blist.pack(fill="both", expand=True, padx=pad)

    def open_browser(self, title, cb, back):
        self.browse_title.config(text=title)
        self.browse_cb = cb
        self.browse_back = back
        start = self.last_dir
        if not os.path.isdir(start):
            start = os.getcwd()
        self.browse_to(start)
        self.show("browse")

    def browse_to(self, path):
        path = os.path.abspath(path.strip().strip("\"'"))
        try:
            names = os.listdir(path)
        except Exception:
            self.browse_msg.config(
                text="Can't open that folder. On Android 11+ allow "
                     "Pydroid storage access.", fg=BAD)
            return
        self.browse_msg.config(text="")
        self.last_dir = path
        self.path_var.set(path)
        dirs, files = [], []
        for n in names:
            if n.startswith("."):
                continue
            full = os.path.join(path, n)
            if os.path.isdir(full):
                dirs.append(n)
            elif n.lower().endswith(AUDIO_EXT):
                files.append(n)
        dirs.sort(key=str.lower)
        files.sort(key=str.lower)
        self.entries = []
        parent = os.path.dirname(path)
        if parent and parent != path:
            self.entries.append((".. (go up)", parent, True))
        for n in dirs:
            self.entries.append(("[+] " + n, os.path.join(path, n), True))
        for n in files:
            self.entries.append((n, os.path.join(path, n), False))
        self.blist.set_items([e[0] for e in self.entries])
        if not (dirs or files):
            self.browse_msg.config(text="No folders or audio files here.",
                                   fg=MUT)

    def browse_pick(self, idx):
        if idx >= len(self.entries):
            return
        _, full, isdir = self.entries[idx]
        if isdir:
            self.browse_to(full)
        else:
            self.last_dir = os.path.dirname(full)
            cb, back = self.browse_cb, self.browse_back
            self.show(back)
            cb(full)


def main():
    try:
        root = tk.Tk()
    except Exception as ex:
        print("Could not open the window: %s" % ex)
        print("Run this inside the Pydroid 3 app (not Termux).")
        return
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()