#!/usr/bin/env python3
"""
End-to-end RX: capture -> RRC matched filter -> timing offset search -> phase correction ->
EVM calculation -> constellation plot -> optional save to Excel.

Requirements:
  - python-uhd (libpyuhd), numpy, scipy, matplotlib, pandas (if saving to xlsx)
Run on the same machine with USRP B210 attached.
"""
import time
import uhd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
import pandas as pd  # optional (only used if save_xlsx=True)

# ---------------- USER PARAMETERS ----------------
center_freq      = 3.5e9    # Hz
sample_rate      = 3e6      # Hz
symbol_rate      = 2.5e5    # 250 kSym/s
gain             = 70       # RX gain (dB) - tune to avoid clipping
capture_time     = 0.15     # seconds to capture (increase for more symbols)
rolloff          = 0.35     # RRC roll-off
span_symbols     = 8        # RRC span in symbols (recommended 6..12)
chan             = 0        # USRP channel index (0 or 1). Use 1 for Rx2 if desired.
save_xlsx        = True     # True -> save raw/processed/demod sheets to rx_results.xlsx
num_symbols_to_save = 5000  # how many symbols to write to Excel (if available)
# -------------------------------------------------

# Derived
sps_float = sample_rate / symbol_rate
sps = int(round(sps_float))
if abs(sps - sps_float) > 1e-6:
    print(f"Warning: sample_rate/symbol_rate = {sps_float:.6f}, rounding sps -> {sps}")
if sps < 1:
    raise ValueError("Computed sps < 1; check sample_rate and symbol_rate")

num_samples = int(np.ceil(capture_time * sample_rate))
print(f"Parameters: fs={sample_rate:.0f} Hz, sym={symbol_rate:.0f} Sym/s, sps={sps}, capture_time={capture_time}s")
print(f"Capturing num_samples = {num_samples}")

# ---------------- RRC generator ----------------
def rrc_taps(sps, alpha, span):
    """
    Generate root-raised-cosine taps.
    length = span*sps + 1
    """
    N = span * sps
    t = np.arange(-N/2, N/2 + 1, dtype=float) / float(sps)  # in symbol times
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            h[i] = 1.0 - alpha + (4*alpha/np.pi)
        elif np.isclose(abs(ti), 1/(4*alpha)):
            # limiting value
            h[i] = (alpha/np.sqrt(2)) * (
                (1 + 2/np.pi) * np.sin(np.pi/(4*alpha)) +
                (1 - 2/np.pi) * np.cos(np.pi/(4*alpha))
            )
        else:
            num = (np.sin(np.pi*ti*(1 - alpha)) +
                   4*alpha*ti*np.cos(np.pi*ti*(1 + alpha)))
            den = (np.pi*ti*(1 - (4*alpha*ti)**2))
            h[i] = num / den
    # Normalize energy (approx) so symbol energy ~1 after matched filtering + sampling
    h = h / np.sqrt(np.sum(h**2))
    return h.astype(np.float64)

# ----------------- Setup USRP -------------------
usrp = uhd.usrp.MultiUSRP()
usrp.set_rx_dc_offset(True, 0)
usrp.set_rx_iq_balance(True, 0)
print("Configuring USRP...")
usrp.set_rx_rate(float(sample_rate), chan)
usrp.set_rx_freq(float(center_freq), chan)
usrp.set_rx_gain(float(gain), chan)
try:
    # choose antenna if supported (B210 front-end)
    usrp.set_rx_antenna("RX2", chan)
except Exception:
    pass
time.sleep(0.05)

# Stream args (Python UHD)
stream_args = uhd.usrp.StreamArgs("fc32","sc16")
stream_args.channels = [chan]
rx_streamer = usrp.get_rx_stream(stream_args)

# allocate buffer and metadata
buff = np.zeros(num_samples, dtype=np.complex64)
md = uhd.types.RXMetadata()

# Issue stream command (finite acquisition)
stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
stream_cmd.num_samps = num_samples
stream_cmd.stream_now = True
rx_streamer.issue_stream_cmd(stream_cmd)

print("Starting capture... (USRP RX LED should blink)")
t0 = time.time()
num_rx = rx_streamer.recv(buff, md, timeout=5.0 + capture_time)
t1 = time.time()
print(f"Received {num_rx} samples in {t1-t0:.3f} s; metadata error_code={md.error_code}")

if num_rx <= 0:
    raise RuntimeError(f"No samples received from USRP (num_rx={num_rx}). Check antenna/gain/connection.")

samples = buff[:num_rx].astype(np.complex64)

# health checks
peak = np.max(np.abs(samples)) if len(samples) else 0.0
print(f"Peak amplitude: {peak:.3f}")
if peak > 0.95:
    print("⚠️ Clipping likely: reduce RX gain.")
if peak < 0.01:
    print("⚠️ Very low signal level: increase RX gain or check connection.")

# ----------------- Matched filter (RRC) ----------------
h_rrc = rrc_taps(sps, rolloff, span_symbols)
filtered = lfilter(h_rrc, 1.0, samples)

# drop transient (half-filter length)
drop = len(h_rrc)//2
if drop >= len(filtered):
    raise RuntimeError("Capture too short relative to filter transient. Increase capture_time.")
filtered = filtered[drop:]

# ----------------- Symbol timing offset search ----------------
# We don't implement full Gardner here; instead search offsets 0..sps-1 and pick offset with max average magnitude
best_off = 0
best_metric = -np.inf
for off in range(sps):
    sym = filtered[off::sps]
    if len(sym) < 10:
        continue
    metric = np.mean(np.abs(sym))  # energy metric; cluster magnitude should be larger when sampling at symbol center
    if metric > best_metric:
        best_metric = metric
        best_off = off
print(f"Best downsample offset = {best_off} (metric={best_metric:.4f})")

rx_symbols = filtered[best_off::sps]
# trim to integer number of symbols requested (if capture produced many)
if len(rx_symbols) > int(num_samples//sps):
    rx_symbols = rx_symbols[:int(num_samples//sps)]
print(f"Recovered {len(rx_symbols)} symbol-spaced samples")

# ----------------- Normalize and coarse CPE (4th-power) ----------------
rx_symbols = rx_symbols - np.mean(rx_symbols)   # DC remove
rms = np.sqrt(np.mean(np.abs(rx_symbols)**2)) or 1.0
rx_symbols = rx_symbols / rms

# coarse common phase error via 4th-power
cpe = 0.25 * np.angle(np.mean(rx_symbols**4))
rx_symbols = rx_symbols * np.exp(-1j * cpe)

# ----------------- Decision-directed phase refine (short loop) ----------------
# small decision-directed phase correction to tighten clusters
const_pts = np.array([(1+1j), (1-1j), (-1+1j), (-1-1j)], dtype=np.complex64) / np.sqrt(2)
phase = 0.0
freq = 0.0
mu = 0.05      # phase loop gain (tune if needed)
beta = 0.0001  # freq loop gain
out = np.zeros_like(rx_symbols)
for i, x in enumerate(rx_symbols):
    v = x * np.exp(-1j*phase)
    out[i] = v
    # make a soft decision to ideal nearest
    idx = np.argmin(np.abs(v - const_pts))
    ref = const_pts[idx]
    # phase error (angle between v and ref)
    err = np.angle(v * np.conj(ref))
    freq += beta * err
    phase += freq + mu * err
rx_corr = out

# ----------------- Decision mapping & EVM ----------------
# nearest mapping
dists = np.abs(rx_corr[None, :] - const_pts[:, None])
idxs = np.argmin(dists, axis=0)
s_ref = const_pts[idxs]
errvec = rx_corr - s_ref

evm_rms = np.sqrt(np.mean(np.abs(errvec)**2))
evm_pct = 100.0 * evm_rms
evm_db = 20.0 * np.log10(evm_rms + 1e-15)

print(f"EVM (QPSK): {evm_pct:.2f}%   ({evm_db:.2f} dB)")
print(f"Symbols used: {len(rx_corr)}  | dropped transient samples: {drop}")

# ----------------- Plot constellation ----------------
plt.figure(figsize=(6,6))
plt.scatter(np.real(rx_corr), np.imag(rx_corr), s=8, alpha=0.45, label="Received (post-RRC & phase)")
plt.scatter(np.real(const_pts), np.imag(const_pts), c='red', marker='x', s=160, label="Ideal QPSK")
plt.axhline(0, color='gray', lw=0.6)
plt.axvline(0, color='gray', lw=0.6)
plt.title(f"QPSK Constellation (EVM {evm_pct:.2f}%)")
plt.xlabel("I")
plt.ylabel("Q")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.show()

# ----------------- Save results to Excel (optional) ----------------
if save_xlsx:
    out_symbols = min(len(rx_corr), num_symbols_to_save)
    df_raw = pd.DataFrame({"I_raw": samples.real[:out_symbols*sps], "Q_raw": samples.imag[:out_symbols*sps]})
    df_proc = pd.DataFrame({"I_proc": rx_corr.real[:out_symbols], "Q_proc": rx_corr.imag[:out_symbols]})
    df_demod = pd.DataFrame({"I_demod": s_ref.real[:out_symbols], "Q_demod": s_ref.imag[:out_symbols], "SymIdx": idxs[:out_symbols]})
    with pd.ExcelWriter("rx_results.xlsx") as writer:
        df_raw.to_excel(writer, sheet_name="Raw_USRP", index=False)
        df_proc.to_excel(writer, sheet_name="Processed_IQ", index=False)
        df_demod.to_excel(writer, sheet_name="Demod_Symbols", index=False)
    print(f"Saved results to rx_results.xlsx (first {out_symbols} symbols)")

# --------------- done ---------------
