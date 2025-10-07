#!/usr/bin/env python3


import sys
import time
import serial
import numpy as np
import uhd
import pandas as pd
from scipy.signal import lfilter
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QPushButton, QVBoxLayout, QLabel
from PyQt5.QtCore import QTimer, Qt
import os
import csv

# ---------------- User params (tweak as needed) ----------------
PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200
DELAY_BETWEEN_PATTERNS = 100  # milliseconds (QTimer)
CENTER_FREQ = 3.5e9
SAMPLE_RATE = 3e6
SYMBOL_RATE = 2.5e5      # QPSK symbol rate (250kSym/s)
RRC_ROLLOFF = 0.35
RRC_SPAN = 12            # symbols
RX_GAIN = 80
SETTLING_DELAY = 1.5     # seconds to wait after sending pattern
SAMPLES_PER_READING = 3  # number of repeats per pattern (you already had 3)
NUM_SAMPS_RX = 4096
VAR_THRESH = 0.5         # threshold for variation when updating max log
CSV_FILE = "max_power_log.csv"
PATTERN_LOG_XLSX = "pattern_power_log.xlsx"
PATTERN_LOG_SORTED_XLSX = "pattern_power_log_sorted.xlsx"
# ---------------------------------------------------------------

# ---------------- RRC + EVM helper functions -------------------
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
    # Normalize energy
    h = h / np.sqrt(np.sum(h**2))
    return h.astype(np.float64)

def compute_evm(samples, sample_rate=SAMPLE_RATE, symbol_rate=SYMBOL_RATE,
                rolloff=RRC_ROLLOFF, span_symbols=RRC_SPAN):
    """
    Compute EVM (percentage and dB) from raw complex samples.
    Returns (evm_pct, evm_db) or (None, None) if insufficient data.
    Assumes QPSK constellation.
    """
    if samples is None or len(samples) < 100:
        return None, None

    # Convert to numpy complex64
    samples = np.asarray(samples, dtype=np.complex64)

    # Compute sps
    sps_float = sample_rate / symbol_rate
    sps = int(round(sps_float))
    if sps < 1:
        return None, None

    # Matched filter (RRC)
    h_rrc = rrc_taps(sps, rolloff, span_symbols)
    filtered = lfilter(h_rrc, 1.0, samples)

    # Drop transient (half filter length)
    drop = len(h_rrc) // 2
    if drop >= len(filtered):
        return None, None
    filtered = filtered[drop:]

    # Symbol timing offset search (energy metric)
    best_off = 0
    best_metric = -np.inf
    for off in range(sps):
        sym = filtered[off::sps]
        if len(sym) < 10:
            continue
        metric = np.mean(np.abs(sym))
        if metric > best_metric:
            best_metric = metric
            best_off = off

    rx_symbols = filtered[best_off::sps]
    if len(rx_symbols) < 16:
        return None, None

    # Normalize and coarse CPE (4th power)
    rx_symbols = rx_symbols - np.mean(rx_symbols)
    rms = np.sqrt(np.mean(np.abs(rx_symbols)**2)) or 1.0
    rx_symbols = rx_symbols / rms
    cpe = 0.25 * np.angle(np.mean(rx_symbols**4))
    rx_symbols = rx_symbols * np.exp(-1j * cpe)

    # Decision-directed phase refine (small loop)
    const_pts = np.array([(1+1j), (-1+1j), (-1-1j), (1-1j)], dtype=np.complex64) / np.sqrt(2)
    phase = 0.0
    freq = 0.0
    mu = 0.05
    beta = 0.0001
    out = np.zeros_like(rx_symbols)
    for i, x in enumerate(rx_symbols):
        v = x * np.exp(-1j * phase)
        out[i] = v
        idx = np.argmin(np.abs(v - const_pts))
        ref = const_pts[idx]
        err = np.angle(v * np.conj(ref))
        freq += beta * err
        phase += freq + mu * err
    rx_corr = out

    # Decision mapping and EVM
    dists = np.abs(rx_corr[None, :] - const_pts[:, None])
    idxs = np.argmin(dists, axis=0)
    s_ref = const_pts[idxs]
    errvec = rx_corr - s_ref
    evm_rms = np.sqrt(np.mean(np.abs(errvec)**2))
    evm_pct = 100.0 * evm_rms
    evm_db = 20.0 * np.log10(evm_rms + 1e-15)

    return evm_pct, evm_db

# ---------------- Patterns generation (unchanged) ----------------
def generate_column_on_off_patterns():
    patterns = []
    for i in range(2**16):
        binary = bin(i)[2:].zfill(16)
        hex_pattern = "!0X"
        for bit in binary:
            hex_pattern += "FFFF" if bit == '1' else "0000"
        patterns.append(hex_pattern)
    return patterns

def hex_to_matrix(hex_str):
    hex_str = hex_str[3:]  # Remove !0X
    matrix = [[0] * 16 for _ in range(16)]
    for col in range(16):
        bits = bin(int(hex_str[col*4:(col+1)*4], 16))[2:].zfill(16)
        for row in range(16):
            matrix[row][col] = int(bits[row])
    return matrix

patterns = generate_column_on_off_patterns()

# ---------------- Main UI class (pattern sender) ----------------
class PatternSender(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("16x16 RIS Pattern Visualizer + Power Meter + EVM")
        self.resize(500, 550)
        self.grid_size = 16
        self.pattern_index = 0
        self.rx_gain = RX_GAIN
        self.ser = serial.Serial(PORT, BAUD_RATE, timeout=2)
        self.patterns = patterns
        self.init_usrp()
        self.init_ui()
        self.pattern_power_log = []
        self.max = -100.0
        self.settling_delay = SETTLING_DELAY
        self.avg_power = 0
        self.variation = 0
        self.samples_per_reading = SAMPLES_PER_READING
        self.start_time_global = time.time()
        self.last_pattern_time = time.time()

        # Timer for iterating patterns
        self.timer = QTimer()
        self.timer.timeout.connect(self.send_next_pattern)
        self.timer.start(DELAY_BETWEEN_PATTERNS)

        self.var_thresh = VAR_THRESH

        self.csv_file = CSV_FILE
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Pattern Index", "Pattern", "Max Power (dB)", "Variation (dB)", "EVM (%)", "EVM (dB)"])

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.grid_layout = QGridLayout()
        self.cells = []
        for row in range(self.grid_size):
            row_cells = []
            for col in range(self.grid_size):
                cell = QPushButton("")
                cell.setFixedSize(25, 25)
                cell.setStyleSheet("background-color: white; border: 1px solid #ccc;")
                cell.setEnabled(False)
                self.grid_layout.addWidget(cell, row, col)
                row_cells.append(cell)
            self.cells.append(row_cells)
        layout.addLayout(self.grid_layout)
        self.power_label = QLabel("Power: -- dB | EVM: --")
        layout.addWidget(self.power_label)
        self.setLayout(layout)

    def init_usrp(self):
        # Initialize USRP (keep your original settings)
        self.usrp = uhd.usrp.MultiUSRP()
        self.usrp.set_rx_rate(float(SAMPLE_RATE))
        try:
            # Accept TuneRequest if available in this UHD version
            self.usrp.set_rx_freq(uhd.types.TuneRequest(CENTER_FREQ))
        except Exception:
            # fallback
            self.usrp.set_rx_freq(float(CENTER_FREQ))
        self.usrp.set_rx_gain(float(self.rx_gain))
        stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
        self.rx_stream = self.usrp.get_rx_stream(stream_args)
        self.num_samps = NUM_SAMPS_RX
        self.recv_buffer = np.zeros(self.num_samps, dtype=np.complex64)
        self.rx_md = uhd.types.RXMetadata()
        # Start continuous streaming
        cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        cmd.stream_now = True
        self.rx_stream.issue_stream_cmd(cmd)

    def send_next_pattern(self):
        # End condition
        if self.pattern_index >= len(self.patterns):
            self.timer.stop()
            try:
                self.ser.close()
            except Exception:
                pass
            # Save final logs
            df = pd.DataFrame(self.pattern_power_log)
            if not df.empty:
                df_sorted = df.sort_values(by="Power (dB)", ascending=False)
                df_sorted.to_excel(PATTERN_LOG_SORTED_XLSX, index=False)
                df.to_excel(PATTERN_LOG_XLSX, index=False)
                print(f"Saved sorted results to {PATTERN_LOG_SORTED_XLSX} and all results to {PATTERN_LOG_XLSX}")
            # stop stream
            try:
                self.rx_stream.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))
            except Exception:
                pass
            print("Finished sending all patterns.")
            return

        pattern = self.patterns[self.pattern_index]
        print(f"\nSending Pattern {self.pattern_index + 1}/{len(self.patterns)}:\n{pattern}")
        try:
            self.ser.write((pattern + '\n').encode())
        except Exception as e:
            print(f"Serial write error: {e}")

        try:
            response = self.ser.readline().decode().strip()
            print(f"RIS Response: {response}")
        except Exception as e:
            print(f"No response or decode error: {e}")

        matrix = hex_to_matrix(pattern)
        self.update_grid(matrix)

        # settling delay
        time.sleep(self.settling_delay)

        # collect multiple readings (power + evm)
        power_readings = []
        evm_readings = []
        for i in range(self.samples_per_reading):
            pwr, evm_pct, evm_db = self.measure_power()
            if pwr is None:
                pwr = -np.inf
            print(f"   [{i+1}] Power: {pwr:.2f} dB | EVM: {evm_pct if evm_pct is not None else 'NA'}% ({evm_db if evm_db is not None else 'NA'} dB)")
            power_readings.append(pwr)
            if evm_pct is not None:
                evm_readings.append((evm_pct, evm_db))
            time.sleep(0.5)  # small pause between repeats

        # Aggregate readings
        if len(power_readings) > 0:
            finite_powers = [p for p in power_readings if np.isfinite(p)]
            if len(finite_powers) == 0:
                power_db = -np.inf
                var = np.inf
            else:
                power_db = round(sum(finite_powers) / len(finite_powers), 2)
                var = round(max(finite_powers) - min(finite_powers), 2)
        else:
            power_db = -np.inf
            var = np.inf

        if evm_readings:
            evm_avg_pct = float(np.mean([e[0] for e in evm_readings]))
            evm_avg_db = float(np.mean([e[1] for e in evm_readings]))
        else:
            evm_avg_pct = None
            evm_avg_db = None

        # Update max log if better and variation acceptable
        if np.isfinite(power_db) and power_db > self.max and var <= self.var_thresh:
            self.max = power_db
            self.variation = var
            with open(self.csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([self.pattern_index, pattern, self.max, self.variation, evm_avg_pct, evm_avg_db])

        # UI & console updates
        evm_display = f"{evm_avg_pct:.2f}% ({evm_avg_db:.2f} dB)" if evm_avg_pct is not None else "NA"
        self.power_label.setText(f"Power: {power_db:.2f} dB | EVM: {evm_display}")
        print(f"Pattern Index {self.pattern_index} Measured Signal Power: {power_db:.2f} dB | Variation: {var:.2f} dB | EVM avg: {evm_display}")
        print(f"Max power found till now {self.max} with {self.variation} dB variation")

        # append to pattern log and save workbook
        self.pattern_power_log.append({
            "Pattern Index": self.pattern_index,
            "Pattern": pattern,
            "Power (dB)": power_db,
            "Variation": var,
            "EVM (%)": evm_avg_pct,
            "EVM (dB)": evm_avg_db
        })
        try:
            df = pd.DataFrame(self.pattern_power_log)
            df.to_excel(PATTERN_LOG_XLSX, index=False)
        except Exception as e:
            print(f"Failed to write excel: {e}")

        # time accounting (same as original)
        now = time.time()
        time_per_pattern = now - self.last_pattern_time
        self.last_pattern_time = now
        time_passed = now - self.start_time_global
        try:
            estimated_total = (time_passed / (self.pattern_index + 1)) * len(self.patterns)
            print(f"Time current this pattern: {time_per_pattern:.2f} s")
            print(f"Time passed: {time_passed/3600:.2f} hrs")
            print(f"Estimated total time: {estimated_total/3600:.2f} hrs")
        except Exception:
            pass

        self.pattern_index += 1

    def update_grid(self, matrix):
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                color = "green" if matrix[row][col] else "white"
                self.cells[row][col].setStyleSheet(f"background-color: {color}; border: 1px solid #ccc;")

    def save_grid_image(self, index):
        # kept for compatibility though not requested
        pixmap = QPixmap(self.grid_layout.sizeHint())
        pixmap.fill(Qt.white)
        painter = QPainter(pixmap)
        self.grid_layout.parentWidget().render(painter)
        power_db = self.pattern_power_log[-1]['Power (dB)'] if self.pattern_power_log else 0
        painter.setPen(Qt.black)
        painter.setFont(self.font())
        painter.drawText(10, 20, f"Power: {power_db:.3f} dB")
        painter.end()
        filename = f"{power_db:.3f}_dB.png"
        pixmap.save(filename)

    def measure_power(self):
        """
        Collect raw samples for up to 1 second (non-blocking recv loop),
        return (power_dB, evm_pct, evm_db).
        """
        samples_list = []
        t_start = time.time()
        # collect until 1 second or until we have some data
        while time.time() - t_start < 1.0:
            try:
                n = self.rx_stream.recv(self.recv_buffer, self.rx_md, timeout=1.0)
            except Exception as e:
                # recv error -> break and process what we have
                print(f"recv error: {e}")
                break
            if n > 0:
                samples_list.append(np.copy(self.recv_buffer[:n]))
            # small yield to avoid busy lock
            if len(samples_list) > 0:
                # if we already have at least one buffer, break early to reduce latency
                break

        if not samples_list:
            return None, None, None

        all_samples = np.concatenate(samples_list)
        # compute power (linear mean of |x|^2)
        power_linear = np.mean(np.abs(all_samples) ** 2) if len(all_samples) > 0 else 0.0
        power_db = 10 * np.log10(power_linear + 1e-12) if power_linear > 0 else -np.inf

        # compute EVM on the captured block (may return None if too short)
        evm_pct, evm_db = compute_evm(all_samples, sample_rate=SAMPLE_RATE, symbol_rate=SYMBOL_RATE,
                                      rolloff=RRC_ROLLOFF, span_symbols=RRC_SPAN)

        return power_db, evm_pct, evm_db

# ----------------- Run app -----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PatternSender()
    window.show()
    sys.exit(app.exec_())
