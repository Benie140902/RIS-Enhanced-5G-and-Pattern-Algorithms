#!/usr/bin/env python3
# ris_excel_4ris_visualizer.py
import sys
import time
import serial
import numpy as np
import pandas as pd
import uhd
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QGridLayout, QPushButton, QLabel
from PyQt5.QtCore import QTimer
import os
import csv
import math

# --- Configuration ---
PORT1 = '/dev/ttyUSB4'  # RIS1 (bottom-left)
PORT2 = '/dev/ttyUSB3'  # RIS2 (bottom-right)
PORT3 = '/dev/ttyUSB2'  # RIS3 (top-right) -> mirrored RIS2
PORT4 = '/dev/ttyUSB0'  # RIS4 (top-left)  -> mirrored RIS1
BAUD_RATE = 115200
EXCEL_FILE = "pattern_power_log_16x32_4RIS.xlsx"
INPUT_EXCEL = "input_ris_patterns.xlsx"  # your input excel with 'Pattern' column
DELAY_BETWEEN_PATTERNS = 3000  # ms


# -----------------------------
# Port-generation algorithm
# -----------------------------
def generate_ports_numpy(A_input):
    """
    Implements the algorithm you described.
    Input: A_input = list/array of indices (0..15)
    Output: ris_port1 (np.array of indices), ris_port2 (np.array of indices)
    """
    A = np.array(A_input, dtype=int)
    original_size = len(A)
    A_expanded = np.concatenate([A, A + 1])
    B = []

    for i in range(original_size):
        v = 2 * int(A[i])
        if v <= 15:
            A_expanded = np.concatenate([A_expanded, np.array([v, v + 1])])
        else:
            B.extend([v, v + 1])

    ris_port1 = np.unique(A_expanded).astype(int)

    # ris_port2 derived from B (may contain values >15)
    ris_port2 = []
    for x in B:
        # divide by 2, if .5 -> ceil to next integer
        val = x / 2.0
        if val.is_integer():
            mapped = int(val)
        else:
            mapped = int(math.ceil(val))
        # clamp to 0..15
        if mapped < 0:
            mapped = 0
        if mapped > 15:
            mapped = 15
        ris_port2.append(mapped)
    if ris_port2:
        ris_port2 = np.unique(np.array(ris_port2, dtype=int))
    else:
        ris_port2 = np.array([], dtype=int)

    return ris_port1, ris_port2


# -----------------------------
# Helpers: hex parsing and expansion
# -----------------------------
def clean_pattern_string(s: str):
    s = str(s).strip().upper()
    if s.startswith("!"):
        s = s[1:]
    if s.startswith("0X"):
        s = s[2:]
    s = s.replace(" ", "")
    return s


def hex64_to_A_input(hex64):
    """
    Accept a 64-hex string representing 16 columns (4 hex chars per column).
    Returns list of column indices (0..15) considered ON (non-zero nibble).
    """
    if len(hex64) != 64:
        raise ValueError("hex64 must be 64 hex characters long (16 * 4 chars).")
    cols = [hex64[i*4:(i+1)*4] for i in range(16)]
    A_input = [i for i, chunk in enumerate(cols) if chunk != "0000"]
    return A_input


def build_16col_hex_from_ports(port_list):
    """
    port_list: iterable of unique ints in 0..15
    returns 64-hex string (16 chunks of 4 hex chars) where ON -> 'FFFF', OFF -> '0000'
    """
    ports = set(int(x) for x in port_list)
    chunks = []
    for col in range(16):
        chunks.append("FFFF" if col in ports else "0000")
    return ''.join(chunks)  # 64 chars


def expand_single_ris_hex_to_4ris(single_hex):
    """
    Input:
      single_hex: cleaned hex string (no ! or 0X). Accepts:
        - 64 hex chars (single RIS: 16 columns × 4 hex chars)
        - 128 hex chars (full RIS1+RIS2 combined) -> returned as-is
    Output:
      full_pattern: string like "!0X" + 128 hex chars (RIS1 64chars + RIS2 64chars)
      ris1_hex, ris2_hex, ris_port1, ris_port2_mapped
    """
    s = clean_pattern_string(single_hex)
    if len(s) == 128:
        # already combined: assume left 64 = ris1, right 64 = ris2
        ris1_hex = s[:64]
        ris2_hex = s[64:]
        full = "!0X" + ris1_hex + ris2_hex
        return full, ris1_hex, ris2_hex, [], []
    elif len(s) == 64:
        # single RIS: parse into A_input
        try:
            A_input = hex64_to_A_input(s)
        except ValueError as e:
            # fallback: treat every nibble separately (if weird length)
            A_input = []
            for i in range(16):
                chunk = s[i*4:(i+1)*4]
                if chunk != "0000":
                    A_input.append(i)

        ris_port1, ris_port2 = generate_ports_numpy(A_input)

        # ris_port2 mapping already done inside generate_ports_numpy (ceil + clamp)
        ris1_hex = build_16col_hex_from_ports(ris_port1)
        ris2_hex = build_16col_hex_from_ports(ris_port2)

        full = "!0X" + ris1_hex + ris2_hex
        return full, ris1_hex, ris2_hex, ris_port1.tolist(), ris_port2.tolist()
    else:
        # Try to salvage: if length == 16 or 32 (short forms) expand to equivalent nibble format
        if len(s) == 16:
            # each hex char -> one 4-bit column; expand to 4-char chunk by repeating
            expanded = ''.join([c*4 for c in s])
            return expand_single_ris_hex_to_4ris(expanded)
        elif len(s) == 32:
            # maybe already 32 nibbles -> 32*4? If ambiguous, reject
            raise ValueError("Unsupported pattern length (32). Expect 64 or 128 hex chars after cleaning.")
        else:
            raise ValueError(f"Unsupported pattern length: {len(s)} chars.")


# -----------------------------
# Read input Excel
# -----------------------------
def load_single_ris_patterns_from_excel(excel_path):
    """
    Loads the 'Pattern' column from the first sheet in excel_path.
    Returns list of cleaned pattern strings (raw cell values).
    """
    df = pd.read_excel(excel_path)
    if "Pattern" not in df.columns:
        # try second column by index if header missing (user said Pattern is 2nd column)
        # but prefer explicit column name
        raise ValueError("Excel must contain a column named 'Pattern'.")

    patterns = df["Pattern"].astype(str).tolist()
    cleaned = [clean_pattern_string(p) for p in patterns]
    return cleaned


# -----------------------------
# Hex -> 16x32 matrix
# -----------------------------
def hex_to_matrix_16x32(full_pattern):
    """
    full_pattern: starts with '!0X' followed by 128 hex chars (32 columns x 4 hex chars).
    returns 16x32 matrix of 0/1 ints (rows x cols).
    """
    if full_pattern.startswith("!0X"):
        hex_str = full_pattern[3:]
    else:
        hex_str = full_pattern
    if len(hex_str) != 128:
        raise ValueError(f"Expected 128 hex characters, got {len(hex_str)}")
    matrix = [[0]*32 for _ in range(16)]
    for col in range(32):
        nib = hex_str[col*4:(col+1)*4]
        bits = bin(int(nib, 16))[2:].zfill(16)
        for row in range(16):
            matrix[row][col] = int(bits[row])
    return matrix


# -----------------------------
# Main GUI + USRP + Serial class
# -----------------------------
class RISVisualizer(QWidget):
    def __init__(self, input_excel=INPUT_EXCEL):
        super().__init__()
        self.setWindowTitle("16x32 RIS Pattern Visualizer + Power Logger (4 RIS)")
        self.grid_size = (16, 32)
        self.pattern_index = 0
        self.pattern_power_log = []
        self.patterns = []  # will store full '!0X' + 128 hex patterns
        self.load_patterns_from_excel(input_excel)
        self.load_existing_excel()

        # Open serial ports (tolerant)
        def try_open(port):
            try:
                s = serial.Serial(port, BAUD_RATE, timeout=2)
                print(f"Opened serial port {port}")
                return s
            except Exception as e:
                print(f"Warning: couldn't open {port}: {e}")
                return None

        self.ser1 = try_open(PORT1)
        self.ser2 = try_open(PORT2)
        self.ser3 = try_open(PORT3)
        self.ser4 = try_open(PORT4)

        # init USRP (tolerant)
        try:
            self.init_usrp()
        except Exception as e:
            print("Warning: failed to init USRP:", e)
            self.usrp = None
            self.rx_stream = None

        self.max = -100
        self.settling_delay = 3
        self.samples_per_reading = 3
        self.var_thresh = 0.5

        self.csv_file = "max_power_log_4RIS.csv"
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Pattern Index", "Max Power (dB)", "Variation (dB)"])

        # UI setup
        self.layout = QVBoxLayout(self)
        self.grid_layout = QGridLayout()
        self.cells = []

        for row in range(16):
            row_cells = []
            for col in range(32):
                cell = QPushButton("")
                cell.setFixedSize(18, 18)
                cell.setEnabled(False)
                cell.setStyleSheet("background-color: white; border: 1px solid #ccc;")
                self.grid_layout.addWidget(cell, row, col)
                row_cells.append(cell)
            self.cells.append(row_cells)
        self.layout.addLayout(self.grid_layout)

        self.power_label = QLabel("Power: -- dB")
        self.layout.addWidget(self.power_label)

        # Timer drives pattern sending
        self.timer = QTimer()
        self.timer.timeout.connect(self.send_next_pattern)
        self.timer.start(DELAY_BETWEEN_PATTERNS)

    def load_patterns_from_excel(self, excel_path):
        try:
            raw_list = load_single_ris_patterns_from_excel(excel_path)
        except Exception as e:
            print("Error loading input Excel:", e)
            raw_list = []

        # Expand each input into 4-RIS full combined pattern
        expanded = []
        for idx, raw in enumerate(raw_list):
            try:
                full_pattern, ris1_hex, ris2_hex, rp1, rp2 = expand_single_ris_hex_to_4ris(raw)
                expanded.append(full_pattern)
                print(f"Loaded row {idx}: produced full pattern; ris1_ports={rp1}, ris2_ports={rp2}")
            except Exception as e:
                print(f"Skipping row {idx} due to error: {e}")

        self.patterns = expanded
        print(f"Total patterns loaded from Excel: {len(self.patterns)}")

    def load_existing_excel(self):
        try:
            df_existing = pd.read_excel(EXCEL_FILE)
            self.pattern_index = len(df_existing)
            print(f"Continuing from pattern index: {self.pattern_index}")
        except Exception:
            print("Starting fresh. No previous log found.")

    def init_usrp(self):
        self.usrp = uhd.usrp.MultiUSRP()
        self.usrp.set_rx_rate(3e6)
        self.usrp.set_rx_freq(uhd.types.TuneRequest(3.5e9))
        self.usrp.set_rx_gain(80)
        stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
        self.rx_stream = self.usrp.get_rx_stream(stream_args)
        self.num_samps = 4096
        self.recv_buffer = np.zeros(self.num_samps, dtype=np.complex64)
        self.rx_md = uhd.types.RXMetadata()
        cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        cmd.stream_now = True
        self.rx_stream.issue_stream_cmd(cmd)

    def measure_power(self):
        if self.rx_stream is None:
            return -np.inf
        samples = []
        t_start = time.time()
        while time.time() - t_start < 1.0:
            try:
                n = self.rx_stream.recv(self.recv_buffer, self.rx_md, timeout=1.0)
                if n > 0:
                    samples.append(np.copy(self.recv_buffer[:n]))
            except Exception as e:
                print("rx recv error:", e)
                break
        if samples:
            all_samples = np.concatenate(samples)
            power_linear = np.mean(np.abs(all_samples) ** 2)
            return 10 * np.log10(power_linear) if power_linear > 0 else -np.inf
        else:
            return -np.inf

    def send_next_pattern(self):
        if not self.patterns:
            print("No patterns to send from Excel.")
            self.timer.stop()
            return

        if self.pattern_index >= len(self.patterns):
            print("All patterns sent.")
            self.timer.stop()
            return

        full_pattern = self.patterns[self.pattern_index]
        if len(full_pattern[3:]) != 128:
            print(f"Skipping invalid pattern (not 128 hex after prefix): {full_pattern}")
            self.pattern_index += 1
            return

        ris1_hex = full_pattern[3:3+64]   # bottom-left
        ris2_hex = full_pattern[3+64:3+128]  # bottom-right

        # Safe write
        def safe_write(ser, data):
            if ser:
                try:
                    ser.write((data + '\n').encode())
                except Exception as e:
                    print("Serial write error:", e)

        safe_write(self.ser1, "!0X" + ris1_hex)
        safe_write(self.ser2, "!0X" + ris2_hex)
        safe_write(self.ser3, "!0X" + ris2_hex)
        safe_write(self.ser4, "!0X" + ris1_hex)

        print(f"Pattern {self.pattern_index} sent to 4 RIS")
        try:
            matrix = hex_to_matrix_16x32(full_pattern)
            self.update_grid(matrix)
        except Exception as e:
            print("Grid update error:", e)

        # Allow RIS settle
        time.sleep(self.settling_delay)

        power_readings = []
        for i in range(self.samples_per_reading):
            val = self.measure_power()
            print(f"   [{i+1}] Power: {val:.2f} dB")
            power_readings.append(val)
            time.sleep(2)

        power = round(sum(power_readings) / len(power_readings), 2) if power_readings else -np.inf
        var = round(max(power_readings) - min(power_readings), 2) if power_readings else 0.0

        if power > self.max and var <= self.var_thresh:
            self.max = power
            with open(self.csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([self.pattern_index, self.max, var])

        self.power_label.setText(f"Power: {power:.2f} dB")
        print(f"Pattern Index {self.pattern_index} Measured Power: {power:.2f} dB, Max so far: {self.max}")

        log_entry = pd.DataFrame([{"Index": self.pattern_index, "Pattern": full_pattern, "Power (dB)": power, "Variation": var}])
        try:
            existing = pd.read_excel(EXCEL_FILE)
            updated = pd.concat([existing, log_entry], ignore_index=True)
        except Exception:
            updated = log_entry
        updated.to_excel(EXCEL_FILE, index=False)
        self.pattern_index += 1

    def update_grid(self, matrix):
        for r in range(16):
            for c in range(32):
                color = "green" if matrix[r][c] else "white"
                self.cells[r][c].setStyleSheet(f"background-color: {color};")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RISVisualizer(input_excel=INPUT_EXCEL)
    window.show()
    sys.exit(app.exec_())

