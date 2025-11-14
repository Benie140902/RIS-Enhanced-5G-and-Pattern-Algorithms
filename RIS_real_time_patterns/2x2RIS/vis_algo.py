#!/usr/bin/env python3
import sys
import math
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QGridLayout, QPushButton,
    QLabel, QLineEdit, QHBoxLayout, QGroupBox
)
from PyQt5.QtCore import Qt


# ======================================================
# === Algorithm (Same as your final 4-RIS generation) ===
# ======================================================

def clean_hex(s):
    s = str(s).strip().upper()
    if s.startswith("!"):
        s = s[1:]
    if s.startswith("0X"):
        s = s[2:]
    return s.replace(" ", "")


def generate_ports_numpy(A_input):
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

    ris_port2 = []
    for x in B:
        val = x / 2
        mapped = math.ceil(val)
        mapped = max(0, min(15, mapped))
        ris_port2.append(mapped)

    if ris_port2:
        ris_port2 = np.unique(np.array(ris_port2, dtype=int))
    else:
        ris_port2 = np.array([], dtype=int)

    return ris_port1, ris_port2


def hex64_to_A_input(hex64):
    cols = [hex64[i*4:(i+1)*4] for i in range(16)]
    A = [i for i, chunk in enumerate(cols) if chunk != "0000"]
    return A


def build_hex_16cols(ports):
    ports = set(ports)
    out = []
    for i in range(16):
        out.append("FFFF" if i in ports else "0000")
    return "".join(out)


def hex64_to_matrix16(hex64):
    matrix = [[0] * 16 for _ in range(16)]
    for col in range(16):
        nib = hex64[col * 4:(col + 1) * 4]
        bits = bin(int(nib, 16))[2:].zfill(16)
        for row in range(16):
            matrix[row][col] = int(bits[row])
    return matrix


# ======================================================
# =================== GUI APPLICATION ==================
# ======================================================
class SimpleRISVisualizer(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("4-RIS Pattern Generator (Test Mode Only)")
        self.layout = QVBoxLayout(self)

        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Enter 64-hex single RIS pattern")
        self.layout.addWidget(self.input_box)

        self.btn = QPushButton("Generate 4-RIS Patterns")
        self.btn.clicked.connect(self.generate)
        self.layout.addWidget(self.btn)

        # Ports output
        self.port_label = QLabel("RIS1 ports: []\nRIS2 ports: []")
        self.layout.addWidget(self.port_label)

        # Four RIS grids
        self.ris_grids = []
        ris_names = ["RIS1", "RIS2", "RIS3 (mirror of RIS2)", "RIS4 (mirror of RIS1)"]
        grid_layout = QHBoxLayout()

        for name in ris_names:
            gbox = QGroupBox(name)
            gl = QGridLayout()
            cells = []
            for r in range(16):
                row_cells = []
                for c in range(16):
                    cell = QLabel()
                    cell.setFixedSize(15, 15)
                    cell.setStyleSheet("background:white;border:1px solid #ccc;")
                    gl.addWidget(cell, r, c)
                    row_cells.append(cell)
                cells.append(row_cells)
            gbox.setLayout(gl)
            grid_layout.addWidget(gbox)
            self.ris_grids.append(cells)

        self.layout.addLayout(grid_layout)

    def update_grid(self, grid_cells, matrix):
        for r in range(16):
            for c in range(16):
                color = "green" if matrix[r][c] == 1 else "white"
                grid_cells[r][c].setStyleSheet(f"background:{color};border:1px solid #ccc;")

    def generate(self):
        raw = clean_hex(self.input_box.text())
        if len(raw) != 64:
            self.port_label.setText("❌ Input must be 64 hex characters (one RIS).")
            return

        A_in = hex64_to_A_input(raw)
        ris1_ports, ris2_ports = generate_ports_numpy(A_in)

        ris1_hex = build_hex_16cols(ris1_ports)
        ris2_hex = build_hex_16cols(ris2_ports)

        ris1_mat = hex64_to_matrix16(ris1_hex)
        ris2_mat = hex64_to_matrix16(ris2_hex)

        # mirror
        ris3_mat = ris2_mat
        ris4_mat = ris1_mat

        self.update_grid(self.ris_grids[0], ris1_mat)
        self.update_grid(self.ris_grids[1], ris2_mat)
        self.update_grid(self.ris_grids[2], ris3_mat)
        self.update_grid(self.ris_grids[3], ris4_mat)

        self.port_label.setText(
            f"RIS1 ports: {ris1_ports.tolist()}\nRIS2 ports: {ris2_ports.tolist()}"
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = SimpleRISVisualizer()
    w.show()
    sys.exit(app.exec_())

