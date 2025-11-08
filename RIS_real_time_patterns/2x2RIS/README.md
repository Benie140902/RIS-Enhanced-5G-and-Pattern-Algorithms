# 2x2 RIS Controller and Power Measurement System

This project implements a **4-RIS controller** and **power measurement system** using Python, UHD (for USRP), and serial communication (for RIS control).  
It automates the transmission of reflection patterns to multiple RIS blocks and measures received signal power through a connected **USRP B210**.

---

## System Overview

The setup consists of:
- **4 RIS panels**, each 16×16 elements, controlled through serial ports of RIS controller.
- **1 USRP B210 SDR** as the receiver.
- A **Python GUI** that:
  - Sends reflection patterns to all 4 RIS units.
  - Waits for a settling delay.
  - Collects received samples via the USRP.
  - Computes and logs the measured power and variation.
  - Updates a live visual grid of the current RIS pattern.

---

## Features

- Automated sequential pattern transmission to all 4 RIS panels.
- Real-time power measurement using the USRP B210.
- GUI visualization of active RIS elements (16×32 grid view).
- Automatic logging of power readings and variation into:
  - **CSV file** (`power_log.csv`)
  - **Excel file** (`patterns_log.xlsx`)
- Segmentation-safe UHD streaming.
- Configurable parameters (delay, samples per reading, serial ports, etc.)

---

## Hardware Requirements

| Component | Description |
|------------|--------------|
| **USRP B210** | SDR used for received power measurement |
| **4 RIS Boards** | Each 16×16 array, controlled via serial USB |
| **Host PC** | Ubuntu 22.04 or later |
| **Cables** | 4 × USB cables for RIS, 1 × USB 3.0 for USRP |

---

##  Software Requirements

| Package | Version / Notes |
|----------|------------------|
| **Python** | ≥ 3.10 |
| **NumPy** | `< 2.0` (use 1.26.x to avoid `_ARRAY_API` errors) |
| **pandas** | For logging |
| **pyserial** | For communication with RIS |
| **PyQt5** | For GUI visualization |
| **uhd** | Ettus UHD Python bindings |
| **openpyxl** | For Excel file export |

### Install Dependencies
```bash
sudo apt update
sudo apt install libuhd-dev uhd-host python3-pyqt5 python3-pandas python3-serial python3-pip
pip install "numpy<2" openpyxl
