RIS CQI Monitoring and Pattern Visualization System

This project provides a real-time monitoring and visualization framework for evaluating Reconfigurable Intelligent Surface (RIS) reflection patterns in a 5G testbed.
It integrates serial communication, UDP-based CQI data collection, and PyQt5 visualization to find optimal RIS configurations that maximize Channel Quality Indicator (CQI).

Components Overview
1. cqi_pattern.py

A PyQt5-based GUI tool that:

Reads RIS pattern configurations (hexadecimal reflection matrices) from an Excel sheet.

Sends these patterns via serial port to the RIS controller.

Receives real-time CQI data via UDP from the gNB or measurement system.

Displays:

RIS element activation pattern in a 16×16 grid.

Live CQI plot with threshold marking.

Automatically switches to the next pattern when CQI drops below a threshold or user manually clicks "Next Pattern".

2. real_cqi_data.py

A real-time CQI and Data Rate logger that:

Listens on the same UDP port as the GUI to collect CQI and data-rate metrics in JSON format.

Extracts relevant values recursively from JSON packets.

Plots CQI and Data Rate (Kbps) dynamically using matplotlib.animation.

Optionally logs all data points into an Excel sheet (cqi_log_pos.xlsx).

3. RIS_gnb_parameters

A configuration file (text or parameter-based) that stores gNB or RIS setup parameters — such as RIS serial ports, gNB IP/port details, and power levels — used by both scripts to ensure synchronization between RIS hardware and the network side.

System Flow
+-------------------+       UDP: CQI data       +-------------------+
|    gNB / UE       |  ---------------------->  |   Python Scripts   |
|  (CQI Generator)  |                          |  (cqi_pattern.py &  |
|                   |                          |  real_cqi_data.py)  |
+-------------------+                          +-------------------+
          ^                                                 |
          |     Serial: RIS Control Commands                |
          +-------------------------------------------------+
                              |
                              v
                      +-----------------+
                      |     RIS Board   |
                      | (via USB/UART)  |
                      +-----------------+

Requirements
Python Environment

Ensure Python ≥ 3.8 with the following packages:
pip install pyqt5 matplotlib pandas numpy openpyxl pyserial
Hardware / Network

RIS board connected via serial (e.g. /dev/ttyUSB0)

gNB or simulation source sending JSON CQI data over UDP (default port 55555)

Excel file 5G_patterns.xlsx containing RIS reflection configurations:
