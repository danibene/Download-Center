# % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# % SPS Short Course: Radar Signal Processing Mastery
# % Theory and Hands-On Applications with mmWave MIMO Radar Sensors
# % Date: 7-11 October 2024
# % Time: 9:00AM-11:00AM ET (New York Time)
# % Presenter: Mohammad Alaee-Kerahroodi
# % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# % Website: https://radarmimo.com/
# % Email: info@radarmimo.com, mohammad.alaee@uni.lu
# % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import pprint
import queue
import sys
import threading
import time
from datetime import datetime
import csv

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from ifxradarsdk import get_version
from ifxradarsdk.ltr11 import DeviceLtr11
from ifxradarsdk.ltr11.types import Ltr11Config
from pyqtgraph.Qt import QtCore

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
prt_index = 1  # 0 = 4000 Hz,  1 = 2000 Hz, 2 = 1000 Hz, 3 = 500 Hz
if prt_index == 0:
    sample_rate = 4000
elif prt_index == 1:
    sample_rate = 2000
elif prt_index == 2:
    sample_rate = 1000
else:
    sample_rate = 500

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialization
ENABLE_I_Q_PLOT = True
save_to_csv = True  # Set this to True to save data to CSV
csv_filename = f"radar_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
csv_save_interval = 1  # seconds between saves

sample_time = 1 / sample_rate
num_of_samples = 256
window_time = 1  # second
buffer_time = 5 * window_time  # second
figure_update_time = 25  # m second
num_rx_antennas = 1
raw_data_size = int(buffer_time * sample_rate)
IQ_xaxis = np.linspace(1, buffer_time, raw_data_size)
epsilon_value = 0.00000001
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# data queue
data_queue = queue.Queue()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def read_data(sensor):
    while True:
        frame_contents = sensor.get_next_frame()
        for frame in frame_contents:
            data_queue.put(frame)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# processing class
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class MyProcessorClass:
    @staticmethod
    def process_data():
        global raw_data, new_data_buffer
        while True:
            time.sleep(1 / sample_rate)
            if not data_queue.empty():
                frame = data_queue.get()
                if np.size(frame) == num_of_samples:
                    raw_data = np.roll(raw_data, -num_of_samples)
                    raw_data[-num_of_samples:] = frame
                    new_data_buffer.append(frame)  # Append new frame to the buffer

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def update_plots():
    if ENABLE_I_Q_PLOT:
        I_Q_PLOT[0][0].setData(IQ_xaxis, np.real(raw_data))
        I_Q_PLOT[1][0].setData(IQ_xaxis, np.imag(raw_data))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def generate_iq_plot():
    plot = pg.plot(title='Inphase and Quadrature')
    plot.showGrid(x=True, y=True)
    plot.setLabel('bottom', 'Time [s]')
    plot.setLabel('left', 'Amplitude')
    plot.addLegend()
    plots = [
        ('lightblue', 'Inphase [I]'),
        ('gold', 'Quadrature [Q]')
    ]
    plot_objects = [[] for _ in range(len(plots))]
    for j, (color, name) in enumerate(plots):
        line_style = {'color': color, 'style': [QtCore.Qt.SolidLine, QtCore.Qt.DashLine, QtCore.Qt.DotLine][0]}
        plot_obj = plot.plot(pen=line_style, name=f'{name}')
        plot_obj.setVisible(True)
        plot_objects[j].append(plot_obj)
    return plot, plot_objects

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def save_data_to_csv(data_buffer, filename):
    # Append raw I and Q data into the existing CSV file
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        for frame in data_buffer:
            current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Include milliseconds
            for i in range(num_of_samples):
                writer.writerow([current_timestamp, i * sample_time, np.real(frame[i]), np.imag(frame[i])])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def periodic_save():
    global new_data_buffer
    if save_to_csv and new_data_buffer:
        save_data_to_csv(new_data_buffer, csv_filename)
        new_data_buffer = []  # Clear buffer after saving
    QTimer.singleShot(csv_save_interval * 1000, periodic_save)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    # Initialize application and plots
    app = QApplication([])

    if ENABLE_I_Q_PLOT:
        iq_figure, I_Q_PLOT = generate_iq_plot()
        iq_figure.show()

    # Connect to the device
    pp = pprint.PrettyPrinter()
    with DeviceLtr11() as device:
        print("Radar SDK Version: " + get_version())
        print("Sampling Frequency [Hz]: ", sample_rate)

        config = Ltr11Config(
            aprt_factor=4,
            detector_threshold=80,
            disable_internal_detector=False,
            hold_time=8,
            mode=0,  # Continuous wave mode
            num_of_samples=num_of_samples,
            prt=prt_index,
            pulse_width=3,
            rf_frequency_Hz=61044000000,
            rx_if_gain=8,
            tx_power_level=7,
        )
        device.set_config(config)

        pp.pprint(device.get_config())

        # Initialize raw data buffer
        raw_data = np.zeros(raw_data_size, dtype=np.complex128)

        # Threads for reading and processing data
        data_thread = threading.Thread(target=read_data, args=(device,))
        data_thread.start()

        radar_processor = MyProcessorClass()
        process_thread = threading.Thread(target=radar_processor.process_data, args=())
        process_thread.start()

        # Periodic save and plot updates
        periodic_save()  # Start saving data periodically
        timer = QTimer()
        timer.timeout.connect(update_plots)
        timer.start(figure_update_time)  # Update plots every 25 ms

        sys.exit(app.exec_())
