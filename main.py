# This work is licensed under the MIT license.
# Copyright (c) 2013–2023 OpenMV LLC.
# All rights reserved.
# https://github.com/openmv/openmv/blob/master/LICENSE
#
# ------------------------------------------------------------
# WiFi IMU Streaming Example (Nicla Vision)
# ------------------------------------------------------------
# This example demonstrates how to:
# 1. Connect the Nicla Vision board to a WiFi network
# 2. Read accelerometer and gyroscope data from the LSM6DSOX IMU
# 3. Send the IMU data to a PC over UDP
#
# The IMU data is streamed in CSV format, making it easy to log
# or visualize on the host PC (e.g., using Python, MATLAB, Excel).
# ------------------------------------------------------------

import network, time, socket
from machine import Pin, SPI, LED
from lsm6dsox import LSM6DSOX

# ------------------------------------------------------------
# User Configuration
# ------------------------------------------------------------

# WiFi credentials (mobile hotspot or router)
SSID = "<Mobile Hotspot Name>"
KEY  = "<Mobile Hotspot Password>"

# IP address of the PC connected to the same WiFi network
PC_IP = "<PC IP Address connected to the same Mobile Hotspot>"

# UDP port number (must match the PC-side receiver)
PORT = 5005

# ------------------------------------------------------------
# WiFi Initialization and Connection
# ------------------------------------------------------------

# Initialize WLAN interface in station mode
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect(SSID, KEY)

# On-board status LEDs
red_led = LED("LED_RED")
green_led = LED("LED_GREEN")

# Attempt to connect to WiFi with a timeout
timeout = 5
while not wlan.isconnected() and timeout > 0:
    print('Trying to connect to "{:s}"...'.format(SSID))
    time.sleep_ms(1000)
    timeout -= 1

# Create a UDP socket for transmitting IMU data
client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ------------------------------------------------------------
# IMU Data Acquisition and Transmission
# ------------------------------------------------------------

def imu_data():
    """
    Continuously reads IMU data from the LSM6DSOX sensor and
    sends it to the host PC via UDP.
    """
    print('Now collecting the IMU data')

    # Initialize SPI interface and chip select pin
    spi = SPI(5)
    cs = Pin("PF6", Pin.OUT_PP, Pin.PULL_UP)

    # Initialize the IMU sensor
    lsm = LSM6DSOX(spi, cs)

    # CSV header for logging or plotting
    print("Timestamp,Ax,Ay,Az,Gx,Gy,Gz")

    while True:
        # Read accelerometer and gyroscope data
        a = lsm.accel()  # (Ax, Ay, Az)
        g = lsm.gyro()   # (Gx, Gy, Gz)

        # Timestamp in milliseconds
        ts = time.ticks_ms()

        # Format data as CSV string
        data = "%d, %f, %f, %f, %f, %f, %f" % (
            ts,
            a[0], a[1], a[2],
            g[0], g[1], g[2]
        )

        # Print locally (USB serial)
        print(data)

        # Send IMU data to PC via UDP
        client.sendto(data.encode(), (PC_IP, PORT))

               # Delay to achieve ~50 Hz sampling rate
        time.sleep_ms(20)

# ------------------------------------------------------------
# Program Entry Point
# ------------------------------------------------------------

if not wlan.isconnected():
    # WiFi connection failed
    print('Failed to connect to Wi-Fi')
    red_led.on()

    # Halt execution
    while True:
        pass
else:
    # WiFi connection successful
    print("WiFi Connected:", wlan.ifconfig())
    green_led.on()

    # Start IMU streaming
    imu_data()
