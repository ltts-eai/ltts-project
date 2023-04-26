# Embedded hardware

# JETSON NANO

## HARDWARE SPECIFICATION:

The NVIDIA Jetson Nano is a small, powerful computer designed for embedded applications and AI projects. 

**CPU:** Quad-core ARM Cortex-A57 CPU @ 1.43 GHz
**GPU:** 128-core NVIDIA Maxwell GPU
**Memory:** 4 GB 64-bit LPDDR4 RAM @ 1600 MHz
**Storage:** MicroSD card slot for storage expansion
**Video:** 4K @ 30 fps (H.264/H.265) / 4K @ 60 fps (VP9)
**Interfaces:** Gigabit Ethernet, HDMI 2.0, USB 3.0, USB 2.0, Micro-USB (power only)
**GPIO:** 40-pin header with GPIO, I2C, I2S, SPI, UART interfaces
**Dimensions:** 69.6 mm x 45 mm

Overall, the Jetson Nano provides a powerful and efficient platform for running AI models and processing data in real-time, while remaining compact and affordable.

# PROCESSING POWER:

The processing power of the NVIDIA Jetson Nano is quite impressive, especially considering its small size and low power consumption.

**CPU:** The Jetson Nano features a quad-core ARM Cortex-A57 CPU running at 1.43 GHz. This provides a lot of processing power for general computing tasks and running the Linux operating system.

**GPU:** The Jetson Nano's GPU is a 128-core NVIDIA Maxwell GPU, which is specifically designed for accelerating deep learning and AI tasks. This GPU provides a significant amount of parallel processing power, which can be used to accelerate tasks such as image recognition and natural language processing.

## MEMORY:

The NVIDIA Jetson Nano has 4 GB of 64-bit LPDDR4 RAM running at 1600 MHz. This is a relatively large amount of memory for an embedded device, and it allows the Jetson Nano to handle complex tasks and large data sets with ease.

Having 4 GB of RAM is especially useful for running deep learning models and other memory-intensive tasks, as it allows the Jetson Nano to load large data sets into memory and perform computations on them quickly. Additionally, LPDDR4 RAM is designed to be low-power, which is important for an embedded device that needs to operate on a limited power budget.
The Jetson Nano's 4 GB of RAM is a significant amount of memory for an embedded device, and it helps to make the Jetson Nano a powerful platform for running AI models and processing large amounts of data in real-time.

## BENCH MARK:

The benchmark performance of the NVIDIA Jetson Nano varies depending on the specific task and software being used.

While running tensorflow, the Jetson Nano achieved a throughput of 1.3 images per second (IPS) while running a MobileNetV2 model for image classification. This is significantly faster than running the same model on a Raspberry Pi, which achieved a throughput of only 0.1 IPS.

The Jetson Nano also performs well when running computer vision tasks using OpenCV. It achieved a frame rate of 23.6 frames per second (FPS) while running a facial recognition algorithm using OpenCV's Haar cascades.

It is also well-suited for robotics applications, and can be used to run SLAM algorithms, object detection and tracking, and other real-time robotics tasks. In one benchmark, the Jetson Nano achieved a frame rate of 24 FPS while running an object detection and tracking algorithm using the YOLOv3 model.

## FEW AI APPLICATIONS THAT ARE RUN ON JETSON NANO:

*	Object detection and recognition:
*	Natural language processing
*	Image and video processing:
*	Autonomous navigation:


# RASPBERRY PI 4B

## HARDWARE SPECIFICATION:

**CPU:** Broadcom BCM2711, quad-core Cortex-A72 (ARM v8) 64-bit SoC @ 1.5GHz
**RAM:** 2GB, 4GB, or 8GB LPDDR4-3200 SDRAM (depending on model)
**Connectivity:**
*   2.4 GHz and 5.0 GHz IEEE 802.11b/g/n/ac wireless LAN
*   Gigabit Ethernet
*	Bluetooth 5.0
*	BLE (Bluetooth Low Energy)
*	2 USB 3.0 ports and 2 USB 2.0 ports
*	Video and Sound:
*	2 micro-HDMI ports (up to 4Kp60 supported)
*	2-lane MIPI DSI display port
*	2-lane MIPI CSI camera port
*	4-pole stereo audio and composite video port
**Multimedia:** H.265 (4Kp60 decode); H.264 (1080p60 decode, 1080p30 encode); OpenGL ES, 3.0 graphics
**Storage:** MicroSD card slot for loading operating system and data storage
**GPIO:** Standard 40-pin GPIO header, fully backwards-compatible with previous Raspberry Pi boards
**Power:** 5V DC via USB-C connector (minimum 3A), or GPIO header (minimum 3A)
**Dimensions:** 88 x 58 x 19.5 mm, 46 g

The Raspberry Pi 4B is a powerful and versatile single-board computer that is suitable for a wide range of applications, from hobbyist projects to commercial products. Its high processing power, built-in connectivity options, and support for a wide range of software

## PROCESSING POWER:

The Raspberry Pi 4B is a powerful single-board computer that can handle a wide range of computing tasks. 

**CPU:** The Raspberry Pi 4B is powered by a Broadcom BCM2711, quad-core Cortex-A72 (ARM v8) 64-bit SoC @ 1.5GHz. This provides a significant amount of processing power for running complex algorithms and data-intensive tasks.

**GPU:** The Raspberry Pi 4B includes a Broadcom VideoCore VI GPU, which is capable of rendering 4K video at 60 frames per second. It also supports OpenGL ES 3.0 graphics, making it well-suited for graphics-intensive applications.

**Memory:** The Raspberry Pi 4B is available with 2GB, 4GB, or 8GB of LPDDR4-3200 SDRAM, which provides high bandwidth and low power consumption. The memory is shared between the CPU and GPU, allowing for efficient data transfer and processing.


## MEMORY:

The Raspberry Pi 4B is available with three different RAM options: 2GB, 4GB, or 8GB LPDDR4-3200 SDRAM.

The RAM on the Raspberry Pi 4B is shared between the CPU and the GPU, which allows for efficient data transfer and processing. The LPDDR4-3200 SDRAM is a high-bandwidth, low-power type of memory that helps to maximize the performance and efficiency of the Raspberry Pi 4B.

## FEW AI APPLICATIONS USED ON RASPBERRY PI 4B:

*	 Computer vision:
*	 Natural language processing:
*	 Robotics
*	 Machine learning:


# ARDUINO UNO


## HARDWARE SPECIFICATION:

Microcontroller: ATmega328P
Operating Voltage: 5V
Input Voltage (recommended): 7-12V
Input Voltage (limit): 6-20V
Digital I/O Pins: 14 (of which 6 provide PWM output)
PWM Digital I/O Pins: 6
Analog Input Pins: 6
DC Current per I/O Pin: 20 mA
DC Current for 3.3V Pin: 50 mA
Flash Memory: 32 KB (ATmega328P)
SRAM: 2 KB (ATmega328P)
EEPROM: 1 KB (ATmega328P)
Clock Speed: 16 MHz

The Arduino Uno board also includes a USB interface for programming and power, a power jack, an ICSP header, and a reset button.

## PROCESSING POWER:

The processing power of the Arduino Uno is relatively low compared to other microcontroller boards. It is based on the ATmega328P microcontroller which has an 8-bit AVR architecture and operates at a clock speed of 16 MHz. It has 32 KB of flash memory, 2 KB of SRAM, and 1 KB of EEPROM. While the processing power of the Arduino Uno is not suitable for complex tasks like machine learning or image processing, it is capable of handling basic input/output operations and simple control tasks. It is ideal for projects that require a low-power and low-cost microcontroller board, such as controlling sensors or actuators, building simple robots, and creating interactive projects.

Flash Memory: 32 KB (ATmega328P microcontroller)
SRAM: 2 KB (ATmega328P microcontroller)
EEPROM: 1 KB (ATmega328P microcontroller)

The flash memory is used to store the program code that is uploaded to the board. The SRAM is used for storing variables and temporary data used during program execution. The EEPROM is used for storing data that needs to be retained even when the board is powered off, such as calibration data or user settings.


## AI APPLICATIONS THAT ARE RUN ON ARDUINO UNO:

 Due to its limited processing power and memory, the Arduino Uno is not well-suited for running complex artificial intelligence algorithms or models. However, it can still be used in a variety of AI-related applications for controlling sensors, collecting data, and interfacing with other devices. Here are a few examples:

*	Machine learning on microcontrollers:
*	Sensor data collection and analysis
*	Robotics and automation:
*	Smart home applications


# ARDUINO NANO

## HARDWARE SPECIFICATION:

Microcontroller: ATmega328P
Operating Voltage: 5V or 3.3V (depending on model)
Input Voltage (recommended): 7-12V (VIN) or 5V (USB)
Input Voltage (limit): 6-20V (VIN) or 5V (USB)
Digital I/O Pins: 14 (of which 6 provide PWM output)
PWM Digital I/O Pins: 6
Analog Input Pins: 8
DC Current per I/O Pin: 20 mA
DC Current for 3.3V Pin: 50 mA
Flash Memory: 32 KB (ATmega328P)
SRAM: 2 KB (ATmega328P)
EEPROM: 1 KB (ATmega328P)
Clock Speed: 16 MHz

## PROCESSING POWER:

The Arduino Nano is powered by an ATmega328P microcontroller, which has a clock speed of 16 MHz. This means that the Arduino Nano can execute up to 16 million instructions per second.

The exact processing power of the Arduino Nano can vary depending on the specific application and the complexity of the program being run. However, in general, the Arduino Nano is capable of handling simple tasks like reading sensor data, controlling LEDs and motors, and communicating with other devices over serial or I2C protocols.

It's important to note that the processing power of the Arduino Nano is relatively limited compared to other microcontrollers or single-board computers, so it may not be suitable for running more complex algorithms or applications.

## MEMORY:

**Flash Memory:** 32 KB (ATmega328P)
**SRAM:** 2 KB (ATmega328P)
**EEPROM:** 1 KB (ATmega328P)


The Flash memory is where the program code is stored. The 32 KB of Flash memory on the Arduino Nano allows for relatively complex programs to be stored on the board.

The SRAM (Static Random Access Memory) is where the program data and variables are stored when the program is running. The 2 KB of SRAM on the Arduino Nano limits the amount of data that can be stored in the program memory at any given time. Therefore, it is important to optimize code to avoid excessive use of SRAM.

The EEPROM (Electrically Erasable Programmable Read-Only Memory) is a non-volatile memory that can be used to store data that needs to be retained even when the power is turned off. The 1 KB of EEPROM on the Arduino Nano can be used to store data such as calibration values or user settings.

## AI APPLICATIONS:

 The Arduino Nano is a relatively low-powered microcontroller and does not have built-in machine learning capabilities. However, it can still be used in AI-related applications as part of a larger system or by using external modules and libraries.

*	Sensor data processing
*	Robotics
*	Edge computing
*	Wearable devices
*	Education
