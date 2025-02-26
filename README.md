# BugSentinel
BugSentinel: A Solar-Powered, Battery-Free AI System for Agricultural Pest Monitoring

BugSentinel is a solar energy-harvesting system designed to offer continuous, sustainable operation for agricultural pest monitoring. Powered solely by solar energy, this system eliminates the need for traditional batteries, making it a truly self-sustaining and eco-friendly solution for precision farming.

Key Features:

Solar-Powered Operation: By harnessing solar energy, BugSentinel operates without a battery, ensuring long-term, low-maintenance functionality, ideal for remote agricultural settings.

Deep Learning Inference on Embedded Systems: A custom deep learning model was trained using my own insect dataset to classify pest species accurately. This model was then converted into an optimized C header file, enabling it to run on the MSP430FR5994 microcontroller using the SONIC inference framework.

Efficient Inference Code: Custom code was written to perform inference on the trained deep neural network (DNN) directly on the MSP430, ensuring the model operates efficiently within the hardware constraints while delivering high accuracy in pest detection.
