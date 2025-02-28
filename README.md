BugSentinel

BugSentinel: **A Solar-Powered, Battery-Free AI System for Agricultural Pest Monitoring**

BugSentinel is a **solar energy-harvesting system** designed to provide continuous, sustainable agricultural **pest monitoring**. By operating entirely on solar power, it eliminates the need for batteries, making it a self-sustaining, eco-friendly solution for precision farming.

![BugSentinel MSP430FR5994](https://github.com/user-attachments/assets/8ba77053-1dd5-4786-809b-47393c657847)


 fig 1. BugSentinel hardware setup


Challenges of Battery-Powered IoT Devices:
IoT devices that rely on batteries face several critical limitations:
- Temperature Sensitivity: Battery performance degrades in extreme hot or cold environments, leading to inconsistent operation.
- Limited Lifespan: The full charge capacity of a battery decreases as charging and discharging cycles are repeated, reducing the device's long-term effectiveness.
- Maintenance Overhead: Batteries require periodic replacement, increasing maintenance costs and limiting deployment in remote areas.
- Environmental Impact: The disposal of depleted batteries poses ecological risks due to hazardous materials used in battery manufacturing.

By eliminating the batteries form IoT, BugSentinel overcomes these issues, making it a robust and sustainable solution for agricultural monitoring.

Experimental Results:
Through real-world testing, we observed that the brightness of sunlight directly affects the system's performance. When sunlight is strong, the system processes data faster, while cloudy conditions slow down execution due to limited energy harvesting. The intermittent operation ensures that the system adapts to these conditions, performing tasks whenever energy is available and pausing during low-energy periods.

Estimated Execution Times:
- Bright Light (Full Illumination -  Sunlight): System wakes up and processes inference in approximately 2-5 seconds.
  
- Dim Light (LED LIGHT - Lower Power Source): System wakes up and processes inference in approximately 10-20 seconds.

- No Light (Power Depleted - System Paused): System enters a waiting state until enough energy is harvested.

**Key Features**:

✅ Solar-Powered, Battery-Free Operation:  
- BugSentinel harnesses solar energy for operation, ensuring long-term, low-maintenance functionality in remote agricultural environments.
  
- The system intermittently operates based on available solar power, performing tasks when energy is sufficient and pausing when power is low. This behavior is controlled by energy stored in a capacitor (2200uF, 10V), which dictates when the system can actively process data. The intermittent nature allows the device to maximize operational efficiency while conserving energy for critical tasks.
  
  <img width="369" alt="intermittent operation of BugSentinal System" src="https://github.com/user-attachments/assets/853372c6-5cc1-4c18-8700-da9747e6c059" />
  
    Fig 2.  "intermittent operation of BugSentinal System" [1]

- The harvesting equipment is configured with MSP430FR5994, and the solar panels used are 5.5cm in length and 2.5cm in width.
  

✅ Deep Learning Inference on Embedded Systems:  
- A custom deep learning model was trained using a  insect dataset for accurate pest species classification.  
- The trained model was optimized and converted into a C header file, allowing it to run efficiently on the MSP430FR5994 microcontroller using the SONIC inference framework [2].  
- The params folder stores the converted C model, making it accessible for execution during runtime.

✅ Optimized On-Device Inference:  
- Custom efficient inference code was developed to execute the trained deep neural network (DNN) directly on the MSP430FR5994.  
- The code ensures that pest detection runs with high accuracy while meeting the hardware constraints of low-power microcontrollers.  

✅ Integrated Learning & Model Deployment:  
- The scripts folder contains Python code for training and converting machine learning models.
- The params folder holds the converted C models ready for execution.
- The SONIC folder enables runtime execution of models, allowing real-time AI-driven decision-making on embedded systems.  

✅ Model Conversion for Embedded AI:  
- Machine learning models are automatically converted into C-compatible formats, enabling seamless deployment on microcontrollers.  

✅ Real-Time Model Execution:  
- SONIC enables runtime execution of AI models, eliminating cloud dependency while delivering on-device intelligence in low-power environments.  

Experimental Results:
Through real-world testing, we observed that the brightness of sunlight directly affects the system's performance. When sunlight is strong, the system processes data faster, while cloudy conditions slow down execution due to limited energy harvesting. The intermittent operation ensures that the system adapts to these conditions, performing tasks whenever energy is available and pausing during low-energy periods.

Estimated Execution Times:
- Bright Light (Full Illumination -  Sunlight): System wakes up and processes inference in approximately 2-5 seconds.
  
- Dim Light (LED LIGHT - Lower Power Source): System wakes up and processes inference in approximately 10-20 seconds.

- No Light (Power Depleted - System Paused): System enters a waiting state until enough energy is harvested.

Future Work:
- Currently, the system executes pest classification by feeding pre-stored input data into the device, without using a real-time camera sensor.
- In the future, I plan to integrate the HM01B0 low-power camera sensor, allowing real-time image acquisition directly from the environment.
- The camera will capture pest images in real-time, sending input to the deep neural network for immediate classification and inference.
- This enhancement will make BugSentinel a fully autonomous, real-time pest detection system capable of adapting to changing environmental conditions.
  

---

References:

[1] ALSUBHI, Arwa, ANARAKY, Reza Ghaiumy, BABATUNDE, Simeon, et al. User-Centered Perspectives on the Design of Batteryless Wearables. International Journal of Human–Computer Interaction, 2024, vol. 40, no 23, p. 8025-8046.

[2] Graham Gobieski, Brandon Lucia, and Nathan Beckmann. 2019. Intelligence Beyond the Edge: Inference on Intermittent Embedded Systems. In Proceedings of the Twenty-Fourth International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS '19). Association for Computing Machinery, New York, NY, USA, 199–213. https://doi.org/10.1145/3297858.3304011





