# simple-vehicle-counting-using-opencv

## Overview

This repository contains a vehicle counting system that processes video input from toll gates or other roadways. The system uses a combination of object tracking and boundary detection to count the number of cars and buses passing through specific points. The program is designed to be flexible and adjustable, allowing for fine-tuning of parameters to suit different environments and video qualities.

## Features

- Counts vehicles (cars and buses) from video footage.
- Uses object tracking and centroid-based counting.
- Configurable parameters for tolerance and width thresholds to differentiate between cars and buses.
- Outputs the number of cars and buses that passed through the counting points.

## Getting Started

### Prerequisites

Before running the program, ensure you have the following installed:

- Python 3.x
- OpenCV (`cv2`)
- Numpy
- Other relevant dependencies (e.g., object detection/tracking libraries)

To install dependencies, you can use `pip`:

```bash
pip install -r requirements.txt
```

### Folder Structure

- `main.py`: The main script to run the vehicle counting system.
- `toll_gate.mp4`: Sample video input (replace with your own video if needed).
- `README.md`: This file.

### Running the Program

To run the vehicle counting system, follow these steps:

1. **Prepare the Input Video**: Ensure your input video (e.g., `toll_gate.mp4`) is in the same directory as the `main.py` script.

2. **Execute the Program**:

```bash
python main.py
```

3. **Output**: The program will process the video and output the number of cars and buses detected in the terminal.

### Configuring Parameters

In the `main.py` script, you can adjust the following parameters:

- **Tolerance**: Adjusts the sensitivity of the vehicle counting logic. Higher tolerance may result in more vehicles being counted, but it may also increase noise.
- **Width Threshold**: Determines the width boundary to differentiate between cars and buses. Increase or decrease this value based on the size of vehicles in your video.

Example:

```python
car_tolerance=80
bus_tolerance=25
```

### Example Usage

The program is tested with a sample video of a toll gate, and the output will be the total number of cars and buses that passed through the gate.

### Troubleshooting

If the vehicle count is inaccurate or the program misses vehicles:

- **Adjust the Tolerance**: Try increasing or decreasing the tolerance parameter.
- **Modify the Width Threshold**: If buses are being misclassified as cars, increase the threshold value.

## Additional Information

- This system is designed to be adaptable to various environments. Feel free to tweak the parameters to fit your specific use case.
- The accuracy of the system depends on the quality of the video input. High-resolution videos with clear object visibility will yield better results.

## Contributions

Feel free to fork this repository and contribute to the project. Pull requests are welcome!

## Contact

For any inquiries, please contact hefrykun10@gmail.com or open an issue in the repository.
