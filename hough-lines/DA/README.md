# Traffic Light and Construction Sign Detection

A simple Python tool to locate and annotate traffic lights and construction signs in your images.

## Features

- **Traffic Light**: Detects the center of a traffic light and its current state (red, yellow, green).  
- **Construction Sign**: Finds the geometric center of a construction sign using edge detection.  
- **Annotation**: Draws cross‑markers and overlays coordinate/state labels on each detection.  
- **Batch Processing**: Handle one traffic‑light image and one construction‑sign image at a time via the command line.

## Requirements

- Python 3.11+
- OpenCV
- NumPy
- Matplotlib

## Installation

1. **Create** and **activate** a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install** the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage
    ```bash
    python run.py <traffic_light_image> <construction_sign_image> 
    ```

    - <traffic_light_image>: Path to your traffic‑light photo (e.g., scene_tl.jpg).
    - <construction_sign_image>: Path to your construction‑sign photo (e.g., scene_constr.png).

# When you run the script:

    An output/ folder is created.
    
    The traffic‑light image is processed, annotated, and saved as
    output/<basename>_out.png (e.g., scene_tl_out.png).
    
    The construction‑sign image is processed, annotated, and saved as
    output/<basename>_out.png (e.g., scene_constr_out.png).

# Example
    ```bash
    python run.py scene_tl.jpg scene_constr.jpg
    ```

## Module Overview
    run.py:
        Parses command‑line arguments
        Calls traffic_light_detection(...) and construction_sign_detection(...)
        Draws markers and labels via helper functions
        Saves annotated images to output/

    detection.py:
        traffic_light_detection(img_in, radii_range):
            Returns (center, state) for a traffic light
        construction_sign_detection(img_in):
            Returns (x, y) centroid of a construction sign