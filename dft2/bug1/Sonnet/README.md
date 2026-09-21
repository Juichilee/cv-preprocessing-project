# Fourier Project

A simple Python library and CLI for applying Fourier‑based compression and low‑pass filtering to BGR images.

## Features

- **1D & 2D DFT/IDFT**  
  Fast matrix‑cached transforms with `dft`, `idft`, `dft2`, `idft2`.  
- **Spectrum‑threshold compression**  
  `compress_image_fft(img, keep_ratio)` keeps only the top‑k Fourier coefficients and reconstructs the image.  
- **Circular low‑pass filter**  
  `low_pass_filter(img, radius)` zeroes out all but the DC and nearby frequencies within a given radius.  
- **Command‑line runner**  
  `run.py` to batch‑process two images (one for compression, one for low‑pass) with multiple settings.

## Requirements

- Python 3.11+
- OpenCV
- NumPy

## Installation

### Step 1: Set up a virtual environment

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` file, create one with the following content:

```
opencv-python
numpy
```

Then run:

```bash
pip install -r requirements.txt
```

## Usage

All processing is done via the run.py script.

```bash
run.py [-h] [-o OUTPUT_DIR] compress_input lowpass_input
```

### Arguments

   compress_input — path to the image you want to compress (e.g. dog.jpg)
   
   lowpass_input — path to the image you want to low‑pass filter (e.g. cat.jpg)
   
   -o, --output-dir — (optional) output directory (default: output_images)


### Basic Example
Compress dog.jpg and low‑pass filter cat.png, write into out/:

```bash
python run.py dog.jpg cat.png -o out
```


