# Template Matching Project

This project implements various template matching algorithms to locate a template image within a larger scene image.

## Features

- Four template matching methods: Sum of Squared Differences (SSD), Normalized SSD, Cross-Correlation, and Normalized Cross-Correlation
- Visual output with matches highlighted and coordinates displayed
- Command-line interface for easy use

## Requirements

- Python 3.11+
- OpenCV
- NumPy
- Numba

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
numba
```

Then run:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python run.py   
```

### Arguments

- `scene_image`: Path to the larger image where we want to find the template
- `template_image`: Path to the smaller image to find within the scene
- `output_label`: Label to include in output filenames

### Optional Arguments

- `--output_dir`: Directory where output images will be saved (default: ./output)

### Example

```bash
python run.py waldo.png waldo_template.png waldo_out
```

This will:
1. Look for `waldo_template.png` within the `waldo.png` image
2. Apply all four matching algorithms
3. Save the results as four separate images in the `output` directory:
   - `tm_ssd-waldo_out.png`
   - `tm_nssd-waldo_out.png`
   - `tm_ccor-waldo_out.png`
   - `tm_nccor-waldo_out.png`

## Project Structure

- `run.py`: Main entry point for running template matching
- `template_match.py`: Implementation of template matching algorithms
