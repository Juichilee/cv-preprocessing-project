# CV Preprocessing Pipeline Setup

## Prerequisites

Before running the script, make sure you have the following installed:

1. **Python 3.13.2** (or compatible version)
2. **pip** - Python's package installer

## Steps to Set Up the Environment

### 1. Install Python 3.13.2

Ensure that you have Python 3.13.2 installed. You can check your current Python version with:

```bash
python3 --version
```

2. Create a Virtual Environment (Optional)
It is recommended to use a Python virtual environment to keep your dependencies isolated. Run the following commands to create and activate a virtual environment:

```bash
python3 -m venv myenv
source myenv/bin/activate
```

3. Install Dependencies
Next, install the required Python packages using the requirements.txt file. This will install opencv-python and numpy:

```bash
pip install -r requirements.txt
```
Alternatively, you can manually install the packages using pip:

```bash
pip install opencv-python numpy
```

4. Prepare Your Images
Make sure you have the images required by the script. You need the following images:

A wide-format image for the wide_image argument.

A tall-format image for the tall_image argument.

These images should be accessible on your system and provided as arguments when running the script.

5. Run the Script
Now that the environment is set up, you can run the run.py script. Use the following command format to execute the script:

```bash
python run.py <path_to_wide_image> <path_to_tall_image>
```

Example:

```bash
python run.py wide.png tall.png
```

This will process the images according to the steps defined in the script and generate the output images.