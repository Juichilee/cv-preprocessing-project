import os
import cv2
import numpy as np
import argparse
import math

import dft

def compression_runner(input_path, output_dir):
    img_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image at {input_path}")

    keeps = [0.1, 0.05, 0.001]
    base = os.path.splitext(os.path.basename(input_path))[0]

    for keep in keeps:
        img_compressed, freq_img = dft.compress_image_fft(img_bgr, keep)
        out_comp = os.path.join(output_dir, f"{base}_compressed_{keep:.3f}.png")
        out_freq = os.path.join(output_dir, f"{base}_compressed_freq_{keep:.3f}.png")
        cv2.imwrite(out_comp, img_compressed)
        cv2.imwrite(out_freq, freq_img)

def low_pass_filter_runner(input_path, output_dir):
    img_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image at {input_path}")
    img_bgr = img_bgr.astype(np.float64)

    radii = [100, 50, 10]
    base = os.path.splitext(os.path.basename(input_path))[0]

    for r in radii:
        img_low, freq_img = dft.low_pass_filter(img_bgr, r)
        out_low = os.path.join(output_dir, f"{base}_lpf_{r}.png")
        out_freq = os.path.join(output_dir, f"{base}_lpf_freq_{r}.png")
        cv2.imwrite(out_low, img_low)
        cv2.imwrite(out_freq, freq_img)

def main():
    parser = argparse.ArgumentParser(
        description="Apply Fourier-based compression and low-pass filtering to images."
    )
    parser.add_argument(
        "compress_input",
        help="Path to the image to compress (e.g. dog.jpg)"
    )
    parser.add_argument(
        "lowpass_input",
        help="Path to the image to low-pass filter (e.g. cat.jpg)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="output_images",
        help="Directory where output images will be saved"
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    compression_runner(args.compress_input, args.output_dir)
    low_pass_filter_runner(args.lowpass_input, args.output_dir)

if __name__ == "__main__":
    main()