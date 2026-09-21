import argparse
import numpy as np
import cv2
from cv_preprocess import (
    extract_red,
    extract_green,
    extract_blue,
    swap_green_blue,
    copy_paste_middle,
    copy_paste_middle_circle,
    image_stats,
    center_and_normalize,
    shift_image_left,
    difference_image,
    add_noise,
    build_hybrid_image,
    vis_hybrid_image,
)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the full image processing pipeline.'
    )
    parser.add_argument(
        'wide_image',
        help='Path to a wide-format image (for parts 1–5)'
    )
    parser.add_argument(
        'tall_image',
        help='Path to a tall-format image (for parts 1–5)'
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # load the two main input images
    img1 = cv2.imread(args.wide_image)
    img2 = cv2.imread(args.tall_image)
    if img1 is None or img2 is None:
        raise FileNotFoundError('Could not load one of the main images.')

    # verify image dimensions and orientation
    assert 100 < img1.shape[0] <= 512, 'Wide image height out of range'
    assert 100 < img1.shape[1] <= 512, 'Wide image width out of range'
    assert 100 < img2.shape[0] <= 512, 'Tall image height out of range'
    assert 100 < img2.shape[1] <= 512, 'Tall image width out of range'
    assert img1.shape[1] > img1.shape[0], 'First image must be wider'
    assert img2.shape[0] > img2.shape[1], 'Second image must be taller'

    # save originals for part 1a
    cv2.imwrite('1-a-1.png', img1)
    cv2.imwrite('1-a-2.png', img2)

    # swap green and blue channels (part 2a)
    swapped = swap_green_blue(img1)
    cv2.imwrite('2-a-1.png', swapped)

    # get green channel (part 2b)
    green = extract_green(img1)
    assert green.ndim == 2, 'Green channel extraction failed'
    cv2.imwrite('2-b-1.png', green)

    # get red channel (part 2c)
    red = extract_red(img1)
    assert red.ndim == 2, 'Red channel extraction failed'
    cv2.imwrite('2-c-1.png', red)

    # prepare two monochrome images for copy‑paste (part 3)
    mono1 = green
    mono2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # rectangular paste (part 3a)
    rect_out = copy_paste_middle(mono1, mono2, (100, 100))
    cv2.imwrite('3-a-1.png', rect_out)

    # circular paste (part 3b)
    circ_out = copy_paste_middle_circle(mono1, mono2, 50)
    cv2.imwrite('3-b-1.png', circ_out)

    # compute stats on green channel (part 4a)
    mn, mx, mean_val, std_val = image_stats(green)
    print('Min:', mn, 'Max:', mx, 'Mean:', mean_val, 'StdDev:', std_val)


if __name__ == '__main__':
    main()
