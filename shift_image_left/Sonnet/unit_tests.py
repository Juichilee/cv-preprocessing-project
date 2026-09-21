import unittest
import numpy as np
from cv_preprocess import * 

class TestImageProcessingFunctions(unittest.TestCase):

    def test_center_and_normalize_valid_input(self):
        # Test for a valid input where the image has variation and scale is positive
        image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
        result = center_and_normalize(image, 2.0)
        self.assertEqual(result.shape, image.shape)
        self.assertTrue(np.issubdtype(result.dtype, np.floating))

    def test_center_and_normalize_single_value(self):
        # Test for an image where all pixel values are the same (no variation)
        image = np.array([[5, 5, 5], [5, 5, 5], [5, 5, 5]], dtype=np.uint8)
        with self.assertRaises(ValueError):
            center_and_normalize(image, 2.0)

    def test_center_and_normalize_zero_variation(self):
        # Test for an image with zero variation (all pixels have the same value)
        image = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.uint8)
        with self.assertRaises(ValueError):
            center_and_normalize(image, 2.0)

    def test_center_and_normalize_invalid_scale(self):
        # Test for invalid scale values (negative scale value)
        image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
        with self.assertRaises(ValueError):
            center_and_normalize(image, -1.0)  # Invalid scale

    def test_center_and_normalize_non_numeric_image(self):
        # Test for an image that is not a NumPy array
        image = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # Not a NumPy array
        with self.assertRaises(ValueError):
            center_and_normalize(image, 2.0)

    def test_center_and_normalize_non_two_dimensional_image(self):
        # Test for an image that is not two-dimensional
        image = np.array([[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]]], dtype=np.uint8)
        with self.assertRaises(ValueError):
            center_and_normalize(image, 2.0)

    def test_shift_image_left_valid_input(self):
        # Test for shifting an image left by 1 column
        image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
        result = shift_image_left(image, 1)
        expected_result = np.array([[2, 3, 1], [5, 6, 4], [8, 9, 7]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected_result)

    def test_shift_image_left_edge_case_zero_shift(self):
        # Test for no shift (shift = 0)
        image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
        result = shift_image_left(image, 0)
        np.testing.assert_array_equal(result, image)

    def test_shift_image_left_invalid_shift(self):
        # Test for invalid shift values (negative shift)
        image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
        with self.assertRaises(ValueError):
            shift_image_left(image, -1)  # Invalid shift

    def test_shift_image_left_shift_greater_than_width(self):
        # Test for shift that exceeds the image width
        image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
        with self.assertRaises(ValueError):
            shift_image_left(image, 4)  # Shift exceeds width

    def test_difference_image_valid_input(self):
        # Test for valid inputs with two different images
        img1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
        img2 = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.uint8)
        result = difference_image(img1, img2, 0.0, 255.0)
        self.assertEqual(result.shape, img1.shape)
        self.assertTrue(np.issubdtype(result.dtype, np.floating))

    def test_difference_image_identical_images(self):
        # Test for identical images, expecting a result of zeros
        img1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
        img2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
        result = difference_image(img1, img2, 0.0, 255.0)
        # Identical images should return a constant array with min value
        np.testing.assert_array_equal(result, np.zeros_like(result))

    def test_difference_image_invalid_input(self):
        # Test for invalid second input (not a NumPy array)
        img1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
        img2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # Invalid second input (not an array)
        with self.assertRaises(ValueError):
            difference_image(img1, img2, 0.0, 255.0)

    def test_difference_image_non_two_dimensional_image(self):
        # Test for non-two-dimensional input images
        img1 = np.array([[[1, 2, 3]]], dtype=np.uint8)
        img2 = np.array([[[1, 2, 3]]], dtype=np.uint8)
        with self.assertRaises(ValueError):
            difference_image(img1, img2, 0.0, 255.0)

    def test_difference_image_shape_mismatch(self):
        # Test for input images with mismatched shapes
        img1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
        img2 = np.array([[1, 2, 3]], dtype=np.uint8)
        with self.assertRaises(ValueError):
            difference_image(img1, img2, 0.0, 255.0)

    def test_difference_image_invalid_output_min_max(self):
        # Test for invalid output min-max range where min >= max
        img1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
        img2 = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.uint8)
        with self.assertRaises(ValueError):
            difference_image(img1, img2, 255.0, 0.0)  # Invalid min-max range

if __name__ == "__main__":
    unittest.main()