import unittest
import numpy as np
import cv2

from cv_preprocess import build_hybrid_image, vis_hybrid_image

class TestBuildHybridImage(unittest.TestCase):
    def setUp(self):
        # create a simple constant image for testing
        self.img_const = np.full((10, 10, 3), 128, dtype=np.uint8)

    def test_success_constant_images(self):
        # When both inputs are identical constant images,
        # the hybrid should equal the original everywhere
        out = build_hybrid_image(self.img_const, self.img_const, cutoff_frequency=1)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape, self.img_const.shape)
        np.testing.assert_array_equal(out, self.img_const)

    def test_dtype_and_range(self):
        # inputs of zeros and max intensity
        img_zero = np.zeros((8, 8, 3), dtype=np.uint8)
        img_max  = np.full((8, 8, 3), 255, dtype=np.uint8)
        out = build_hybrid_image(img_zero, img_max, cutoff_frequency=2)
        # output must be uint8 and between 0 and 255
        self.assertEqual(out.dtype, np.uint8)
        self.assertTrue(out.min() >= 0 and out.max() <= 255)

    def test_shape_mismatch_raises(self):
        # inputs must have same shape
        other = np.full((10, 11, 3), 50, dtype=np.uint8)
        with self.assertRaises(ValueError):
            build_hybrid_image(self.img_const, other, cutoff_frequency=1)

    def test_wrong_channels_raises(self):
        # inputs must be 3-channel
        gray = np.zeros((10, 10), dtype=np.uint8)
        with self.assertRaises(ValueError):
            build_hybrid_image(gray, gray, cutoff_frequency=1)

    def test_invalid_cutoff_raises(self):
        # cutoff must be positive integer
        with self.assertRaises(ValueError):
            build_hybrid_image(self.img_const, self.img_const, cutoff_frequency=0)
        with self.assertRaises(ValueError):
            build_hybrid_image(self.img_const, self.img_const, cutoff_frequency=-1)
        with self.assertRaises(ValueError):
            build_hybrid_image(self.img_const, self.img_const, cutoff_frequency=1.5)

    def test_non_numpy_inputs(self):
        # inputs must be numpy arrays
        with self.assertRaises(ValueError):
            build_hybrid_image(list(), list(), cutoff_frequency=1)

class TestVisHybridImage(unittest.TestCase):

    def setUp(self):
        # Create a dummy image for testing
        self.hybrid_image = np.random.rand(100, 200, 3).astype(np.float32)

    def test_base_case(self):
        # Test with default parameters
        result = vis_hybrid_image(self.hybrid_image)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[0], self.hybrid_image.shape[0])
        self.assertEqual(result.shape[2], self.hybrid_image.shape[2])
        self.assertGreater(result.shape[1], self.hybrid_image.shape[1])

    def test_edge_case_small_image(self):
        # Test with very small image (1x1 pixel)
        small_image = np.random.rand(1, 1, 3).astype(np.float32)
        result = vis_hybrid_image(small_image)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[0], small_image.shape[0])
        self.assertGreater(result.shape[1], small_image.shape[1])

    def test_edge_case_invalid_input(self):
        # Test invalid input (non-3-channel image)
        invalid_image = np.random.rand(100, 200).astype(np.float32)
        with self.assertRaises(ValueError):
            vis_hybrid_image(invalid_image)

    def test_edge_case_zero_height(self):
        # Test input with zero height
        zero_height_image = np.random.rand(0, 200, 3).astype(np.float32)
        with self.assertRaises(ValueError):
            vis_hybrid_image(zero_height_image)

    def test_edge_case_invalid_num_scales(self):
        # Test invalid number of scales
        with self.assertRaises(ValueError):
            vis_hybrid_image(self.hybrid_image, num_scales=0)
        with self.assertRaises(ValueError):
            vis_hybrid_image(self.hybrid_image, num_scales=-1)

    def test_edge_case_invalid_scale_factor(self):
        # Test invalid scale factor
        with self.assertRaises(ValueError):
            vis_hybrid_image(self.hybrid_image, scale_factor=0)
        with self.assertRaises(ValueError):
            vis_hybrid_image(self.hybrid_image, scale_factor=1)

    def test_edge_case_invalid_padding(self):
        # Test invalid padding
        with self.assertRaises(ValueError):
            vis_hybrid_image(self.hybrid_image, padding=-1)

    def test_custom_parameters(self):
        # Test with custom parameters
        result = vis_hybrid_image(self.hybrid_image, num_scales=3, scale_factor=0.8, padding=10)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[0], self.hybrid_image.shape[0])
        self.assertEqual(result.shape[2], self.hybrid_image.shape[2])
        self.assertGreater(result.shape[1], self.hybrid_image.shape[1])

if __name__ == "__main__":
    unittest.main()