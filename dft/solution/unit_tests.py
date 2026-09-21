import unittest
import numpy as np
import time
from dft import (
    _dft_matrix, _idft_matrix, dft, idft, dft2, idft2,
    compress_image_fft, low_pass_filter
)

class TestDFTFunctions(unittest.TestCase):
    # ----------------------------
    # Tests for _dft_matrix
    # ----------------------------
    def test_dft_matrix_valid_n1(self):
        # For n = 1, the DFT matrix should be [[1]]
        mat = _dft_matrix(1)
        expected = np.array([[1+0j]])
        np.testing.assert_allclose(mat, expected)

    def test_dft_matrix_valid_n2(self):
        # For n = 2, DFT matrix is [[1, 1], [1, -1]]
        mat = _dft_matrix(2)
        expected = np.array([[1+0j, 1+0j],
                             [1+0j, -1+0j]])
        np.testing.assert_allclose(mat, expected)

    def test_dft_matrix_invalid_n_zero(self):
        with self.assertRaises(ValueError):
            _dft_matrix(0)

    def test_dft_matrix_invalid_n_negative(self):
        with self.assertRaises(ValueError):
            _dft_matrix(-3)

    def test_dft_matrix_invalid_type(self):
        with self.assertRaises(ValueError):
            _dft_matrix("4")

    # ----------------------------
    # Tests for _idft_matrix
    # ----------------------------
    def test_idft_matrix_valid_n1(self):
        # For n = 1, IDFT matrix should be [[1]]
        mat = _idft_matrix(1)
        expected = np.array([[1+0j]])
        np.testing.assert_allclose(mat, expected)

    def test_idft_matrix_valid_n2(self):
        # For n = 2, IDFT matrix is (1/2)*[[1, 1], [1, -1]]
        mat = _idft_matrix(2)
        expected = np.array([[0.5+0j, 0.5+0j],
                             [0.5+0j, -0.5+0j]])
        np.testing.assert_allclose(mat, expected)

    def test_idft_matrix_invalid_n_zero(self):
        with self.assertRaises(ValueError):
            _idft_matrix(0)

    def test_idft_matrix_invalid_n_negative(self):
        with self.assertRaises(ValueError):
            _idft_matrix(-5)

    def test_idft_matrix_invalid_type(self):
        with self.assertRaises(ValueError):
            _idft_matrix(3.5)

    # ----------------------------
    # Tests for dft and idft (1D)
    # ----------------------------
    def test_dft_valid_simple_vector(self):
        # x = [1, 2]; DFT should be [3, -1]
        x = [1, 2]
        result = dft(x)
        expected = np.array([3+0j, -1+0j])
        np.testing.assert_allclose(result, expected)

    def test_idft_valid_simple_vector(self):
        # X = [3, -1]; IDFT should recover [1, 2]
        X = [3, -1]
        result = idft(X)
        expected = np.array([1+0j, 2+0j])
        np.testing.assert_allclose(result, expected)

    def test_dft_then_idft_inverse(self):
        x = np.random.rand(4) + 1j * np.random.rand(4)
        X = dft(x)
        x_rec = idft(X)
        np.testing.assert_allclose(x_rec, x, atol=1e-8)

    def test_dft_invalid_dimension(self):
        # Passing 2D array to dft should raise ValueError
        arr2d = np.array([[1, 2], [3, 4]])
        with self.assertRaises(ValueError):
            dft(arr2d)

    def test_idft_invalid_dimension(self):
        # Passing 2D array to idft should raise ValueError
        arr2d = np.array([[1, 2], [3, 4]])
        with self.assertRaises(ValueError):
            idft(arr2d)

    def test_dft_empty_input(self):
        # Empty 1D array: _dft_matrix(0) should raise ValueError
        with self.assertRaises(ValueError):
            dft([])

    def test_idft_empty_input(self):
        # Empty 1D array: _idft_matrix(0) should raise ValueError
        with self.assertRaises(ValueError):
            idft([])

    # ----------------------------
    # Tests for dft2 and idft2 (2D)
    # ----------------------------
    def test_dft2_valid_small_matrix(self):
        # For a small 2x2 input, verify shape and type
        img = np.array([[1, 2],
                        [3, 4]], dtype=float)
        F = dft2(img)
        # After computing 2D DFT and fftshift, result is complex 2x2
        self.assertEqual(F.shape, (2, 2))
        self.assertTrue(np.iscomplexobj(F))

    def test_idft2_valid_inverse(self):
        # dft2 followed by idft2 should recover original
        img = np.random.rand(3, 3) + 1j * np.random.rand(3, 3)
        F = dft2(img)
        img_rec = idft2(F)
        np.testing.assert_allclose(img_rec, img, atol=1e-8)

    def test_dft2_invalid_dimension(self):
        # Passing 1D array to dft2 should raise ValueError
        arr1d = np.array([1, 2, 3])
        with self.assertRaises(ValueError):
            dft2(arr1d)

    def test_idft2_invalid_dimension(self):
        # Passing 1D array to idft2 should raise ValueError
        arr1d = np.array([1, 2, 3])
        with self.assertRaises(ValueError):
            idft2(arr1d)

    def test_dft2_empty_input(self):
        # Empty 2D array (shape (0,0)) should raise ValueError via _dft_matrix
        empty2d = np.empty((0, 0))
        with self.assertRaises(ValueError):
            dft2(empty2d)

    def test_idft2_empty_input(self):
        # Empty 2D array (shape (0,0)) should raise ValueError via _idft_matrix
        empty2d = np.empty((0, 0))
        with self.assertRaises(ValueError):
            idft2(empty2d)


# ----------------------------
# Tests for compress_image_fft
# ----------------------------
class TestCompressImageFFT(unittest.TestCase):
    def setUp(self):
        # simple 4×4 RGB test image with distinct values
        self.img = np.arange(48, dtype=np.uint8).reshape((4,4,3))

    def test_full_ratio(self):
        comp, spec = compress_image_fft(self.img, 1.0)
        np.testing.assert_allclose(comp, self.img.astype(np.float64), atol=1e-6)
        self.assertEqual(spec.shape, self.img.shape)

    def test_zero_ratio(self):
        comp, spec = compress_image_fft(self.img, 0.0)
        zeros = np.zeros_like(self.img, dtype=np.float64)
        np.testing.assert_allclose(comp, zeros, atol=1e-8)
        self.assertEqual(spec.shape, self.img.shape)

    def test_invalid_type(self):
        with self.assertRaises(TypeError):
            compress_image_fft("not an array", 0.5)
        with self.assertRaises(TypeError):
            compress_image_fft(self.img, "half")

    def test_invalid_shape(self):
        bad = np.zeros((4,4))  # missing channel dimension
        with self.assertRaises(ValueError):
            compress_image_fft(bad, 0.5)

    def test_invalid_threshold_value(self):
        with self.assertRaises(ValueError):
            compress_image_fft(self.img, -0.1)
        with self.assertRaises(ValueError):
            compress_image_fft(self.img, 1.1)


# ----------------------------
# Tests for low_pass_filter
# ----------------------------
class TestLowPassFilter(unittest.TestCase):
    def setUp(self):
        # constant image; DFT has only DC component
        self.img = np.ones((8,8,3), dtype=np.uint8) * 128

    def test_constant_image(self):
        target = self.img.astype(np.float64)
        for r in [0, 1, 10]:
            filtered, spec = low_pass_filter(self.img, r)
            np.testing.assert_allclose(filtered, target, atol=1e-6)
            self.assertEqual(spec.shape, self.img.shape)

    def test_invalid_type(self):
        with self.assertRaises(TypeError):
            low_pass_filter("not an array", 5)
        with self.assertRaises(TypeError):
            low_pass_filter(self.img, "five")

    def test_invalid_shape(self):
        bad = np.zeros((8,8))  # missing channel dimension
        with self.assertRaises(ValueError):
            low_pass_filter(bad, 3)

    def test_negative_radius(self):
        with self.assertRaises(ValueError):
            low_pass_filter(self.img, -1)

if __name__ == "__main__":
    unittest.main()