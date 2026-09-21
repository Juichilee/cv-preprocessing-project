import unittest
import numpy as np
from fourier import (
    dft, idft, dft2, idft2,
    compress_image_fft, low_pass_filter
)

class TestDFT1D(unittest.TestCase):
    def test_dft_zero(self):
        x = np.zeros(5)
        y = dft(x)
        np.testing.assert_allclose(y, np.zeros(5), atol=1e-8)

    def test_dft_delta(self):
        x = np.zeros(4)
        x[0] = 1
        y = dft(x)
        np.testing.assert_allclose(y, np.ones(4), atol=1e-8)

    def test_idft_inverse(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        X = dft(x)
        x_rec = idft(X)
        np.testing.assert_allclose(x_rec, x, rtol=1e-6, atol=1e-8)

    def test_n1(self):
        x = np.array([7 + 3j])
        y = dft(x)
        self.assertEqual(y.shape, (1,))
        self.assertAlmostEqual(y[0], x[0], places=8)
        x_rec = idft(y)
        self.assertAlmostEqual(x_rec[0], x[0], places=8)

    def test_dimension_error(self):
        with self.assertRaises(ValueError):
            dft(np.zeros((2,2)))
        with self.assertRaises(ValueError):
            idft(np.zeros((2,2)))

class TestDFT2D(unittest.TestCase):
    def test_dft2_zero(self):
        img = np.zeros((3,4))
        F = dft2(img)
        np.testing.assert_allclose(F, np.zeros((3,4)), atol=1e-8)

    def test_idft2_inverse(self):
        img = np.random.rand(5,6)
        F = dft2(img)
        img_rec = idft2(F)
        np.testing.assert_allclose(img_rec, img, rtol=1e-6, atol=1e-8)

    def test_n1x1(self):
        img = np.array([[42]])
        F = dft2(img)
        self.assertEqual(F.shape, (1,1))
        np.testing.assert_allclose(F[0,0], 42)
        img_rec = idft2(F)
        np.testing.assert_allclose(img_rec, img, atol=1e-8)

    def test_dimension_error(self):
        with self.assertRaises(ValueError):
            dft2(np.zeros((2,2,2)))
        with self.assertRaises(ValueError):
            idft2(np.zeros((2,2,2)))

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

    def test_negative_radius(self):
        # negative radius == mask empty == all‑zero output
        filtered, spec = low_pass_filter(self.img, -1)
        zeros = np.zeros_like(self.img)
        np.testing.assert_array_equal(filtered, zeros)
        self.assertEqual(spec.shape, self.img.shape)
        
if __name__ == "__main__":
    unittest.main()