import unittest
import numpy as np
import cv2
from template_match import template_match  # adjust module name/path as needed

class TestTemplateMatch(unittest.TestCase):
    def setUp(self):
        # a small image with a 2×2 non‑constant patch placed at (2, 1)
        self.img = np.zeros((5, 5), dtype=float)
        # non‑constant template so variance > 0
        self.tmpl = np.array([[1, 2],
                              [3, 4]], dtype=float)
        # embed template at x=1, y=2
        self.img[2:4, 1:3] = self.tmpl

    # Basic Cases

    def test_tm_ssd_basic(self):
        loc = template_match(self.img, self.tmpl, "tm_ssd")
        self.assertEqual(loc, (1, 2))

    def test_tm_nssd_basic(self):
        loc = template_match(self.img, self.tmpl, "tm_nssd")
        self.assertEqual(loc, (1, 2))

    def test_tm_ccor_basic(self):
        loc = template_match(self.img, self.tmpl, "tm_ccor")
        self.assertEqual(loc, (1, 2))

    def test_tm_nccor_basic(self):
        loc = template_match(self.img, self.tmpl, "tm_nccor")
        self.assertEqual(loc, (1, 2))

    # Type and dimension errors

    def test_type_error_img_not_array(self):
        with self.assertRaises(TypeError):
            template_match([[1,2],[3,4]], self.tmpl, "tm_ssd")

    def test_type_error_tmpl_not_array(self):
        with self.assertRaises(TypeError):
            template_match(self.img, [[1,2],[3,4]], "tm_ssd")

    def test_value_error_img_not_2d(self):
        bad_img = np.zeros((3,3,3))
        with self.assertRaises(ValueError) as ctx:
            template_match(bad_img, self.tmpl, "tm_ssd")

    def test_value_error_tmpl_not_2d(self):
        bad_tmpl = np.zeros((2,2,1))
        with self.assertRaises(ValueError) as ctx:
            template_match(self.img, bad_tmpl, "tm_ssd")

    # Empty‑array errors

    def test_value_error_img_empty(self):
        empty = np.zeros((0,0))
        with self.assertRaises(ValueError) as ctx:
            template_match(empty, np.ones((1,1)), "tm_ssd")

    def test_value_error_tmpl_empty(self):
        empty = np.zeros((0,0))
        with self.assertRaises(ValueError) as ctx:
            template_match(self.img, empty, "tm_ssd")

    # Template‑size error

    def test_value_error_template_larger_than_image(self):
        large_tmpl = np.zeros((6,6))
        with self.assertRaises(ValueError):
            template_match(self.img, large_tmpl, "tm_ssd")

    # Unsupported method

    def test_not_implemented_unknown_method(self):
        with self.assertRaises(NotImplementedError):
            template_match(self.img, self.tmpl, "tm_foobar")

    # Zero‑variance template errors

    def test_value_error_zero_variance_nssd(self):
        const = np.full((2,2), 5.0)
        with self.assertRaises(ValueError) as ctx:
            template_match(self.img, const, "tm_nssd")

    def test_value_error_zero_variance_nccor(self):
        const = np.full((2,2), 7.0)
        with self.assertRaises(ValueError) as ctx:
            template_match(self.img, const, "tm_nccor")

if __name__ == "__main__":
    unittest.main()
