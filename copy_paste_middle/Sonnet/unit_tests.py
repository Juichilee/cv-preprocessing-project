import unittest
import numpy as np
from cv_preprocess import * 

class TestCopyPasteMiddle(unittest.TestCase):
    def setUp(self):
        # small source and destination for reuse
        self.src = np.arange(16).reshape(4,4)
        self.dst = np.zeros((6,6), dtype=int)

    # Error cases
    def test_non_numpy_inputs(self):
        with self.assertRaises(ValueError):
            copy_paste_middle([[1,2],[3,4]], self.dst, (1,1))
        with self.assertRaises(ValueError):
            copy_paste_middle(self.src, [[0]*6]*6, (1,1))

    def test_non_2d_inputs(self):
        with self.assertRaises(ValueError):
            copy_paste_middle(np.zeros((3,3,3)), self.dst, (1,1))
        with self.assertRaises(ValueError):
            copy_paste_middle(self.src, np.zeros((6,6,1)), (1,1))

    def test_shape_non_positive(self):
        with self.assertRaises(ValueError):
            copy_paste_middle(self.src, self.dst, (0,2))
        with self.assertRaises(ValueError):
            copy_paste_middle(self.src, self.dst, (2,-1))

    def test_shape_larger_than_src(self):
        # height too large
        with self.assertRaises(ValueError):
            copy_paste_middle(self.src, self.dst, (5,2))
        # width too large
        with self.assertRaises(ValueError):
            copy_paste_middle(self.src, self.dst, (2,5))

    def test_shape_larger_than_dst(self):
        # dst height too small
        with self.assertRaises(ValueError):
            copy_paste_middle(self.src, np.zeros((3,6)), (4,2))
        # dst width too small
        with self.assertRaises(ValueError):
            copy_paste_middle(self.src, np.zeros((6,3)), (2,4))

    # Normal cases
    def test_copy_center_even_shape(self):
        # copy a 2x2 block from center of 4x4 into 6x6
        out = copy_paste_middle(self.src, self.dst, (2,2))
        # center of src is between rows 1-2, cols 1-2: block = [[5,6],[9,10]]
        expected_patch = np.array([[5,6],[9,10]])
        # pasted at center of 6x6 (rows 2-3, cols 2-3)
        np.testing.assert_array_equal(out[2:4,2:4], expected_patch)
        # outside stays zero
        self.assertTrue(np.all(out[:2] == 0))
        self.assertTrue(np.all(out[4:] == 0))

    def test_copy_center_odd_shape(self):
        # for (3,3), half_h = half_w = 1 => block = src[1:3,1:3] = [[5,6],[9,10]]
        dst = np.ones((5,5), dtype=int)
        out = copy_paste_middle(self.src, dst, (3,3))
        
        expected_patch = np.array([[5,6],[9,10]])
        # inserted at dst[1:3,1:3]
        np.testing.assert_array_equal(out[1:3,1:3], expected_patch)

        # everywhere else remains 1
        mask = np.ones_like(dst, dtype=bool)
        mask[1:3,1:3] = False
        self.assertTrue(np.all(out[mask] == 1))

class TestCopyPasteMiddleCircle(unittest.TestCase):
    def setUp(self):
        # 5×5 source and destination
        self.src = np.arange(25).reshape(5,5)
        self.dst = np.zeros((5,5), dtype=int)

    # Error cases
    def test_non_numpy_inputs(self):
        with self.assertRaises(ValueError):
            copy_paste_middle_circle([[0]], self.dst, 1)
        with self.assertRaises(ValueError):
            copy_paste_middle_circle(self.src, [[0]], 1)

    def test_non_2d_inputs(self):
        with self.assertRaises(ValueError):
            copy_paste_middle_circle(np.zeros((2,2,2)), self.dst, 1)
        with self.assertRaises(ValueError):
            copy_paste_middle_circle(self.src, np.zeros((5,5,1)), 1)

    def test_invalid_radius(self):
        with self.assertRaises(ValueError):
            copy_paste_middle_circle(self.src, self.dst, -1)
        with self.assertRaises(ValueError):
            copy_paste_middle_circle(self.src, self.dst, "large")

    # Normal & edge cases
    def test_radius_zero(self):
        out = copy_paste_middle_circle(self.src, self.dst, 0)
        cx = (5-1)//2
        self.assertEqual(out[cx,cx], self.src[cx,cx])
        mask = np.ones_like(self.dst, dtype=bool)
        mask[cx, cx] = False
        self.assertTrue(np.all(out[mask] == 0))

    def test_small_radius(self):
        out = copy_paste_middle_circle(self.src, self.dst, 1)
        cx, cy = (5-1)//2, (5-1)//2

        # Expected positions
        expected = []
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                if dx*dx + dy*dy <= 1:
                    expected.append((cx+dx, cy+dy))

        for x,y in expected:
            self.assertEqual(out[x,y], self.src[x,y])

        # corners excluded
        self.assertEqual(out[cx+1,cy+1], 0)
        self.assertEqual(out[cx-1,cy-1], 0)

    def test_large_radius_covers_all(self):
        out = copy_paste_middle_circle(self.src, self.dst, 10)
        np.testing.assert_array_equal(out, self.src)

class TestImageStats(unittest.TestCase):
    # Error cases
    def test_non_numpy_input(self):
        with self.assertRaises(ValueError):
            image_stats([[1,2],[3,4]])

    def test_non_2d_array(self):
        with self.assertRaises(ValueError):
            image_stats(np.zeros((3,3,3)))

    # Normal cases
    def test_uniform_array(self):
        arr = np.full((4,4), 7)
        mn, mx, avg, std = image_stats(arr)
        self.assertEqual(mn, 7.0)
        self.assertEqual(mx, 7.0)
        self.assertEqual(avg, 7.0)
        self.assertEqual(std, 0.0)

    def test_varying_array(self):
        arr = np.array([[0,2],[4,6]], dtype=float)
        mn, mx, avg, std = image_stats(arr)
        self.assertEqual(mn, 0.0)
        self.assertEqual(mx, 6.0)
        self.assertAlmostEqual(avg, 3.0)
        self.assertAlmostEqual(std, np.sqrt(5))

if __name__ == '__main__':
    unittest.main()
