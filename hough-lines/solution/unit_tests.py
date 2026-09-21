import unittest
import numpy as np
import cv2
from detection import traffic_light_detection, construction_sign_detection

class TestTrafficLightDetection(unittest.TestCase):
    def setUp(self):
        # radius values for circle detection
        self.radii = range(5, 30)
        # allowed pixel margin for centroid checks
        self.margin = 2

        # Base-case placeholders: (image_path, (expected_center, expected_state))
        self.base_tests = [
            ('test/scene_tl.png', ((438, 250), 'green')),
            ('test/scene_tl2.png', ((606, 448), 'yellow')),
            ('test/scene_tl3.png', ((172, 252), 'red')),
            ('test/scene_tl4.png', ((408, 122), 'green')),
        ]

    def test_base_cases(self):
        for img_path, (exp_center, exp_state) in self.base_tests:
            with self.subTest(img=img_path):
                img = cv2.imread(img_path)
                (det_center, det_state) = traffic_light_detection(img, self.radii)

                # check centroid within margin
                self.assertTrue(
                    abs(det_center[0] - exp_center[0]) <= self.margin and
                    abs(det_center[1] - exp_center[1]) <= self.margin,
                    f"Detected center {det_center} not within {self.margin}px of expected {exp_center}"
                )
                # state must still match exactly
                self.assertEqual(det_state, exp_state)

    def test_non_array_input(self):
        with self.assertRaises(TypeError):
            traffic_light_detection("not an array", self.radii)

    def test_empty_array(self):
        empty = np.array([], dtype=np.uint8)
        with self.assertRaises(ValueError):
            traffic_light_detection(empty, self.radii)

    def test_wrong_dimensions_and_channels(self):
        # 2D image
        img2d = np.zeros((100, 100), dtype=np.uint8)
        with self.assertRaises(ValueError):
            traffic_light_detection(img2d, self.radii)
        # 4-channel image
        img4c = np.zeros((100, 100, 4), dtype=np.uint8)
        with self.assertRaises(ValueError):
            traffic_light_detection(img4c, self.radii)

    def test_non_iterable_radii(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        with self.assertRaises(TypeError):
            traffic_light_detection(img, None)

    def test_empty_radii(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            traffic_light_detection(img, [])

    def test_invalid_radii_values(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            traffic_light_detection(img, range(-5, 0))

    def test_radii_too_large_for_image(self):
        small_img = np.zeros((20, 20, 3), dtype=np.uint8)
        # maxRadius * 2 = 12*2 = 24 > min_dimension=20
        with self.assertRaises(ValueError):
            traffic_light_detection(small_img, range(12, 13))

    def test_no_circles_detected(self):
        blank = np.zeros((200, 200, 3), dtype=np.uint8)
        # no circles
        with self.assertRaises(ValueError):
            traffic_light_detection(blank, range(5, 10))


class TestConstructionSignDetection(unittest.TestCase):
    def setUp(self):
        # allowed pixel margin for centroid checks
        self.margin = 2

        # Base-case placeholders: (image_path, expected_centroid)
        self.base_tests = [
            ('test/scene_constr.png', (251, 400)),
            ('test/scene_constr2.png', (599, 283)),
            ('test/scene_constr3.png', (667, 142)),
            ('test/scene_constr4.png', (298, 485)),
        ]

    def test_base_cases(self):
        for img_path, exp_centroid in self.base_tests:
            with self.subTest(img=img_path):
                img = cv2.imread(img_path)
                det_centroid = construction_sign_detection(img)

                # check centroid within margin
                self.assertTrue(
                    abs(det_centroid[0] - exp_centroid[0]) <= self.margin and
                    abs(det_centroid[1] - exp_centroid[1]) <= self.margin,
                    f"Detected centroid {det_centroid} not within {self.margin}px of expected {exp_centroid}"
                )

    def test_non_array_input(self):
        with self.assertRaises(TypeError):
            construction_sign_detection("not an array")

    def test_empty_array(self):
        empty = np.array([], dtype=np.uint8)
        with self.assertRaises(ValueError):
            construction_sign_detection(empty)

    def test_invalid_dimensions(self):
        arr1d = np.zeros((10,), dtype=np.uint8)
        with self.assertRaises(ValueError):
            construction_sign_detection(arr1d)

    def test_invalid_channel_count(self):
        img2ch = np.zeros((100, 100, 2), dtype=np.uint8)
        with self.assertRaises(ValueError):
            construction_sign_detection(img2ch)

    def test_no_lines_detected(self):
        blank = np.zeros((200, 200), dtype=np.uint8)
        with self.assertRaises(ValueError):
            construction_sign_detection(blank)


if __name__ == '__main__':
    unittest.main()
