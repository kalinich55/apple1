import unittest

from detector import estimate_center


class EstimateCenterTests(unittest.TestCase):
    def test_prefers_bbox_center_when_mask_centroid_is_far(self):
        center = estimate_center(mask_center=(10, 20), bbox=(0, 0, 100, 100), confidence=0.99)
        self.assertEqual(center, (50.0, 50.0))

    def test_blends_center_when_mask_centroid_is_close(self):
        center = estimate_center(mask_center=(49, 51), bbox=(40, 40, 60, 60), confidence=0.99)
        self.assertAlmostEqual(center[0], 49.3)
        self.assertAlmostEqual(center[1], 50.7)


if __name__ == "__main__":
    unittest.main()
