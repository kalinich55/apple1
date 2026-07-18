import unittest

from tracker import ObjectTracker


class ObjectTrackerTests(unittest.TestCase):
    def test_fast_motion_is_allowed_when_its_speed_is_plausible(self):
        tracker = ObjectTracker()
        self.assertTrue(tracker.add_point(0.0, 0.0, timestamp=0.0))
        self.assertTrue(tracker.add_point(500.0, 0.0, timestamp=0.1))

    def test_impossible_jump_is_rejected(self):
        tracker = ObjectTracker()
        tracker.add_point(0.0, 0.0, timestamp=0.0)
        self.assertFalse(tracker.add_point(2_000.0, 0.0, timestamp=0.1))
        self.assertEqual(len(tracker), 1)

    def test_duplicate_timestamp_is_rejected(self):
        tracker = ObjectTracker()
        tracker.add_point(0.0, 0.0, timestamp=1.0)
        self.assertFalse(tracker.add_point(1.0, 1.0, timestamp=1.0))


if __name__ == "__main__":
    unittest.main()
