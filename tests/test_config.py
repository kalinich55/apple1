import unittest

import config


class ObjectProfileTests(unittest.TestCase):
    def test_selected_profile_has_detector_fields(self):
        profile = config.get_object_profile()
        self.assertEqual(set(profile), {"model_path", "target_class_name", "display_name"})

    def test_known_tennis_ball_profile_matches_model_class(self):
        profile = config.OBJECT_PROFILES["tennis_ball"]
        self.assertEqual(profile["model_path"], "best_tennis_ball_50epochs.pt")
        self.assertEqual(profile["target_class_name"], "ball")


if __name__ == "__main__":
    unittest.main()
