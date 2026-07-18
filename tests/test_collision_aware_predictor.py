import unittest

from predictors.collision_aware_predictor import CollisionAwarePredictor
from predictors.factory import create_predictor


class CollisionAwarePredictorTests(unittest.TestCase):
    def _predictor_approaching_right_wall(self):
        predictor = CollisionAwarePredictor()
        predictor.set_environment(960, 540, 20, walls={"right": 0.84}, enabled=True)
        for timestamp, x in ((0.0, 850.0), (1.0 / 30.0, 880.0), (2.0 / 30.0, 910.0)):
            predictor.update(x, 150.0, timestamp)
        return predictor

    def test_reflects_before_a_known_wall(self):
        predictor = self._predictor_approaching_right_wall()

        x_pred, y_pred = predictor.predict_future(0.08)

        # At t=0.1 the center reaches x=940, then travels back at 84% of
        # 900 px/s for the remaining 46.67 ms.
        self.assertAlmostEqual(x_pred, 904.72, places=2)
        self.assertAlmostEqual(y_pred, 150.0, places=6)

    def test_without_active_walls_behaves_like_plain_kalman_forecast(self):
        predictor = CollisionAwarePredictor()
        for timestamp, x in ((0.0, 850.0), (1.0 / 30.0, 880.0), (2.0 / 30.0, 910.0)):
            predictor.update(x, 150.0, timestamp)

        x_pred, _ = predictor.predict_future(0.08)

        self.assertAlmostEqual(x_pred, 982.0, places=2)

    def test_factory_exposes_collision_predictor(self):
        self.assertIsInstance(create_predictor("collision_ca"), CollisionAwarePredictor)


if __name__ == "__main__":
    unittest.main()
