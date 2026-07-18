from .linear_predictor import LinearPredictor
from .poly2_predictor import Poly2Predictor
from .poly3_predictor import Poly3Predictor
from .kalman_accel_predictor import KalmanAccelPredictor
from .kalman_cv_predictor import KalmanCVPredictor
from .collision_aware_predictor import CollisionAwarePredictor


def create_predictor(name):
    name = name.lower()

    if name == "linear":
        return LinearPredictor()
    elif name == "poly2":
        return Poly2Predictor()
    elif name == "poly3":
        return Poly3Predictor()
    elif name == "kalman_ca":
        return KalmanAccelPredictor()
    elif name == "kalman_cv":
        return KalmanCVPredictor()
    elif name == "collision_ca":
        return CollisionAwarePredictor()
    else:
        raise ValueError(f"Unknown predictor: {name}")
