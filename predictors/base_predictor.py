class BasePredictor:
    def update(self, x, y, t):
        raise NotImplementedError

    def predict_future(self, delta_t):
        raise NotImplementedError

    def get_current_state(self):
        raise NotImplementedError

    def is_ready(self):
        raise NotImplementedError

    def reset(self):
        pass