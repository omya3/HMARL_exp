import numpy as np


class Utility:
    def __init__(self) -> None:
        pass

    def get_value(self, mu, sigma):
        return np.ceil(np.random.normal(mu, sigma, 1))

    def generate_slav(self, sl_class):
        if sl_class == "C1":
            num = np.random.exponential(3.45)
            if num > 15:
                return 1
        elif sl_class == "C2":
            num = np.random.uniform(0, 1)
            if num > 0.9:
                return 1
        else:
            num = np.random.uniform(0, 1)
            if num > 0.95:
                return 1
