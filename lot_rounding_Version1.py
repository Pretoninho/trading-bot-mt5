import math

def round_down_to_step(x: float, step: float) -> float:
    return math.floor(x / step) * step