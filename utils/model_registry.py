from typing import Callable

MODEL_REGISTRY: dict[str, Callable] = {}

def register_model(name: str):
    def decorator(fn: Callable):
        MODEL_REGISTRY[name.upper()] = fn
        return fn
    return decorator