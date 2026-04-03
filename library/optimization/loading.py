import importlib


def load_target(target: str):
    module_name, attr_name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)
