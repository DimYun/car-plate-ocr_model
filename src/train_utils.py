"""Module containing utility."""
import importlib
import typing as tp


def load_object(obj_path: str, default_obj_path: str = "") -> tp.Any:
    """Load an object from config.

    :param obj_path: path to python module for upload
    :param default_obj_path: paramaters for upload
    :return: loaded object
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object `{obj_name}` cannot be loaded from `{obj_path}`.")
    return getattr(module_obj, obj_name)
