from importlib import import_module
from logging import getLogger

logger = getLogger()


# pylint: disable=W0718
def import_cpp_extension(function_name: str):
    try:
        module = import_module(".cpp_algo", package=__package__)
        return getattr(module, function_name)
    except Exception as e:
        logger.info(
            "Could not import CayleyPy C++ extension: '%s'. "
            "Most likely the C++ extensions are not installed or this extension does not exist. "
            "The exception: %s",
            function_name,
            e,
        )
        return None


# pylint: enable=W0718
