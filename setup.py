from typing import Any
from setuptools import setup
from os import getenv


def build_optional_cpp_extensions(
    ext_modules: list[str], cmdclass: dict[str, Any]
) -> tuple[list[str], dict[str, type]]:
    """
    Build optional cpp extentions using pybind11 and torch cpp_extension.
    Controlled by env variables CAYLEYPY_BUILD_CPP and CAYLEYPY_INCLUDE_OPENMP.

    If CAYLEYPY_BUILD_CPP is set to 1, the cpp extensions will be built.
    If CAYLEYPY_INCLUDE_OPENMP is set to 1, OpenMP flags will be included.

    Currently only Linux-like systems are supported for OpenMP.
    """

    CAYLEYPY_BUILD_CPP = getenv("CAYLEYPY_BUILD_CPP", "0")
    CAYLEYPY_INCLUDE_OPENMP = getenv("CAYLEYPY_INCLUDE_OPENMP", "1")
    if CAYLEYPY_BUILD_CPP == "1":
        try:
            import pybind11
            from torch.utils import cpp_extension

            print("CAYLEYPY_BUILD_CPP is set to 1, building CayleyPy with optional torch cpp extensions")

            extra_compile_args = {"cxx": ["-std=c++17"]}
            extra_link_args = []

            if CAYLEYPY_INCLUDE_OPENMP != "0":
                extra_compile_args["cxx"].append("-fopenmp")
                extra_link_args.append("-fopenmp")

            ext_modules.append(
                cpp_extension.CppExtension(
                    "cayleypy.cpp_algo",
                    [
                        "cayleypy/cpp_algo/pybind11.cpp",
                        "cayleypy/cpp_algo/random_walks.cpp",
                    ],
                    include_dirs=[pybind11.get_include()],
                    language="c++",
                    extra_compile_args=extra_compile_args,
                    extra_link_args=extra_link_args,
                ),
            )
            cmdclass["build_ext"] = cpp_extension.BuildExtension

        except Exception as e:
            if CAYLEYPY_BUILD_CPP:
                raise Exception(
                    f"""
                    Got the following exception during setting up cpp extensions:
                    \n{e}.\n 
                    CAYLEYPY_BUILD_CPP is set to {CAYLEYPY_BUILD_CPP}. 
                    If you want to install CayleyPy without cpp extensions, set it to 0.
                    """
                )
            else:
                raise Exception(e)
    else:
        print("CAYLEYPY_BUILD_CPP is not set to 1, not building cpp extensions.")

    return ext_modules, cmdclass


if __name__ == "__main__":
    ext_modules = []
    cmdclass = {}
    ext_modules, cmdclass = build_optional_cpp_extensions(ext_modules, cmdclass)
    setup(ext_modules=ext_modules, cmdclass=cmdclass)
