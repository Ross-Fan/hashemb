from setuptools import setup, Extension
import pybind11
import sys

# Determine compile flags
extra_compile_args = []
extra_link_args = []
if sys.platform == "darwin":
    extra_compile_args += ["-std=c++17", "-O2", "-Wall"]
else:
    extra_compile_args += ["-std=c++17", "-O2", "-Wall", "-fopenmp"]
    extra_link_args += ["-fopenmp"]

ext_modules = [
    Extension(
        "hashemb._hashemb_cpp",
        sources=[
            "csrc/hash_table.cpp",
            "csrc/embedding_table.cpp",
            "csrc/pybind_binding.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
        ],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="hashemb",
    version="0.1.0",
    description="Host-memory Hash Embedding Table for PyTorch",
    ext_modules=ext_modules,
    install_requires=["pybind11", "torch"],
    python_requires=">=3.8",
)
