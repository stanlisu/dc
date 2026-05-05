from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "agamotto.core",
        ["src/agamotto/core.py"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "agamotto.lib_binance",
        ["src/agamotto/lib_binance.py"],
    ),
]

setup(
    name="agamotto",
    version="0.1.0",
    description="Agamotto research and trading pipeline",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
        }
    ),
    install_requires=[
        "pandas",
        "numpy",
        "joblib",
        "requests",
        "cython",
    ],
    python_requires=">=3.7",
    zip_safe=False,
)
