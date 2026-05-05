from setuptools import setup, find_packages

setup(
    name="vibranium",
    version="0.1.0",
    description="Vibranium stationary portfolio mean-reversion framework",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "statsmodels",
    ],
    python_requires=">=3.7",
)
