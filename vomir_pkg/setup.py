from setuptools import setup, find_packages

setup(
    name="vomir",
    version="0.1.0",
    description="Vomir mean-reversion classifier and trading system",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pandas",
        "numpy",
        "joblib",
    ],
    python_requires=">=3.7",
)
