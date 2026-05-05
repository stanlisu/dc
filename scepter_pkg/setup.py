from setuptools import setup, find_packages

setup(
    name="scepter",
    version="0.1.0",
    description="Scepter cross-symbol anchor features",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pandas",
        "numpy",
        "joblib",
    ],
    python_requires=">=3.7",
)
