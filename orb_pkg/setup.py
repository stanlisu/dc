from setuptools import setup, find_packages

setup(
    name="orb",
    version="0.1.0",
    description="Orb cross-timeframe feature alignment system",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pandas",
        "numpy",
        "joblib",
    ],
    python_requires=">=3.7",
)
