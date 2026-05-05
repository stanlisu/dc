from setuptools import setup, find_packages

setup(
    name="stormbreaker",
    version="0.1.0",
    description="Stormbreaker ultra-high-frequency research system",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pandas",
        "numpy",
        "joblib",
    ],
    python_requires=">=3.7",
)
