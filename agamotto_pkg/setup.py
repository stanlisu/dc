from setuptools import setup, find_packages

setup(
    name="agamotto",
    version="0.1.0",
    description="Agamotto research and trading pipeline",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pandas",
        "numpy",
        "joblib",
        "requests",
    ],
    python_requires=">=3.7",
)

