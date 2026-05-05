from setuptools import setup, find_packages

setup(
    name="aether",
    version="0.1.0",
    description="Aether pooled cross-TF ML with regime stacks",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pandas",
        "numpy",
        "joblib",
    ],
    python_requires=">=3.7",
)
