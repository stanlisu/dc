from setuptools import setup, find_packages

setup(
    name="mjolnir",
    version="0.1.0",
    description="Mjolnir tick-data ML research and trading system",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pandas",
        "numpy",
        "joblib",
    ],
    python_requires=">=3.7",
)
