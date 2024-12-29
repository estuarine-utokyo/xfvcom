from setuptools import setup, find_packages

setup(
    name="xfvcom",
    version="0.1",
    description="Tools for preprocessing and postprocessing FVCOM data using xarray.",
    author="Jun SASAKI",
    author_email="jsasaki.ece@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "xarray",
        "matplotlib",
        "pyproj",
        "scikit-learn"
    ],
    python_requires=">=3.11",
)

