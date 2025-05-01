from setuptools import setup, find_packages

setup(
    name="maze",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.26.0",
        "pygame>=2.1.0",
        "numpy>=1.21.0"
    ],
)