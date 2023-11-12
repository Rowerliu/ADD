from setuptools import setup

setup(
    name="ADBD",
    py_modules=["ADBD"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)
