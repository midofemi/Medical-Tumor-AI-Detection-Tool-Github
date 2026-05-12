from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="Medical Tumor AI Detection Tool with XAI",
    version="0.1",
    author="Amidu Kamara",
    packages=find_packages(),
    install_requires=requirements,
)