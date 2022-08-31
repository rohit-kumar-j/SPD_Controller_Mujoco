#!/usr/bin/env python3
import os
from setuptools import find_packages, setup

VERSION = "0.1.0"

INSTALL_REQUIRES = [
    "mujoco >= 2.1.5",
    "glfw >= 2.5.0",
    "numpy >= 1.18.0",
    "imageio",
]

setup(
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
    python_requires=">=3.6",
    zip_safe=False,
)
