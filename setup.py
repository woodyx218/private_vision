import os
import re

import setuptools

# for simplicity we actually store the version in the __version__ attribute in the source
here = os.path.realpath(os.path.dirname(__file__))
with open(os.path.join(here, 'private_vision', '__init__.py')) as f:
    meta_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if meta_match:
        version = meta_match.group(1)
    else:
        raise RuntimeError("Unable to find __version__ string.")

with open(os.path.join(here, 'README.md'), encoding="utf8") as f:
    readme = f.read()

setuptools.setup(
    name="private_vision",
    version=version,
    author="Zhiqi Bu, Jialin Mao, Shiyun Xu",
    author_email="zbu@upenn.edu",
    description="Train convolutional vision transformers and CNN with differential privacy.",
    long_description=readme,
    url="https://github.com/woodyx218/private_vision",
    packages=setuptools.find_packages(exclude=['examples', 'tests']),
    install_requires=[
        "torch>=1.8.0",
        "prv-accountant",
        "transformers",
        "numpy",
        "scipy",
        "opacus>=1.0",
        "timm>=0.6.2",
    ],
    python_requires='~=3.7',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
)
