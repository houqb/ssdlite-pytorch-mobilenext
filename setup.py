from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="ssdlite-pytorch",
    version="1.0",
    packages=find_packages(exclude=['ext']),
    install_requires=[
        "pytorch~=1.2",
        "torchvision~=0.4",
        "opencv-python~=4.0",
        "yacs==0.1.6",
        "Vizer~=0.1.4",
    ],
    author="Qibin Hou",
    author_email="andrewhoux@gmail.com",
    description="Implementation of SSDLite in PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Andrew-Qibin/ssdlite-pytorch",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="MIT",
    python_requires=">=3.6",
    include_package_data=True,
)
