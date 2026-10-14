from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="COBALT",
    version="1.0.0",
    author="Mohd Mujtaba Akhtar, Girish, Sanjam Wadhwa, Muskaan Singh, Ning Ma",
    description="Non-Invasive Tuberculosis Detection from Cough Audio",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Helixometry/COBALT",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
