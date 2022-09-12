import setuptools

setuptools.setup(
    name="calbert",
    version="1.0.3",
    author="Aditeya Baral",
    author_email="aditeya.baral@gmail.com",
    maintainer="Aronya Baksy",
    maintainer_email="abaksy@gmail.com",
    description="Code-mixed Adaptive Language representations using BERT",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    keywords="NLP deep learning transformer pytorch BERT",
    url="https://github.com/aditeyabaral/calbert",
    download_url="https://pypi.org/project/calbert/",
    python_requires=">=3.7.0",
    license="MIT",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch==1.12.0",
        "tqdm==4.64.0",
        "transformers==4.20.1",
        "tensorboard==2.9.1"
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
