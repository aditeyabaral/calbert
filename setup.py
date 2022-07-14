import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="calbert",
    version="0.1.0",
    author="Aditeya Baral <aditeya.baral@gmail.com>, Aronya Baksy <abaksy@gmail.com>, Ansh Sarkar <anshsarkar1@gmail.com>, Deeksha D <deekshad132@gmail.com>",
    author_email="aditeya.baral@gmail.com",
    maintainer="Aronya Baksy",
    maintainer_email="abaksy@gmail.com",
    description="Code-mixed Adaptive Language representations using BERT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aditeyabaral/calbert",
    packages=setuptools.find_packages(),
    install_requires=[
        "tqdm==4.64.0",
        "transformers==4.20.1"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
