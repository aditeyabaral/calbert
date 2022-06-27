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
    url="https://github.com/aditeyabaral/calbert",
    packages=setuptools.find_packages(),
    install_requires=['brotlipy==0.7.0', 'certifi==2022.5.18.1', 'cffi==1.15.0', 'charset-normalizer==2.0.4', 
            'click==8.0.3', 'colorama==0.4.4', 'cryptography==36.0.0', 'filelock==3.4.0', 
            'huggingface-hub==0.7.0', 'idna==3.1', 'importlib-metadata==4.8.2', 'joblib==1.1.0', 
            'mkl-fft==1.3.1', 'mkl-random==1.2.2', 'mkl-service==2.4.0', 'nltk==3.6.5', 'numpy==1.22.3', 
            'olefile==0.46', 'packaging==21.3', 'pandas==1.4.2', 'Pillow==8.4.0', 'pip==21.2.4', 
            'pycparser==2.21', 'pyOpenSSL==21.0.0', 'pyparsing==3.0.4', 'PySocks==1.7.1', 
            'python-dateutil==2.8.2', 'pytz==2021.3', 'PyYAML==6.0', 'regex==2022.3.15', 
            'requests==2.26.0', 'sacremoses==0.0.43', 'scipy==1.7.3', 'selenium==3.141.0', 
            'setuptools==58.0.4', 'six==1.16.0', 'tokenizers==0.12.1', 'torch==1.11.0', 
            'torchaudio==0.11.0', 'torchvision==0.12.0', 'tqdm==4.62.3', 'transformers==4.20.0', 
            'typing-extensions==3.10.0.2', 'urllib3==1.26.7', 'wheel==0.37.0', 
            'win-inet-pton==1.1.0', 'wincertstore==0.2', 'zipp==3.6.0'
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
