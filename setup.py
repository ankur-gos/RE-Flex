from setuptools import setup, find_packages
# TODO: Transformers version
setup(
    name="reflex",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'dataclasses==0.7',
        'requests==2.23.0',
        'fairseq==0.8.0',
        'fastBPE==0.1.0',
        'fasttext==0.9.1',
        'pandas==0.23.4',
        'numpy==1.15.1',
        'scipy==1.3.2',
        'PyYAML==5.1.2',
        'tqdm>=4.27.0',
        'transformers==2.2.0',
        'gensim==3.8.1',
        'sacred==0.8.1',
        'pymongo==3.10.1',
        'torch==1.2.0'
    ]
)
