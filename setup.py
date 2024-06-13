from setuptools import setup, find_packages

REQUIRED = [
    'numpy==1.23.5',
    'scikit-learn==1.3.0',
    'pandas==1.5.0',
    'scipy==1.11.3',
    'seaborn==0.12.2',
    'matplotlib==3.7.2',
    'tqdm==4.66.1',
    'scanpy==1.9.4',
    'anndata==0.9.2',
    'scvelo==0.2.5',
    'statsmodels==0.14.0',
    'torch==2.0.0',
    'torchdiffeq==0.2.3',
    'numba==0.56.4',
    'networkx==3.1',
]

setup(
    name='neurovelo',
    version='1.0.0',
    author='Idris Kouadri Boudjelthia',
    author_email='ikouadri@sissa.it',
    description='Interpretable learning of cellular dynamics and gene interactions network',
    packages=find_packages(),
    install_requires=REQUIRED,
    python_requires='>=3.8, <3.10',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
)
