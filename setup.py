from setuptools import setup, find_packages

REQUIRED = [
    'numpy>=1.20',
    'scikit-learn>=0.22',
    'pandas',
    'scipy>=1.4.1',
    'seaborn',
    'matplotlib>=3.3.0',
    'tqdm',
    'scanpy>=1.5',
    'anndata>=0.7.5',
    'scvelo>=0.2.2',
    'IPython',
    'ipykernel',
    'IProgress',
    'ipywidgets',
    'jupyter',
    'torch',
    'torchdiffeq'
]

setup(
    name='neurovelo',
    version='0.0.1',
    author='Idris Kouadri Boudjelthia',
    author_email='ikouadri@sissa.it',
    description='Interpretable learning of cellular dynamics',
    packages=find_packages(),
    install_requires=REQUIRED,
    python_requires='>=3.7',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
)
