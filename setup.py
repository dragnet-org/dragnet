
from distutils.core import setup

setup(name='Dragnet',
      version='0.2',
      description='Train/evaluate page authoriy models.\nA flexible ranking model training/predicting library with abstracted regression, loss functions and optimization methods.\nInput transforming for structured data.\nPCA\nA map reduce framework for working with large data sets.',
      author='Matt Peters',
      author_email='matt@seomoz.org',
      package_dir = {'dragnet': 'dragnet'},
      packages = ['dragnet']
)


