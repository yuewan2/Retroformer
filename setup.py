import os
from setuptools import setup, find_packages

long_description = '''Retroformer'''

setup(
    name='retroformer',
    version='0.0.1',
    author='Yue Wan',
    author_email='wanyue1996@gmail.com',
    py_modules=['retroformer'],
    description='Retroformer',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
              'lmdb',
          ],
    license='MIT',
    packages=find_packages()
)
