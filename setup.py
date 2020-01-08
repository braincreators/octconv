from setuptools import setup
from os import path
import os


this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


requirements = [
    'torch>=0.4.1'
]

setup(
    name='octconv',
    version=os.getenv('TRAVIS_TAG', '0.1.0'),
    packages=['octconv'],
    url='https://github.com/braincreators/octconv',
    license='MIT',
    install_requires=requirements,
    author='BrainCreators',
    author_email='miguel@braincreators.com',
    description='Octave Convolution Implementation in PyTorch',
    long_description=long_description,
    long_description_content_type='text/markdown'
)
