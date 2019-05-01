from setuptools import setup


requirements = [
    'torch>=0.4.1'
]

setup(
    name='octconv',
    version='0.1.0',
    packages=['octconv'],
    url='',
    license='MIT',
    install_requires=requirements,
    author='Miguel Varela Ramos',
    author_email='miguelvramos92@gmail.com',
    description='Octave Convolution Implementation in PyTorch'
)
