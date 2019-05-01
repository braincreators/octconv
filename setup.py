from setuptools import setup


requirements = [
    'torch>=0.4.1'
]

setup(
    name='octconv',
    version='0.1.0',
    packages=['octconv'],
    url='https://github.com/braincreators/octconv',
    license='MIT',
    install_requires=requirements,
    author='BrainCreators',
    author_email='miguel@braincreators.com',
    description='Octave Convolution Implementation in PyTorch'
)
