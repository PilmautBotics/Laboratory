"""File to enable a pip installation"""

import setuptools

with open('README.md', 'r') as readme_file:
    long_description = readme_file.read()

setuptools.setup(
    name='laboratory',
    version='0.1.0',
    author='Pierre Guillemaut',
    author_email='',
    description='',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    python_requires='>=3.6.8',
    install_requires=[
        'tqdm>=4.31.1',
        'bidict>=0.21.2',
        'torch>=1.1.0',
        'tensorboard>=1.14.0',
        'webcolors>=1.1.11'
    ]
)