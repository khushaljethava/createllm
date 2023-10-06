
import os
import sys

from setuptools import setup, find_packages

def read(rel_path: str) -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with open(os.path.join(here, rel_path)) as fp:
        return fp.read()


VERSION = '0.1.4' 
DESCRIPTION = 'Python package that let you create own transformers based models on your own data'
LONG_DESCRIPTION = read("README.rst")

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="createllm", 
        version=VERSION,
        author="Khushal Jethava",
        author_email="Khushaljethava14@outlook.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        url='https://github.com/khushaljethava/createllm',
        packages=find_packages(),
        install_requires=['dill','torch','torchvision'], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'first package'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)