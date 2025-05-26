from setuptools import setup,find_packages
from typing import List

HYPEN="-e ."
def get_packages(file_path:str)->List[str]:
    """Get all packages from a file path."""
    packages = []
    with open(file_path, encoding='utf-8') as f:
        packages=f.readlines()
        packages=[req.replace("\n"," ") for req in packages]
    if HYPEN in packages:
        packages.remove(HYPEN)
    return packages

setup(
    name='ml',
    version='1.0',
    author='Raghav',
    packages=find_packages(),
    install_requires=get_packages('requirements.txt')
    )