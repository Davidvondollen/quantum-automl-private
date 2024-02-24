from setuptools import setup, find_packages


with open("quautoml/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")


setup(
    name='quautoml',
    version=version,
    author='David Von Dollen',
    install_requires=["numpy~=1.24", "scikit-learn~=1.2"],
    packages=find_packages(),
    include_package_data=True
)
