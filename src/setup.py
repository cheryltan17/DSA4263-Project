from setuptools import find_packages, setup

# Read requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='DSA4262-Project',
    version='0.1.0',
    description='Job Fraud Detection',
    author='DSA4263',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=requirements,
    include_package_data=True,
    zip_safe=False,
)
