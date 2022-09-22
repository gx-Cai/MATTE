from setuptools import setup,find_packages

## Adding following code if Requirements install needed.
# import os
# os.system("pip3 install -r requirements.txt")

requires_packages = [
    "scikit-learn", "numpy", "biopython", "tqdm", "pandas", "scipy>=1.7.0", "statsmodels",
    "matplotlib","seaborn","joblib","dill"]

setup(
    name = 'MATTE',
    version = '1.2.2',
    packages = find_packages(),
    install_requires = requires_packages,
    url = 'https://github.com/gx-Cai/MATTE',
    author = 'Cai Guoxin',
    author_email = 'gxcai@zju.edu.cn',
    include_package_data=True
)
