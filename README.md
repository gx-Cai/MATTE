# MATTE package User Guide

## Description

MATTE (Module Alignment of TranscripTomE) is a python package aiming to analyze transcriptome from samples with different phenotypes in a module view. Differential expression (DE) is commonly used in analyzing transcriptome data. However, genes do not work alone; they collaborate. In recent years, network and module-based differential methods have been developed to obtain more information. New problems appear to make sure module or network structure is preserved in all of the phenotypes. To that end, we proposed MATTE to find the conserved module and diverged module by treating genes from different phenotypes as individual ones. By doing so, meaningful markers and modules can be found to understand the difference between phenotypes better.


**Advantages**

In the first place, MATTE merges the data from phenotypes, seeing genes from different phenotypes as new analyzing unite. By doing so, benefits got as follows:

1. MATTE considers the information in phenotypes in the preprocessing stage, hoping to find a more exciting conclusion.
2. MATTE is making transcriptome analysis that includes the relationship between phenotypes, which is of significance in cancer or other complex phenotypes.
3. MATTE can deal with more noise thanks to calculating relative different expression (RDE) and ignoring the batch effect.
4. In a module view, "Markers" can be easily transferred to other cases but not over fits compared to a gene view.
5. The result of MATTE can be easily analyzed.  


## Install
1. Install from pip is recommended.
```
pip install MATTE
```

2. Install from source

download all files under this repository

```bash
unzip main.zip
cd main
python3 setup.py build
python3 setup.py install
```

## Guidance

Detailed User guide and api can be seen in [documention](https://mattedoc.readthedocs.io/en/latest/) and jupyter notebook file under this repository. 

## Files under this repository
```
- MATTE:Source Code of the package
    - __init__.py
    _ ...
- CaseStudy: Three case code in the paper
    - GSE100796
    - scRNA
    - Simulation
- QuickStart.ipynb: Quick start user guide
- UserGuide.ipynb: Detailed advance guidance
- setup.py
- .gitignore
- README.md
```