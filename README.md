# PV parameter identification using evolutionary algorithms

## Overview
This repository provides the implementation of PV parameter identification using evolutionary algorithms. The objective is to robustly estimate the Single Diode Model (SDM) parameters from an experimental IV curve of a PV module or a PV cell.

## Installation Guide

### Prerequisites
We recommend using **[Anaconda](https://store.continuum.io/cshop/anaconda/)** as it bundles most of the required dependencies. If you haven't installed Anaconda yet, please download and install it from the official website before proceeding.

### Steps to Install PV_EVOLVE

#### 1. Create a New Anaconda Environment
Open a terminal (Unix) or Anaconda Prompt (Windows) and execute:
```bash
conda create --name pv_evolve
```

#### 2. Activate the New Environment
```bash
conda activate pv_evolve
```

#### 3. Install Required Packages

- **Pymoo** (for optimization algorithms):
  ```bash
  conda install pymoo==0.6.1
  ```

- **Pandas** (for data handling):
  ```bash
  conda install pandas
  ```

- **IPykernel** (for Jupyter integration):
  ```bash
  conda install ipykernel
  ```

#### 4. Add Jupyter Kernel for the Environment
```bash
python -m ipykernel install --user --name pv_evolve --display-name "PV EVOLVE"
```

#### 5. Launch Jupyter Notebook
To start a Jupyter Notebook and use the created kernel:
```bash
jupyter notebook
```
Then, select the **"PV EVOLVE"** kernel from the available options.

#### 6. Deactivating the Environment
When done, deactivate the environment with:
```bash
conda deactivate
```

## Example Notebook
In the folder **notebooks**, there is an example notebook demonstrating the usage.