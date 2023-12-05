# Path Signatures for Human Action Recognition using Esig
[jhmdb_demo.ipynb](jhmdb_demo.ipynb) is a [Jupyter](https://jupyter.org/) notebook which demonstrates the use of path signatures obtained using esig for human action recognition. The analysis is closely related to the work of [Yang et al. (2019)](https://arxiv.org/abs/1707.03993). The notebook is partially viewable directly from gitlab, however Jupyter or Binder are recommended for viewing the example videos included in the notebook.

## How to Run
The notebook attempts to install the packages listed in [requirements.txt](requirements.txt) automatically when executed. It provides the option to also install the dependency on [PyTorch](https://pytorch.org/), however __it is recommended to install PyTorch manually__ following the [official instructions](https://pytorch.org/get-started/locally/). This is because the correct version of PyTorch to install depends on your hardware, operating system and CUDA version. The notebook was developed and tested using PyTorch version 1.5.0, instructions for installing this version can be found [here](https://pytorch.org/get-started/previous-versions/#v150).

Alternatively, a typical process for installing the package dependencies involves creating a new Python virtual environment and then inside the environment executing

    pip install -r requirements.txt

Then install PyTorch manually following the [official instructions](https://pytorch.org/get-started/locally/), (see [here](https://pytorch.org/get-started/previous-versions/#v150) for PyTorch version 1.5.0).
Alternatively (not recommended) you can run

    pip install -r requirements_torch.txt
