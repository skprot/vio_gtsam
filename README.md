# Visual GTSAM implementation

## Launch with [Google Colab](https://colab.research.google.com/)
In the cell use next command:
```!pip install git+https://<token_name>:<token_password>@gitlab.com/isr-lab-mobile-group/simple-gtsam.git@develop```

An example of launching is in the file [`initial_launching.ipynb`](https://gitlab.com/isr-lab-mobile-group/simple-gtsam/-/blob/develop/notebooks/initial_launching.ipynb)

## Installation issue on the local PC
There is an issue with installing on the local computer. Basic command ```pip install gtsam``` is not working for me without ```setuptools``` and ```wheel```. But it is not solving my final problem: interactive viewing of all modules with ```PyCharm```.
For successfully installation you need 
- to have a last pip (```pip install --upgrade pip```)
- to use ```python3.7``` (didn't try for other)
- to build lib on the local PC with ```cmake GUI```