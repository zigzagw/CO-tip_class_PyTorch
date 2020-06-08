# Assessing quality of CO functionalized tips

 The machine learning models are implemented in PyTorch. The code is currently written in Python 3. At least the following Python packages are required:
* numpy
* matplotlib
* pytorch
* jupyter

Additionally, you need to have Cuda and cuDNN correctly configured on your system in order to train the models on an Nvidia GPU.

If you are using Anaconda, you can create the required Python environment with
```sh
conda env create -f environment.yml
```
This will create a conda enviroment named tf-gpu with the all the required packages. It also has a suitable version of the Cuda toolkit and cuDNN already installed. Activate the environment with
```sh
conda activate pytorch-gpu
```

To create the datasets and train the models, run `jupyter notebook` in the repository folder, open the `train_CO_tips.ipynb` notebook, and follow the instructions therein.

The folder `pretrained_weights` holds the weights for pretrained model.

