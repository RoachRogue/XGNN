# X2-GNN
X2-GNN is an attention based graph neural network architecture designed for utilizing the symmetrized one electron integral molecule feature. It's highlighted by the generalization ability.
## Requirements
the following python packages are needed to use X2-GNN, and the conda environment for this project is provided in 'env.yaml':\
numpy==1.23.5\
pyscf==2.2.1\
sympy==1.11.1\
torch==1.12.1\
torch-geometric==2.1.0\
torch-scatter==2.1.0\
scipy==1.10.1\
ase==3.22.1
## How to use
the following content will help you to reproduce the results presented in X2-GNN paper.
### download dataset
 All datasets used in this research is provided in Figshare: https://figshare.com/articles/dataset/X2-GNN_data/25848238. Download and copy them into "./raw/" and run infer scripts. these datasets are either provided in .xyz or .extxyz(supported by ASE) format.
### represent molecules as graph
 The graph representation of molecules and calculation of integrals are handled by 'qm9_allprop.py'. It stores the obtained results in torch_geometric 'Data' objects and forms an 'InMemoryDataset' object. The processed dataset will be stored at './processed/'. To use custum dataset, just specify the corresponding input xyz/extxyz file.
### train an energy model
 After the dataset is prepared, a training script 'train_ema.py' is provided. hyperparameters are specified by a json file 'config.json'. the trainer will automatically generate log file in directory './log/'. Model checkpoint will be saved in directory './modelsaves/'
### load trained model
 A notebook 'load.ipynb' is provided to use model trained on QM9(U0). In the notebook we test a trained model on AID/OCELOT dataset. Other properties and dataset can be used by adjusting the config file. Another notebook "HS_test.ipynb" in "./scripts/" is provided to reproduce the results of comparison between model including both h_core and overlap and solely overlap.
