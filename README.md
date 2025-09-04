# VI-HMC
This repository implements the hybrid **VI-HMC** approach for posterior inference in Bayesian neural networks and Bayesian neural operators as described in this [paper](https://arxiv.org/abs/2507.14652). 

--------------------------------------------------------------------------------
###  File structure
The folders are categorised into neural networks and neural operators each having four subfolders as follows:
```bash 
.
└── Neural_network/Operator_network
  ├── Data: Directory containing training and validation data
  ├── HMC: Directory containing scripts to run HMC
  ├── VI: Directory containing scripts to train a Bayesian network using VI
  └── VI_HMC: Directory containing scripts to implement the VI-HMC approach
```
---------------------------------------------------------------------------------
### Setup
This repository requires **Python 3.11.8**. A requirements.txt is provided with all the requried additional packages and thier compatible versions. 

---------------------------------------------------------------------------------
### Running examples
* The scripts that start with the name `main_*.py` are executable to run the corresponding inference method. 
* Files named `config.py` are input files that contains the configuration parameters required to run the main scripts.
* VI directories contain additional scripts named `sensitivity.py` to perform the sensitivity analysis.
* Deeponet data should be downladed from [this link](https://osf.io/x64h7) and saved to the 'Data' folder before running operator network scripts.

---------------------------------------------------------------------------------
### Reference
Please cite the following article when using this repository:
```
@article{thiagarajan2025accelerating,
  title={Accelerating Hamiltonian Monte Carlo for Bayesian Inference in Neural Networks and Neural Operators},
  author={Thiagarajan, Ponkrshnan and Zaki, Tamer A and Shields, Michael D},
  journal={arXiv preprint arXiv:2507.14652},
  year={2025}
}
```
