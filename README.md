# HT-FL: Hybrid Training Federated Learning for Heterogeneous Edge-based IoT Networks
## Dependencies and Setup
All code runs on Python 3.10.10 using PyTorch version 2.0.0
In addition, you will need to install
- torch
- numpy
- torchvision
- pickle
- argparse
- torchsummary
- random
## Organization
- `HT-FL.py`: The main algorithm code.
- `test_iid.py`: Test the distribution of data.
- `modelsize.py`: Calculate the size of local model.
- 'run_exps2.sh': Code of our experiments.
- `graphs/`: Different type of hubs communication graph.
- `Net/`: Realization of different local models.
- `results/`: Results of our stimulation.

## Description of main parameters
This is only a stimulation code for "HT-FL: Energy Efficient Federated Learning for IoT Applications with Non-IID Data" and have many aspects to improve. Experiments we run are written in run_exps2.sh. Key params are given as follows:  
`--data`: dataset  
`--model`: training model  
`--hubs`: hub amount  
`--workers`: worker amount   
`--tau`: local training rounds  
`--q`: global aggregation frequency  
`--graph`: hub communication graph type  
`--epochs`: training epoches  
`--batch`: training batch size  
`--prob`: probability for worker to attend training   
`--fed & --non_iid`: data distribution paradigm  
`--percentage`: controlling param  
`--num_class`: amount of class on a worker  
`--uniform`: uniform data percentage  
`--dir`: dirichlet param 

