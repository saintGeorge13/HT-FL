import pickle
filename = f"accuracy_modelresnet34_datatiny-imagenet_hubs10_workers10_tau4_q2_graph5_prob0_per0.75_noniid0_num_class2_uniform0.1_dir0.3.p"
costs1 = pickle.load(open(filename,'rb'))
print(costs1)