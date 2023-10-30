# ex2 Probability skews  
# python HT-FL.py --data "cifar" --model "resnet" --hubs 10 --workers 10 --tau 4 --q 2 --graph 5 --epochs 40 --batch 64 --fed True
# python HT-FL.py --data "cifar" --model "resnet" --hubs 10 --workers 10 --tau 4 --q 2 --graph 5 --epochs 40 --batch 64 --prob 1 --fed True
# python HT-FL.py --data "cifar" --model "resnet" --hubs 10 --workers 10 --tau 4 --q 2 --graph 5 --epochs 40 --batch 64 --prob 2 --fed True
# python HT-FL.py --data "cifar" --model "resnet" --hubs 10 --workers 10 --tau 4 --q 2 --graph 5 --epochs 40 --batch 64 --prob 3 --fed True
# python HT-FL.py --data "cifar" --model "resnet" --hubs 10 --workers 10 --tau 4 --q 2 --graph 5 --epochs 40 --batch 64 --prob 4 --fed True

# python HT-FL.py --data "cifar" --model "resnet" --hubs 10 --workers 10 --tau 4 --q 2 --graph 5 --epochs 40 --batch 64 --fed False
# python HT-FL.py --data "cifar" --model "resnet" --hubs 10 --workers 10 --tau 4 --q 2 --graph 5 --epochs 40 --batch 64 --prob 1 --fed False
# python HT-FL.py --data "cifar" --model "resnet" --hubs 10 --workers 10 --tau 4 --q 2 --graph 5 --epochs 40 --batch 64 --prob 2 --fed False
# python HT-FL.py --data "cifar" --model "resnet" --hubs 10 --workers 10 --tau 4 --q 2 --graph 5 --epochs 40 --batch 64 --prob 3 --fed False
# python HT-FL.py --data "cifar" --model "resnet" --hubs 10 --workers 10 --tau 4 --q 2 --graph 5 --epochs 40 --batch 64 --prob 4 --fed False
# python HT-FL.py --data "cifar" --model "resnet" --hubs 10 --workers 10 --tau 4 --q 2 --graph 5 --epochs 40 --batch 64 --prob 5 --fed False

# ex6 考虑能耗0.75下的hubs worker
# python HT-FL.py --data "cifar" --model "resnet" --hubs 20 --workers 5 --tau 4 --q 2 --graph 5 --epochs 40 --batch 64 --fed False --percentage 0.75 # MLL
# python HT-FL.py --data "cifar" --model "resnet" --hubs 10 --workers 10 --tau 4 --q 2 --graph 5 --epochs 40 --batch 64 --fed False --percentage 0.75 # MLL
# python HT-FL.py --data "cifar" --model "resnet" --hubs 5 --workers 20 --tau 4 --q 2 --graph 5 --epochs 40 --batch 64 --fed False --percentage 0.75 # MLL

# ex8 对比MLL和Local SGD, Distributed SGD
# python HT-FL.py --data "cifar" --model "resnet" --hubs 10 --workers 10 --tau 1 --q 8 --graph 5 --epochs 40 --batch 64 --fed False --percentage 0.75 # distributed
# python HT-FL.py --data "cifar" --model "resnet" --hubs 10 --workers 10 --tau 8 --q 1 --graph 5 --epochs 40 --batch 64 --fed False --percentage 0.75 # local
# python HT-FL.py --data "cifar" --model "resnet" --hubs 10 --workers 10 --tau 2 --q 4 --graph 5 --epochs 40 --batch 64 --fed False --percentage 0.75 # MLL
# python HT-FL.py --data "cifar" --model "resnet" --hubs 10 --workers 10 --tau 4 --q 2 --graph 5 --epochs 40 --batch 64 --fed False --percentage 0.75 # MLL

# ex9 noniid num_class
# python HT-FL.py --data "cifar" --model "resnet" --hubs 10 --workers 10 --tau 4 --q 2 --graph 5 --epochs 40 --batch 64 --fed False --percentage 0.75 --non_iid 1 --num_class 1 # MLL 准确率太低不要了
# python HT-FL.py --data "cifar" --model "resnet" --hubs 10 --workers 10 --tau 4 --q 2 --graph 5 --epochs 40 --batch 64 --fed False --percentage 0.75 --non_iid 1 --num_class 2 # MLL
# python HT-FL.py --data "cifar" --model "resnet" --hubs 10 --workers 10 --tau 4 --q 2 --graph 5 --epochs 40 --batch 64 --fed False --percentage 0.75 --non_iid 1 --num_class 4 # MLL

#ex10 Mixup uniform
# python HT-FL.py --data "cifar" --model "resnet" --hubs 10 --workers 10 --tau 4 --q 2 --graph 5 --epochs 40 --batch 64 --fed False --percentage 0.75 --non_iid 2 --uniform 0.1 # MLL 
# python HT-FL.py --data "cifar" --model "resnet" --hubs 10 --workers 10 --tau 4 --q 2 --graph 5 --epochs 40 --batch 64 --fed False --percentage 0.75 --non_iid 2 --uniform 0.2 # MLL
# python HT-FL.py --data "cifar" --model "resnet" --hubs 10 --workers 10 --tau 4 --q 2 --graph 5 --epochs 40 --batch 64 --fed False --percentage 0.75 --non_iid 2 --uniform 0.4 # MLL

#ex11 Dirichlet dir
# python HT-FL.py --data "cifar" --model "resnet" --hubs 10 --workers 10 --tau 4 --q 2 --graph 5 --epochs 40 --batch 64 --fed False --percentage 0.75 --non_iid 3 --dir 0.1 # MLL 
# python HT-FL.py --data "cifar" --model "resnet" --hubs 10 --workers 10 --tau 4 --q 2 --graph 5 --epochs 40 --batch 64 --fed False --percentage 0.75 --non_iid 3 --dir 1 # MLL
# python HT-FL.py --data "cifar" --model "resnet" --hubs 10 --workers 10 --tau 4 --q 2 --graph 5 --epochs 40 --batch 64 --fed False --percentage 0.75 --non_iid 3 --dir 10 # MLL

#ex12 Graph
# python HT-FL.py --data "cifar" --model "resnet" --hubs 10 --workers 10 --tau 4 --q 2 --graph 0 --epochs 40 --batch 64 --fed False --percentage 0.75 # MLL
# python HT-FL.py --data "cifar" --model "resnet" --hubs 10 --workers 10 --tau 4 --q 2 --graph 1 --epochs 40 --batch 64 --fed False --percentage 0.75 # MLL
# python HT-FL.py --data "cifar" --model "resnet" --hubs 10 --workers 10 --tau 4 --q 2 --graph 2 --epochs 40 --batch 64 --fed False --percentage 0.75 # MLL
# python HT-FL.py --data "cifar" --model "resnet" --hubs 10 --workers 10 --tau 4 --q 2 --graph 3 --epochs 40 --batch 64 --fed False --percentage 0.75 # MLL
# python HT-FL.py --data "cifar" --model "resnet" --hubs 10 --workers 10 --tau 4 --q 2 --graph 4 --epochs 40 --batch 64 --fed False --percentage 0.75 # MLL
# python HT-FL.py --data "cifar" --model "resnet" --hubs 10 --workers 10 --tau 4 --q 2 --graph 5 --epochs 40 --batch 64 --fed False --percentage 0.75 # MLL

# ex13 mnist
# python HT-FL.py --data "mnist" --model "cnn" --hubs 10 --workers 10 --tau 4 --q 2 --graph 5 --epochs 40 --batch 64 --fed False --percentage 0.75
# python HT-FL.py --data "tiny-imagenet" --model "resnet34" --hubs 10 --workers 10 --tau 4 --q 2 --graph 5 --epochs 40 --batch 64 --fed False --percentage 0.75

# ex14 大KL
# python HT-FL.py --data "cifar" --model "resnet" --hubs 10 --workers 10 --tau 4 --q 2 --graph 5 --epochs 40 --batch 64 --fed False --percentage 0.75 --non_iid 2 --uniform 0.08
# python HT-FL.py --data "cifar" --model "resnet" --hubs 10 --workers 10 --tau 4 --q 2 --graph 5 --epochs 40 --batch 64 --fed False --percentage 0.75 --non_iid 2 --uniform 0.06
# python HT-FL.py --data "cifar" --model "resnet" --hubs 10 --workers 10 --tau 4 --q 2 --graph 5 --epochs 40 --batch 64 --fed False --percentage 0.75 --non_iid 2 --uniform 0.04
# python HT-FL.py --data "cifar" --model "resnet" --hubs 10 --workers 10 --tau 4 --q 2 --graph 5 --epochs 40 --batch 64 --fed False --percentage 0.75 --non_iid 2 --uniform 0.02
