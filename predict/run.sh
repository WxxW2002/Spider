#!/bin/bash
echo "Running Linear Regression"
python ./run.py --model linear
echo "Running Decision Tree"
python ./run.py --model decision_tree
echo "Running Random Forest"
python ./run.py --model random_forest
echo "Running AdaBoost"
python ./run.py --model adaboost
echo "Running Bagging"
python ./run.py --model bagging
echo "Running Gradient Boost"
python ./run.py --model gradient_boost
echo "Running SVR"
python ./run.py --model svr
echo "Running Bayesian Ridge"
python ./run.py --model bayesian_ridge
echo "Running Elastic Net"
python ./run.py --model elastic_net
echo "Running Multi-layer Perceptron"
python ./run.py --model mlp --hidden_layer_sizes "(100, 50)" --max_iter 500
