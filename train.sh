#! /bin/bash

source activate ucfadar-relnet


graph_model_array=("BA_n_20_m_2" "GNM_n_20_m_38")
# edge_budget_percentage_array=("1.0" "2.5" "5.0")
edge_budget_percentage_array=("1.0" "2.5")
num_training_steps=200
# num_train_graphs=16384
num_train_graphs=10
# num_validation_graphs=128
num_validation_graphs=10
# method_array=("targeted_removal" "random_removal")
method_array=("targeted_removal")


for graph_model in ${graph_model_array[@]}
do
  for edge_budget_percentage in ${edge_budget_percentage_array[@]}
  do
    for method in ${method_array[@]}
    do
      date=$(date +%Y%m%d%H%M%S)
      python relnet/experiment_launchers/run_rnet_dqn.py \
       --graph_model ${graph_model} \
       --edge_budget_percentage ${edge_budget_percentage} \
       --num_training_steps ${num_training_steps} \
       --num_train_graphs ${num_train_graphs} \
       --num_validation_graphs ${num_validation_graphs} \
       --method ${method} | tee ${date}.out
    done
  done
done





