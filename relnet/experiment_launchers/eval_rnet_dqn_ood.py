import sys
from pathlib import Path

sys.path.append('/relnet')

from relnet.agent.pytorch_agent import PyTorchAgent
from relnet.agent.rnet_dqn.rnet_dqn_agent import RNetDQNAgent
from relnet.environment.graph_edge_env import GraphEdgeEnv
from relnet.evaluation.file_paths import FilePaths
from relnet.objective_functions.objective_functions import CriticalFractionTargeted, CriticalFractionRandom
from relnet.state.network_generators import NetworkGenerator, BANetworkGenerator, GNMNetworkGenerator

import argparse
import pprint as pp


def get_run_params():
    parser = argparse.ArgumentParser(
        description="Parameters for Evaluating NEP-DQN"
    )

    parser.add_argument('--graph_model', type=str, default='BA_n_20_m_2', help="The networks to train the model.")
    parser.add_argument('--edge_budget_percentage', type=float, default=1.0, help="The unit is percentage.")
    parser.add_argument('--ood_graph_model', type=str, default='BA_n_100_m_2', help="The OOD networks to evaluate.")
    parser.add_argument('--ood_edge_budget_percentage', type=float, default=1.0, help="The unit is percentage.")
    parser.add_argument('--num_test_graphs', type=int, default=128)
    parser.add_argument('--method', type=str, default='targeted_removal', help="Attack method: random_removal, or "
                                                                               "targeted_removal")
    params = parser.parse_args()

    params.model_identifier_prefix = f"rnet_dqn-{params.graph_model}-{params.method}-{params.edge_budget_percentage}"

    return params


def get_gen_params():
    gp = {}
    gp['n'] = 20
    gp['m_ba'] = 2
    gp['m_percentage_er'] = 20
    gp['m'] = NetworkGenerator.compute_number_edges(gp['n'], gp['m_percentage_er'])
    return gp


def get_options(file_paths):
    options = {"log_progress": True,
               "log_filename": str(file_paths.construct_log_filepath()),
               "log_tf_summaries": True,
               "random_seed": 42,
               "models_path": file_paths.models_dir,
               "restore_model": False}
    return options


def get_file_paths():
    parent_dir = '/experiment_data'
    experiment_id = 'development'
    file_paths = FilePaths(parent_dir, experiment_id)
    return file_paths


def get_training_steps(edge_budget_percentage):
    return int(4 * edge_budget_percentage * (10 ** 4))


if __name__ == '__main__':
    # Read Parameters
    params = get_run_params()
    pp.pprint(vars(params))

    num_test_graphs = params.num_test_graphs

    gen_params = get_gen_params()
    # Modify gen_params
    # BA_n_20_m_2 : [n, 20, m, 2]
    # GNM_n_20_m_38 : [n, 20, m, 38]
    graph_model, *graph_model_params = params.graph_model.split('_')
    gen_params['n'] = int(graph_model_params[1])
    assert graph_model == 'BA' or graph_model == 'GNM', 'Unrealized Network Model'
    if graph_model == 'BA':
        gen_params['m_ba'] = int(graph_model_params[3])
        Generator = BANetworkGenerator
    elif graph_model == 'GNM':
        gen_params['m'] = int(graph_model_params[3])
        Generator = GNMNetworkGenerator

    file_paths = get_file_paths()

    options = get_options(file_paths)
    # Modify options
    options["restore_model"] = True
    options['model_identifier_prefix'] = params.model_identifier_prefix

    storage_root = Path('/experiment_data/stored_graphs')
    original_dataset_dir = Path('/experiment_data/real_world_graphs/processed_data')
    kwargs = {'store_graphs': True, 'graph_storage_root': storage_root}

    # gen = BANetworkGenerator(**kwargs)
    gen = Generator(**kwargs)

    train_graph_seeds, validation_graph_seeds, test_graph_seeds = NetworkGenerator.construct_network_seeds(
        0, 0, num_test_graphs)

    train_graphs = gen.generate_many(gen_params, train_graph_seeds)
    validation_graphs = gen.generate_many(gen_params, validation_graph_seeds)
    test_graphs = gen.generate_many(gen_params, test_graph_seeds)

    # edge_percentage = 2.5
    edge_percentage = params.edge_budget_percentage

    obj_fun_kwargs = {"random_seed": 42, "num_mc_sims": gen_params['n'] * 2}

    # Set the Env
    assert params.method == 'targeted_removal' or params.method == 'random_removal', 'Unrealized Attack Method'
    if params.method == 'targeted_removal':
        targ_env = GraphEdgeEnv(CriticalFractionTargeted(), obj_fun_kwargs, edge_percentage)
    elif params.method == 'random_removal':
        targ_env = GraphEdgeEnv(CriticalFractionRandom(), obj_fun_kwargs, edge_percentage)

    agent = RNetDQNAgent(targ_env)
    agent.setup(options, agent.get_default_hyperparameters())

    avg_perf = agent.eval(test_graphs)
    print('Average Robustness Improvement: {0:.3f}'.format(avg_perf))
