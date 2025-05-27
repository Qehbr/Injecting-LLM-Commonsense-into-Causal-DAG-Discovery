# project/main.py
import os
import numpy as np
import pandas as pd
import time
import torch
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from utils.data_processing import (
    load_data, load_ground_truth_edges,
    convert_edges_to_adj_matrix, get_dataset_name
)
from utils.evaluation import evaluate_dag_metrics
from visualization.graph_visualization import (
    ensure_visualization_folder
)
from algorithms.algorithm_factory import get_algorithm

# --- Standard Parameters for DAG-GNN ---
# Based on the original paper's implementation for reliable performance.
# We will use these fixed parameters instead of tuning.
DAG_GNN_SUGGESTED_PARAMS = {
    "epochs": 300,
    "lr": 3e-3,  # From original paper's code
    "batch_size": 100,
    "lambda_A": 0.0,  # The augmented Lagrangian handles the constraint
    "c_A": 1.0,  # Initial penalty coefficient
    "k_max_iter": 100,  # Paper's default
    "graph_threshold": 0.3,  # Recommended threshold [cite: 144]
    "h_tol": 1e-8,  # Convergence tolerance
    "tau_A": 0.0,  # L1 regularization on A, can be 0.
    "encoder_hidden": 64,
    "decoder_hidden": 64,
    "seed": 42
}

# --- Configuration for Datasets ---
DATASETS_CONFIG = [
    {
        "name_id": "asia_n2000",
        "data_path": "datasets/asia_N2000.csv",
        "gt_path": "datasets/asia_ground_truth_edges.txt",
        "data_type": "discrete",
        "algo_params_override": {
            "ges": {"bdeu_sample_prior": 10.0},
            "pc": {"alpha": 0.05, "indep_test": "gsq"},
            "dag-gnn": DAG_GNN_SUGGESTED_PARAMS
        }
    },
    {
        "name_id": "sachs_continuous",
        "data_path": "datasets/sachs_observational_continuous.csv",
        "gt_path": "datasets/sachs_ground_truth_edges_17.txt",
        "data_type": "continuous",
        "algo_params_override": {
            "ges": {"penalty_discount": 2.0},
            "pc": {"alpha": 0.05},
            "dag-gnn": {
                **DAG_GNN_SUGGESTED_PARAMS,
                "tau_A": 0.01  # Add small L1 regularization for application data as suggested by paper [cite: 145]
            }
        }
    }
]

MAIN_VIZ_OUTPUT_FOLDER = "visualizations_output_project"


def run_single_experiment(dataset_config, algorithm_name="ges", viz_enabled=True, verbose_eval=True):
    """
    Runs a single causal discovery experiment for a given dataset and algorithm.
    """
    dataset_id_str = dataset_config['name_id']
    if verbose_eval:
        print(f"\n--- Processing Dataset: {dataset_id_str} with Algorithm: {algorithm_name.upper()} ---")

    # Setup visualization folder
    current_run_viz_folder = None
    if viz_enabled:
        ensure_visualization_folder(MAIN_VIZ_OUTPUT_FOLDER)
        dataset_viz_subfolder_name = f"{dataset_id_str}_{algorithm_name.lower()}"
        current_run_viz_folder = os.path.join(MAIN_VIZ_OUTPUT_FOLDER, dataset_viz_subfolder_name)
        ensure_visualization_folder(current_run_viz_folder)

    # Load data and ground truth
    data_np_orig, variable_names, _ = load_data(dataset_config["data_path"])
    true_adj_matrix_dag = None
    try:
        true_edges_list = load_ground_truth_edges(dataset_config["gt_path"])
        true_adj_matrix_dag = convert_edges_to_adj_matrix(true_edges_list, variable_names)
    except Exception as e:
        if verbose_eval: print(f"  Warning/Error loading GT: {e}")

    # Prepare data shape (DAG-GNN expects 3D)
    if data_np_orig.ndim == 2:
        data_np_for_algo = np.expand_dims(data_np_orig, axis=2)
    else:
        data_np_for_algo = data_np_orig

    if verbose_eval: print(f"  Data loaded: {data_np_orig.shape[0]} samples, {data_np_orig.shape[1]} variables.")

    # --- Run Algorithm ---
    execution_time = -1.0
    try:
        algo_params = dataset_config.get("algo_params_override", {}).get(algorithm_name.lower(), {})

        algorithm_instance = get_algorithm(
            algorithm_name=algorithm_name,
            data_type=dataset_config["data_type"],
            parameters_override=algo_params
        )

        if verbose_eval: print(f"  Running {algorithm_name.upper()} with params: {algo_params}")

        algorithm_instance.learn_structure(data_np_for_algo, variable_names=variable_names)
        execution_time = algorithm_instance.get_execution_time()

        learned_cpdag_adj_raw = algorithm_instance.get_learned_cpdag_adj_matrix()
        learned_strict_dag_adj = algorithm_instance.get_learned_strict_dag_adj_matrix()

    except Exception as e:
        print(f"  FATAL ERROR running {algorithm_name.upper()} on {dataset_id_str}: {e}")
        import traceback
        traceback.print_exc()
        return None

    # --- Evaluate and Visualize ---
    metrics = {"execution_time_sec": execution_time}
    if true_adj_matrix_dag is not None and learned_strict_dag_adj is not None:
        if verbose_eval: print("\n  Evaluating Learned Graph against Ground Truth...")
        eval_metrics = evaluate_dag_metrics(
            learned_strict_dag_adj, learned_cpdag_adj_raw, true_adj_matrix_dag, verbose=verbose_eval
        )
        metrics.update(eval_metrics)

    if viz_enabled and current_run_viz_folder:
        from visualization.graph_visualization import visualize_true_graph, visualize_learned_graph_nx, \
            visualize_comparison_graphs
        from visualization.plotting_utils import get_shared_layout

        viz_file_basename = get_dataset_name(dataset_config['data_path'])
        shared_pos = get_shared_layout([true_adj_matrix_dag, learned_cpdag_adj_raw], variable_names)

        if true_adj_matrix_dag is not None:
            visualize_true_graph(true_adj_matrix_dag, variable_names, f"1_true_dag_{viz_file_basename}.png",
                                 pos=shared_pos, viz_folder=current_run_viz_folder)
        if learned_cpdag_adj_raw is not None:
            visualize_learned_graph_nx(learned_cpdag_adj_raw, variable_names, f"2_learned_{viz_file_basename}.png",
                                       pos=shared_pos,
                                       dataset_name_str=f"{viz_file_basename} ({algorithm_name.upper()})",
                                       viz_folder=current_run_viz_folder)
            if true_adj_matrix_dag is not None:
                visualize_comparison_graphs(learned_cpdag_adj_raw, true_adj_matrix_dag, variable_names,
                                            f"3_comparison_{viz_file_basename}.png", pos=shared_pos,
                                            dataset_name_str=f"{viz_file_basename} ({algorithm_name.upper()})",
                                            viz_folder=current_run_viz_folder)

        if verbose_eval: print(f"  Visualizations saved to: {current_run_viz_folder}")

    return metrics


# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Starting Causal Discovery Pipeline (No Tuning) ---")
    all_run_results = {}

    # Iterate through each dataset configuration
    for d_config in DATASETS_CONFIG:
        # Define which algorithms to run
        algorithms_to_run = ["ges", "pc", "dag-gnn"]
        for algo_name in algorithms_to_run:
            experiment_key = f"{d_config['name_id']}_{algo_name.lower()}"
            result_metrics = run_single_experiment(
                d_config,
                algorithm_name=algo_name,
                viz_enabled=True,
                verbose_eval=True
            )
            if result_metrics:
                all_run_results[experiment_key] = result_metrics

    # --- Print Final Summary Table ---
    print("\n\n--- Overall Experiment Results Summary ---")
    if not all_run_results:
        print("No experiments were successfully run or no metrics were collected.")
    else:
        results_summary_df = pd.DataFrame.from_dict(all_run_results, orient='index')
        preferred_column_order = [
            'execution_time_sec', 'shd_custom_cpdag_vs_dag', 'f1_skeleton',
            'precision_skeleton', 'recall_skeleton', 'f1_strict_directed',
            'precision_strict_directed', 'recall_strict_directed'
        ]
        display_columns = [col for col in preferred_column_order if col in results_summary_df.columns]

        if not results_summary_df.empty and display_columns:
            print(results_summary_df[display_columns].to_string(float_format="%.4f"))
            summary_csv_path = os.path.join(MAIN_VIZ_OUTPUT_FOLDER, "experiment_results_summary.csv")
            try:
                os.makedirs(os.path.dirname(summary_csv_path), exist_ok=True)
                results_summary_df[display_columns].to_csv(summary_csv_path, float_format="%.4f")
                print(f"\nResults summary also saved to: {summary_csv_path}")
            except Exception as e:
                print(f"\nCould not save results summary CSV: {e}")
        else:
            print("Results DataFrame is empty or no displayable columns.")

    print("\n--- Causal Discovery Pipeline Finished ---")