# project/main.py
import os
import numpy as np
import pandas as pd  # For results summary table

# Import from your project's modules
from utils.data_processing import (
    load_data, load_ground_truth_edges,
    convert_edges_to_adj_matrix, get_dataset_name
)
from utils.evaluation import evaluate_dag_metrics
from visualization.graph_visualization import (
    visualize_true_graph,
    visualize_learned_graph_nx,
    visualize_comparison_graphs,
    ensure_visualization_folder  # Utility from graph_visualization
)
from visualization.plotting_utils import get_shared_layout
from algorithms.algorithm_factory import get_algorithm  # Using the factory function

# --- Configuration for Datasets ---
DATASETS_CONFIG = [
    {
        "name_id": "asia_n2000",
        "data_path": "datasets/asia_N2000.csv",
        "gt_path": "datasets/asia_ground_truth_edges.txt",
        "data_type": "discrete",
        "algo_params_override": {  # General override, can contain params for multiple algos
            "ges": {"bdeu_sample_prior": 10.0, "bdeu_structure_prior": 1.0},
            "pc": {"alpha": 0.05, "indep_test": "chisq"}  # Example PC override
        }
    },
    {
        "name_id": "asia_n5000",
        "data_path": "datasets/asia_N5000.csv",
        "gt_path": "datasets/asia_ground_truth_edges.txt",
        "data_type": "discrete",
        "algo_params_override": {  # Use defaults for PC for discrete if not specified
            "pc": {"alpha": 0.01}  # Or specify alpha only
        }
    },
    {
        "name_id": "sachs_continuous",
        "data_path": "datasets/sachs_observational_continuous.csv",
        "gt_path": "datasets/sachs_ground_truth_edges_17.txt",
        "data_type": "continuous",
        "algo_params_override": {
            "ges": {"penalty_discount": 2.0},
            "pc": {"alpha": 0.05, "indep_test": "fisherz"}  # Default for continuous, but can be explicit
        }
    }
]

# Centralized output folder for all visualizations
MAIN_VIZ_OUTPUT_FOLDER = "visualizations_output_project"


def run_single_experiment(dataset_config, algorithm_name="ges"):
    """
    Runs a causal discovery algorithm on a single dataset configuration,
    evaluates the results, and generates visualizations.
    """
    dataset_id_str = dataset_config['name_id']
    print(f"\n--- Processing Dataset: {dataset_id_str} with Algorithm: {algorithm_name.upper()} ---")

    ensure_visualization_folder(MAIN_VIZ_OUTPUT_FOLDER)
    dataset_viz_subfolder_name = f"{dataset_id_str}_{algorithm_name.lower()}"  # Use lower for folder name
    current_run_viz_folder = os.path.join(MAIN_VIZ_OUTPUT_FOLDER, dataset_viz_subfolder_name)
    ensure_visualization_folder(current_run_viz_folder)

    try:
        data_np, variable_names, _ = load_data(dataset_config["data_path"])
        print(f"  Data loaded: {data_np.shape[0]} samples, {data_np.shape[1]} variables.")
    except FileNotFoundError as e:
        print(f"  Error: Data file not found. {e}. Skipping this experiment.")
        return None
    except Exception as e:
        print(f"  Error loading data for {dataset_id_str}: {e}. Skipping.")
        return None

    true_adj_matrix_dag = None
    try:
        true_edges_list = load_ground_truth_edges(dataset_config["gt_path"])
        true_adj_matrix_dag = convert_edges_to_adj_matrix(true_edges_list, variable_names)
        num_true_edges = int(np.sum(true_adj_matrix_dag))
        print(f"  Ground truth DAG loaded: {num_true_edges} edges.")
    except FileNotFoundError as e:
        print(f"  Warning: Ground truth file not found. {e}. Ground truth dependent metrics will be affected.")
    except Exception as e:
        print(f"  Error loading/processing ground truth for {dataset_id_str}: {e}")

    learned_cpdag_adj_raw = None
    learned_strict_dag_adj = None
    algorithm_instance = None
    execution_time = None

    try:
        # Get general algorithm parameters from dataset_config if they exist
        # The factory will sort out which ones are for which algorithm.
        algo_specific_params_from_config = dataset_config.get("algo_params_override", {}).get(algorithm_name.lower(),
                                                                                              {})

        # score_name_override is mainly for GES. PC uses indep_test.
        # The factory logic needs to handle this.
        score_name_for_ges = algo_specific_params_from_config.get("score_func", None)  # If GES params specify a score

        algorithm_instance = get_algorithm(
            algorithm_name=algorithm_name,
            data_type=dataset_config["data_type"],
            score_name_override=score_name_for_ges if algorithm_name.lower() == 'ges' else None,
            parameters_override=algo_specific_params_from_config  # Pass the dict for this specific algo
        )

        algorithm_instance.learn_structure(data_np, variable_names=variable_names)

        learned_cpdag_adj_raw = algorithm_instance.get_learned_cpdag_adj_matrix()
        learned_strict_dag_adj = algorithm_instance.get_learned_strict_dag_adj_matrix()
        execution_time = algorithm_instance.get_execution_time()

    except Exception as e:
        print(f"  Error running {algorithm_name.upper()} on {dataset_id_str}: {e}")
        import traceback
        traceback.print_exc()
        return None

    if learned_strict_dag_adj is not None:
        num_learned_strict_edges = int(np.sum(learned_strict_dag_adj))
        print(f"  Learned graph (strict DAG part): {num_learned_strict_edges} strictly directed edges.")

    if learned_cpdag_adj_raw is not None:
        num_learned_undirected_edges = 0
        for i in range(learned_cpdag_adj_raw.shape[0]):
            for j in range(i + 1, learned_cpdag_adj_raw.shape[0]):
                if learned_cpdag_adj_raw[i, j] == -1 and learned_cpdag_adj_raw[j, i] == -1:
                    num_learned_undirected_edges += 1
        print(f"  Learned graph (CPDAG part): {num_learned_undirected_edges} undirected edges (i--j).")

    metrics = {"execution_time_sec": execution_time if execution_time is not None else -1.0}
    if true_adj_matrix_dag is not None and learned_strict_dag_adj is not None and learned_cpdag_adj_raw is not None:
        print("\n  Evaluating Learned Graph against Ground Truth...")
        eval_metrics = evaluate_dag_metrics(
            learned_strict_dag_adj,
            learned_cpdag_adj_raw,
            true_adj_matrix_dag
        )
        metrics.update(eval_metrics)
    else:
        print("\n  Skipping evaluation against ground truth (ground truth or learned graph not fully available).")

    viz_file_basename = f"{get_dataset_name(dataset_config['data_path'])}"
    layout_matrices = []
    if true_adj_matrix_dag is not None:
        layout_matrices.append(true_adj_matrix_dag)
    if learned_cpdag_adj_raw is not None:
        layout_matrices.append(learned_cpdag_adj_raw)

    shared_pos = {}
    if layout_matrices and variable_names:
        shared_pos = get_shared_layout(layout_matrices, variable_names)
    else:
        print("  Skipping layout generation as no valid graphs or variable names are available.")

    if true_adj_matrix_dag is not None:
        true_dag_filename = f"1_true_dag_{viz_file_basename}.png"
        visualize_true_graph(true_adj_matrix_dag, variable_names, true_dag_filename,
                             pos=shared_pos, dataset_name_str=viz_file_basename,
                             viz_folder=current_run_viz_folder)

    if learned_cpdag_adj_raw is not None:
        learned_cpdag_filename = f"2_{algorithm_name.lower()}_learned_cpdag_{viz_file_basename}.png"
        visualize_learned_graph_nx(learned_cpdag_adj_raw, variable_names, learned_cpdag_filename,
                                   pos=shared_pos, dataset_name_str=f"{viz_file_basename} ({algorithm_name.upper()})",
                                   viz_folder=current_run_viz_folder)

    if learned_cpdag_adj_raw is not None and true_adj_matrix_dag is not None:
        comparison_filename = f"3_comparison_{algorithm_name.lower()}_vs_true_{viz_file_basename}.png"
        visualize_comparison_graphs(learned_cpdag_adj_raw, true_adj_matrix_dag, variable_names, comparison_filename,
                                    pos=shared_pos, dataset_name_str=f"{viz_file_basename} ({algorithm_name.upper()})",
                                    viz_folder=current_run_viz_folder)

    print(f"  Visualizations saved to: {current_run_viz_folder}")
    return metrics


if __name__ == "__main__":
    all_run_results = {}

    print("--- Starting Causal Discovery Pipeline ---")

    for d_config in DATASETS_CONFIG:
        if not os.path.exists(d_config["data_path"]):
            print(
                f"Critical Error: Data file {d_config['data_path']} for dataset '{d_config['name_id']}' not found. Skipping.")
            continue
        if "gt_path" not in d_config or not os.path.exists(d_config["gt_path"]):  # Make GT path check more robust
            print(
                f"Warning: Ground truth file path missing or file {d_config.get('gt_path')} for dataset '{d_config['name_id']}' not found. Evaluation will be limited.")

        # --- Specify which algorithms to run on this dataset ---
        algorithms_to_run = ["ges", "pc"]  # Now running both GES and PC

        for algo_name in algorithms_to_run:
            experiment_key = f"{d_config['name_id']}_{algo_name.lower()}"
            result_metrics = run_single_experiment(d_config, algorithm_name=algo_name)
            if result_metrics:
                all_run_results[experiment_key] = result_metrics
            else:
                print(f"Experiment {experiment_key} did not complete successfully or returned no metrics.")

    print("\n\n--- Overall Experiment Results Summary ---")
    if not all_run_results:
        print("No experiments were successfully run or no metrics were collected.")
    else:
        results_summary_df = pd.DataFrame.from_dict(all_run_results, orient='index')

        preferred_column_order = [
            'execution_time_sec', 'shd_custom_cpdag_vs_dag',
            'f1_strict_directed', 'precision_strict_directed', 'recall_strict_directed',
            'tp_strict_directed', 'fp_strict_directed', 'fn_strict_directed',
            'f1_skeleton', 'precision_skeleton', 'recall_skeleton',
            'tp_skeleton', 'fp_skeleton', 'fn_skeleton'
        ]
        display_columns = [col for col in preferred_column_order if col in results_summary_df.columns]

        if not results_summary_df.empty and display_columns:
            print(results_summary_df[display_columns].to_string(float_format="%.4f"))

            summary_csv_path = os.path.join(MAIN_VIZ_OUTPUT_FOLDER, "experiment_results_summary.csv")
            try:
                # Make sure directory for summary CSV exists
                os.makedirs(os.path.dirname(summary_csv_path), exist_ok=True)
                results_summary_df[display_columns].to_csv(summary_csv_path, float_format="%.4f")
                print(f"\nResults summary also saved to: {summary_csv_path}")
            except Exception as e:
                print(f"\nCould not save results summary CSV: {e}")
        else:
            print("Results DataFrame is empty or no displayable columns were found.")

    print("\n--- Causal Discovery Pipeline Finished ---")