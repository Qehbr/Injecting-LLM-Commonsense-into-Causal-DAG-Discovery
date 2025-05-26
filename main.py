# project/main.py
import os
import numpy as np
import pandas as pd
import optuna  # Import Optuna
import time

# Import from your project's modules
from utils.data_processing import (
    load_data, load_ground_truth_edges,
    convert_edges_to_adj_matrix, get_dataset_name
)
from utils.evaluation import evaluate_dag_metrics
from visualization.graph_visualization import (
    ensure_visualization_folder
)
from algorithms.algorithm_factory import get_algorithm

# --- Configuration for Datasets (remains the same) ---
DATASETS_CONFIG = [
    {
        "name_id": "asia_n2000",
        "data_path": "datasets/asia_N2000.csv",
        "gt_path": "datasets/asia_ground_truth_edges.txt",
        "data_type": "discrete",
        "algo_params_override": {  # Default/Initial params
            "ges": {"bdeu_sample_prior": 10.0},
            "pc": {"alpha": 0.05, "indep_test": "gsq"},
            "dag-gnn": {
                "epochs": 300, "lr": 1e-3, "batch_size": 128,
                "lambda_A": 0.1, "c_A": 1.0, "k_max_iter": 150,
                "graph_threshold": 0.2
            }
        }
    },
    {
        "name_id": "sachs_continuous",
        "data_path": "datasets/sachs_observational_continuous.csv",
        "gt_path": "datasets/sachs_ground_truth_edges_17.txt",
        "data_type": "continuous",
        "algo_params_override": {  # Default/Initial params
            "ges": {"penalty_discount": 2.0},
            "pc": {"alpha": 0.05},
            "dag-gnn": {
                "epochs": 300, "lr": 1e-3, "batch_size": 100,
                "lambda_A": 0.1, "c_A": 1.0, "k_max_iter": 150,
                "graph_threshold": 0.2
            }
        }
    }
]
MAIN_VIZ_OUTPUT_FOLDER = "visualizations_output_project"
OPTUNA_OUTPUT_BASE_FOLDER = "optuna_study_results"


# --- Optuna Objective Function (Optimizing for SHD) ---
def dagnn_objective(trial, dataset_config_entry, base_output_folder="optuna_runs_temp"):
    dataset_id_str = dataset_config_entry['name_id']
    algorithm_name = "dag-gnn"

    print(f"\n--- Optuna Trial #{trial.number} for DAG-GNN on {dataset_id_str} (Optimizing SHD) ---")

    suggested_params = {
        "epochs": trial.suggest_int("epochs", 200, 800, step=100),
        "lr": trial.suggest_float("lr", 5e-5, 5e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "lambda_A": trial.suggest_float("lambda_A", 1e-2, 5.0, log=True),
        "c_A": trial.suggest_float("c_A", 0.5, 20.0, log=True),
        "k_max_iter": trial.suggest_int("k_max_iter", 100, 300, step=50),
        "graph_threshold": trial.suggest_float("graph_threshold", 0.05, 0.35, step=0.025),
        "encoder_hidden": trial.suggest_categorical("encoder_hidden", [32, 64, 96]),
        "decoder_hidden": trial.suggest_categorical("decoder_hidden", [32, 64, 96]),
        "z_dims": trial.suggest_int("z_dims", 1, 5),
        "h_tol": trial.suggest_float("h_tol", 1e-9, 1e-6, log=True)
    }
    print(f"Suggested params: {suggested_params}")

    temp_dataset_config = dataset_config_entry.copy()
    current_algo_params_override = temp_dataset_config.get("algo_params_override", {}).copy()
    current_algo_params_override[algorithm_name.lower()] = suggested_params
    temp_dataset_config["algo_params_override"] = current_algo_params_override

    trial_output_folder = os.path.join(base_output_folder, dataset_id_str, f"trial_{trial.number}")
    ensure_visualization_folder(trial_output_folder)

    try:
        data_np, variable_names, _ = load_data(temp_dataset_config["data_path"])
    except Exception as e:
        print(f"  Error loading data for trial: {e}. Pruning.")
        raise optuna.exceptions.TrialPruned()

    true_adj_matrix_dag = None
    try:
        true_edges_list = load_ground_truth_edges(temp_dataset_config["gt_path"])
        true_adj_matrix_dag = convert_edges_to_adj_matrix(true_edges_list, variable_names)
    except Exception as e:
        print(f"  Error loading ground truth for trial: {e}. Pruning.")
        raise optuna.exceptions.TrialPruned()

    if true_adj_matrix_dag is None: return float('inf')  # Should be pruned before this

    try:
        algorithm_instance = get_algorithm(
            algorithm_name=algorithm_name,
            data_type=temp_dataset_config["data_type"],
            parameters_override=temp_dataset_config["algo_params_override"]
        )

        print(f"Starting DAG-GNN learn_structure for trial {trial.number}...")
        start_learn_time = time.time()
        algorithm_instance.learn_structure(data_np, variable_names=variable_names)
        learn_time = time.time() - start_learn_time
        print(f"Finished DAG-GNN learn_structure for trial {trial.number} in {learn_time:.2f}s.")

        learned_cpdag_adj_raw = algorithm_instance.get_learned_cpdag_adj_matrix()
        learned_strict_dag_adj = algorithm_instance.get_learned_strict_dag_adj_matrix()

    except Exception as e:
        print(f"  Trial {trial.number} failed during algorithm execution: {e}")
        import traceback
        traceback.print_exc()
        raise optuna.exceptions.TrialPruned()

    if learned_strict_dag_adj is not None and learned_cpdag_adj_raw is not None:
        print(f"Evaluating trial {trial.number}...")
        # Consider making evaluate_dag_metrics accept a verbose flag
        eval_metrics = evaluate_dag_metrics(
            learned_strict_dag_adj,
            learned_cpdag_adj_raw,
            true_adj_matrix_dag  # Pass verbose=False here if implemented
        )
        shd = eval_metrics.get("shd_custom_cpdag_vs_dag", float('inf'))  # Default to high value if not found
        f1_skeleton = eval_metrics.get("f1_skeleton", 0.0)
        print(
            f"Trial {trial.number} on {dataset_id_str} - SHD: {shd}, F1 Skeleton: {f1_skeleton:.4f}, Time: {learn_time:.2f}s")
        trial.set_user_attr("f1_skeleton", f1_skeleton)  # Store F1 as a user attribute
        return shd  # Minimize SHD
    else:
        print(f"Trial {trial.number} for {dataset_id_str} - Learned graph is None. Returning high SHD.")
        return float('inf')


# --- Original run_single_experiment (modified for optional verbosity/visualization) ---
def run_single_experiment(dataset_config, algorithm_name="ges", viz_enabled=True, verbose_eval=True):
    # ... (This function remains the same as in the previous response)
    dataset_id_str = dataset_config['name_id']
    if verbose_eval: print(f"\n--- Processing Dataset: {dataset_id_str} with Algorithm: {algorithm_name.upper()} ---")

    current_run_viz_folder = None
    if viz_enabled:
        ensure_visualization_folder(MAIN_VIZ_OUTPUT_FOLDER)
        dataset_viz_subfolder_name = f"{dataset_id_str}_{algorithm_name.lower()}"
        current_run_viz_folder = os.path.join(MAIN_VIZ_OUTPUT_FOLDER, dataset_viz_subfolder_name)
        ensure_visualization_folder(current_run_viz_folder)

    try:
        data_np, variable_names, _ = load_data(dataset_config["data_path"])
        if verbose_eval: print(f"  Data loaded: {data_np.shape[0]} samples, {data_np.shape[1]} variables.")
    except Exception as e:
        print(f"  Error loading data for {dataset_id_str}: {e}. Skipping.")
        return None

    true_adj_matrix_dag = None
    try:
        true_edges_list = load_ground_truth_edges(dataset_config["gt_path"])
        true_adj_matrix_dag = convert_edges_to_adj_matrix(true_edges_list, variable_names)
        if verbose_eval: print(f"  Ground truth DAG loaded: {int(np.sum(true_adj_matrix_dag))} edges.")
    except Exception as e:
        if verbose_eval: print(f"  Warning/Error loading GT for {dataset_id_str}: {e}")

    learned_cpdag_adj_raw = None
    learned_strict_dag_adj = None
    execution_time = -1.0

    try:
        algo_params_from_config = dataset_config.get("algo_params_override", {})
        current_algo_params = algo_params_from_config.get(algorithm_name.lower(), {})
        if not current_algo_params and algorithm_name.lower() in algo_params_from_config:
            current_algo_params = algo_params_from_config[algorithm_name.lower()]

        algorithm_instance = get_algorithm(
            algorithm_name=algorithm_name,
            data_type=dataset_config["data_type"],
            score_name_override=current_algo_params.get("score_func"),
            parameters_override=current_algo_params
        )

        if verbose_eval: print(f"  Running {algorithm_name.upper()} with params: {current_algo_params}")
        algorithm_instance.learn_structure(data_np, variable_names=variable_names)
        learned_cpdag_adj_raw = algorithm_instance.get_learned_cpdag_adj_matrix()
        learned_strict_dag_adj = algorithm_instance.get_learned_strict_dag_adj_matrix()
        execution_time = algorithm_instance.get_execution_time()
    except Exception as e:
        print(f"  Error running {algorithm_name.upper()} on {dataset_id_str}: {e}")
        import traceback
        traceback.print_exc()
        return None

    metrics = {"execution_time_sec": execution_time if execution_time is not None else -1.0}
    if true_adj_matrix_dag is not None and learned_strict_dag_adj is not None and learned_cpdag_adj_raw is not None:
        if verbose_eval: print("\n  Evaluating Learned Graph against Ground Truth...")

        # If evaluate_dag_metrics is modified to take a verbose flag:
        # eval_metrics = evaluate_dag_metrics(
        #     learned_strict_dag_adj, learned_cpdag_adj_raw, true_adj_matrix_dag, verbose=verbose_eval
        # )
        # For now, assuming evaluate_dag_metrics prints internally if verbose_eval is True for the outer function
        eval_metrics = evaluate_dag_metrics(
            learned_strict_dag_adj, learned_cpdag_adj_raw, true_adj_matrix_dag
        )
        metrics.update(eval_metrics)

    if viz_enabled and current_run_viz_folder:
        # Ensure visualization functions are imported
        from visualization.graph_visualization import visualize_true_graph, visualize_learned_graph_nx, \
            visualize_comparison_graphs
        from visualization.plotting_utils import get_shared_layout

        viz_file_basename = f"{get_dataset_name(dataset_config['data_path'])}"
        layout_matrices = []
        if true_adj_matrix_dag is not None: layout_matrices.append(true_adj_matrix_dag)
        if learned_cpdag_adj_raw is not None: layout_matrices.append(learned_cpdag_adj_raw)

        shared_pos = {}
        if layout_matrices and variable_names:
            shared_pos = get_shared_layout(layout_matrices, variable_names)

        if true_adj_matrix_dag is not None:
            true_dag_filename = f"1_true_dag_{viz_file_basename}.png"
            visualize_true_graph(true_adj_matrix_dag, variable_names, true_dag_filename,
                                 pos=shared_pos, dataset_name_str=viz_file_basename,
                                 viz_folder=current_run_viz_folder)
        if learned_cpdag_adj_raw is not None:
            learned_cpdag_filename = f"2_{algorithm_name.lower()}_learned_cpdag_{viz_file_basename}.png"
            visualize_learned_graph_nx(learned_cpdag_adj_raw, variable_names, learned_cpdag_filename,
                                       pos=shared_pos,
                                       dataset_name_str=f"{viz_file_basename} ({algorithm_name.upper()})",
                                       viz_folder=current_run_viz_folder)
        if learned_cpdag_adj_raw is not None and true_adj_matrix_dag is not None:
            comparison_filename = f"3_comparison_{algorithm_name.lower()}_vs_true_{viz_file_basename}.png"
            visualize_comparison_graphs(learned_cpdag_adj_raw, true_adj_matrix_dag, variable_names, comparison_filename,
                                        pos=shared_pos,
                                        dataset_name_str=f"{viz_file_basename} ({algorithm_name.upper()})",
                                        viz_folder=current_run_viz_folder)
        if verbose_eval: print(f"  Visualizations saved to: {current_run_viz_folder}")

    return metrics


# --- Main Execution Block ---
if __name__ == "__main__":
    PERFORM_OPTUNA_TUNING = True
    ALGORITHM_TO_TUNE = "dag-gnn"
    DATASETS_FOR_DAGNN_TUNING = ["asia_n2000", "sachs_continuous"]
    N_OPTUNA_TRIALS_PER_DATASET = 10

    if PERFORM_OPTUNA_TUNING and ALGORITHM_TO_TUNE.lower() == "dag-gnn":
        ensure_visualization_folder(OPTUNA_OUTPUT_BASE_FOLDER)
        for dataset_name_to_tune in DATASETS_FOR_DAGNN_TUNING:
            dataset_config_for_tuning = next(
                (item for item in DATASETS_CONFIG if item["name_id"] == dataset_name_to_tune), None
            )
            if not dataset_config_for_tuning:
                print(f"Config for dataset '{dataset_name_to_tune}' not found. Skipping Optuna tuning for it.")
                continue

            print(f"\n--- Starting Optuna Tuning for DAG-GNN on {dataset_name_to_tune} (Optimizing SHD) ---")

            study_name = f"dagnn_shd_tuning_{dataset_name_to_tune}"
            # Example for SQLite storage (uncomment to use):
            # storage_path = os.path.join(OPTUNA_OUTPUT_BASE_FOLDER, f"{study_name}.db")
            # storage_name = f"sqlite:///{storage_path}"
            # study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True,
            #                             direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))

            study = optuna.create_study(direction="minimize",
                                        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))  # Minimize SHD
            study.optimize(
                lambda trial: dagnn_objective(trial, dataset_config_for_tuning, OPTUNA_OUTPUT_BASE_FOLDER),
                n_trials=N_OPTUNA_TRIALS_PER_DATASET,
            )

            print(f"\n--- Optuna Tuning Finished for DAG-GNN on {dataset_name_to_tune} ---")
            best_trial = study.best_trial
            print(
                f"  Best SHD: {best_trial.value:.4f} (F1 Skeleton: {best_trial.user_attrs.get('f1_skeleton', 'N/A')})")
            print("  Best Params: ")
            for key, value in best_trial.params.items():
                print(f"    {key}: {value}")

            for i, config in enumerate(DATASETS_CONFIG):
                if config["name_id"] == dataset_name_to_tune:
                    if "algo_params_override" not in DATASETS_CONFIG[i]:
                        DATASETS_CONFIG[i]["algo_params_override"] = {}
                    if ALGORITHM_TO_TUNE.lower() not in DATASETS_CONFIG[i]["algo_params_override"]:
                        DATASETS_CONFIG[i]["algo_params_override"][ALGORITHM_TO_TUNE.lower()] = {}
                    DATASETS_CONFIG[i]["algo_params_override"][ALGORITHM_TO_TUNE.lower()].update(best_trial.params)
                    print(f"Updated config for {dataset_name_to_tune} with best Optuna params for DAG-GNN.")
                    break
        print("\nCompleted all Optuna tuning sessions for DAG-GNN.")

    print("\n--- Starting Regular Experiment Runs ---")
    all_run_results = {}
    for d_config in DATASETS_CONFIG:
        if not os.path.exists(d_config["data_path"]):
            print(f"Crit Error: Data file {d_config['data_path']} for '{d_config['name_id']}' not found. Skip.")
            continue
        if "gt_path" not in d_config or not os.path.exists(d_config["gt_path"]):
            print(
                f"Warn: GT file path missing or file {d_config.get('gt_path')} for '{d_config['name_id']}' not found. Eval limited.")

        algorithms_to_run = ["ges", "pc", "dag-gnn"]

        for algo_name in algorithms_to_run:
            print(f"\nRunning {algo_name.upper()} on {d_config['name_id']} with current parameters:")
            print(
                f"  Parameters: {d_config.get('algo_params_override', {}).get(algo_name.lower(), 'Using algorithm defaults')}")

            experiment_key = f"{d_config['name_id']}_{algo_name.lower()}"
            result_metrics = run_single_experiment(d_config, algorithm_name=algo_name, viz_enabled=True,
                                                   verbose_eval=True)
            if result_metrics:
                all_run_results[experiment_key] = result_metrics
            else:
                print(f"Experiment {experiment_key} failed or no metrics.")

    print("\n\n--- Overall Experiment Results Summary ---")
    if not all_run_results:
        print("No experiments were successfully run or no metrics were collected.")
    else:
        results_summary_df = pd.DataFrame.from_dict(all_run_results, orient='index')
        preferred_column_order = [
            'execution_time_sec', 'shd_custom_cpdag_vs_dag', 'f1_skeleton',
            'precision_skeleton', 'recall_skeleton', 'tp_skeleton', 'fp_skeleton', 'fn_skeleton',
            'f1_strict_directed', 'precision_strict_directed', 'recall_strict_directed',
            'tp_strict_directed', 'fp_strict_directed', 'fn_strict_directed'
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