# project/main_optuna.py
import os
import numpy as np
import pandas as pd
import optuna
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

# --- Configuration for Datasets (remains the same) ---
DATASETS_CONFIG = [
    {
        "name_id": "asia_n2000",
        "data_path": "datasets/asia_N2000.csv",
        "gt_path": "datasets/asia_ground_truth_edges.txt",
        "data_type": "discrete",
        "n_samples_for_batch": 2000,  # Manually add or load this before Optuna
        "algo_params_override": {
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
        "n_samples_for_batch": 853,  # Manually add or load this before Optuna
        "algo_params_override": {
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
    trial_num = trial.number

    print(f"\n--- Optuna Trial #{trial_num} for DAG-GNN on {dataset_id_str} (Optimizing SHD) ---")

    if trial_num == 0:
        # First trial: use exact defaults
        suggested_params = {
            "epochs": 300,
            "lr": 3e-3,
            "batch_size": 100,
            "lambda_A": 0.01,
            "c_A": 1.0,
            "k_max_iter": 150,
            "graph_threshold": 0.3,
            "encoder_hidden": 64,
            "decoder_hidden": 64,
            "z_dims": 1,
            "h_tol": 1e-8,
            "seed": 42
        }
        print("Using original DAG-GNN defaults for first trial")
    else:

        suggested_params = {
            # Narrow ranges around defaults
            "epochs": trial.suggest_int("epochs", 200, 500, step=50),  # Default: 300
            "lr": trial.suggest_float("lr", 1e-3, 1e-2, log=True),  # Default: 3e-3
            "batch_size": 100,
            "lambda_A": trial.suggest_float("lambda_A", 0.005, 0.1, log=True),  # Default: 0.01
            "c_A": trial.suggest_float("c_A", 0.5, 5.0, log=True),  # Default: 1.0
            "k_max_iter": trial.suggest_int("k_max_iter", 100, 200, step=25),  # Default: 150
            "graph_threshold": trial.suggest_float("graph_threshold", 0.2, 0.4, step=0.05),  # Default: 0.3
            "encoder_hidden": trial.suggest_categorical("encoder_hidden", [32, 64, 96]),  # Default: 64
            "decoder_hidden": trial.suggest_categorical("decoder_hidden", [32, 64, 96]),  # Default: 64
            "z_dims": trial.suggest_int("z_dims", 1, 3),  # Default: 1
            "h_tol": trial.suggest_float("h_tol", 1e-9, 1e-7, log=True),  # Default: 1e-8
            "seed": 42
        }
        print(f"Optimizing around defaults - Trial {trial_num}")

    print(f"Suggested params: {suggested_params}")

    temp_dataset_config = dataset_config_entry.copy()
    # Ensure the nested dictionary structure for parameters is correctly built
    if "algo_params_override" not in temp_dataset_config:
        temp_dataset_config["algo_params_override"] = {}
    if algorithm_name.lower() not in temp_dataset_config["algo_params_override"]:
        temp_dataset_config["algo_params_override"][algorithm_name.lower()] = {}
    temp_dataset_config["algo_params_override"][algorithm_name.lower()].update(suggested_params)

    trial_artifacts_folder = os.path.join(base_output_folder, dataset_id_str, algorithm_name, f"trial_{trial_num}")
    ensure_visualization_folder(trial_artifacts_folder)

    data_np = None
    variable_names = None
    try:
        data_np, variable_names, _ = load_data(temp_dataset_config["data_path"])
        # The faulty dynamic batch size update block has been removed here.
    except Exception as e:
        print(f"  Error loading data for trial #{trial_num}: {e}. Pruning.")
        raise optuna.exceptions.TrialPruned()

    true_adj_matrix_dag = None
    try:
        true_edges_list = load_ground_truth_edges(temp_dataset_config["gt_path"])
        true_adj_matrix_dag = convert_edges_to_adj_matrix(true_edges_list, variable_names)
    except Exception as e:
        print(f"  Error loading ground truth for trial #{trial_num}: {e}. Pruning.")
        raise optuna.exceptions.TrialPruned()

    if true_adj_matrix_dag is None:  # Should have been pruned if GT loading failed
        print(f"  Ground truth not available for trial #{trial_num}. Pruning.")
        raise optuna.exceptions.TrialPruned()

    algorithm_instance = None
    learn_results = None
    try:
        # Parameters for the specific algorithm being tuned
        parameters_for_algo = temp_dataset_config["algo_params_override"].get(algorithm_name.lower(), {})

        algorithm_instance = get_algorithm(
            algorithm_name=algorithm_name,
            data_type=temp_dataset_config["data_type"],
            parameters_override=parameters_for_algo  # Pass only the DAG-GNN specific params
        )

        print(f"Starting DAG-GNN learn_structure for trial {trial_num} with seed {suggested_params['seed']}...")
        start_learn_time = time.time()
        # Ensure data_np is passed correctly to learn_structure (it expects 3D for DAG-GNN wrapper)
        if data_np.ndim == 2:
            data_np_for_algo = np.expand_dims(data_np, axis=2)
        else:
            data_np_for_algo = data_np
        learn_results = algorithm_instance.learn_structure(data_np_for_algo, variable_names=variable_names)
        learn_time = time.time() - start_learn_time
        print(f"Finished DAG-GNN learn_structure for trial {trial_num} in {learn_time:.2f}s.")

        learned_cpdag_adj_raw = algorithm_instance.get_learned_cpdag_adj_matrix()
        learned_strict_dag_adj = algorithm_instance.get_learned_strict_dag_adj_matrix()

    except Exception as e:
        print(f"  Trial {trial_num} failed during algorithm execution: {e}")
        import traceback
        traceback.print_exc()
        raise optuna.exceptions.TrialPruned()

    if learned_strict_dag_adj is not None and learned_cpdag_adj_raw is not None:
        eval_metrics = evaluate_dag_metrics(
            learned_strict_dag_adj, learned_cpdag_adj_raw, true_adj_matrix_dag, verbose=False
        )
        shd = eval_metrics.get("shd_custom_cpdag_vs_dag", float('inf'))
        f1_skeleton = eval_metrics.get("f1_skeleton", 0.0)
        print(
            f"Trial {trial_num} on {dataset_id_str} - SHD: {shd}, F1 Skeleton: {f1_skeleton:.4f}, Time: {learn_time:.2f}s")

        trial.set_user_attr("f1_skeleton", f1_skeleton)
        trial.set_user_attr("execution_time", learn_time)

        if learn_results and "encoder_state" in learn_results:
            model_save_path = os.path.join(trial_artifacts_folder, f"trial_{trial_num}_encoder_state.pt")
            torch.save(learn_results["encoder_state"], model_save_path)
            trial.set_user_attr("model_path", model_save_path)
            print(f"  Saved model state for trial {trial_num} to {model_save_path}")
        return shd
    else:
        print(f"Trial {trial_num} for {dataset_id_str} - Learned graph is None. Returning high SHD.")
        return float('inf')


# --- run_single_experiment (remains mostly the same, ensure verbose_eval=False for Optuna calls if modifying it)
# For brevity, its definition is omitted here but it's assumed to be the version from the previous response
# that supports loading models via "best_model_path" and "best_model_exec_time"
def run_single_experiment(dataset_config, algorithm_name="ges", viz_enabled=True, verbose_eval=True):
    # (Ensure this function is the one from the previous response that handles model loading)
    # ... it will check for "best_model_path" in dataset_config["algo_params_override"][algorithm_name.lower()]
    # ... and call algorithm_instance.load_trained_model(...) if present for dag-gnn.
    # (Copied from previous for completeness, ensure it's aligned)
    dataset_id_str = dataset_config['name_id']
    if verbose_eval: print(f"\n--- Processing Dataset: {dataset_id_str} with Algorithm: {algorithm_name.upper()} ---")

    current_run_viz_folder = None
    if viz_enabled:
        ensure_visualization_folder(MAIN_VIZ_OUTPUT_FOLDER)
        dataset_viz_subfolder_name = f"{dataset_id_str}_{algorithm_name.lower()}"
        current_run_viz_folder = os.path.join(MAIN_VIZ_OUTPUT_FOLDER, dataset_viz_subfolder_name)
        ensure_visualization_folder(current_run_viz_folder)

    data_np_orig, variable_names, _ = load_data(dataset_config["data_path"])

    # Prepare data_np in the shape expected by algorithms (especially DAG-GNN)
    if data_np_orig.ndim == 2:
        data_np_for_algo = np.expand_dims(data_np_orig, axis=2)
    else:
        data_np_for_algo = data_np_orig

    n_nodes = data_np_for_algo.shape[1]
    data_feature_dim_or_num_classes = 0
    if dataset_config["data_type"] == 'discrete':
        data_feature_dim_or_num_classes = int(np.max(data_np_for_algo)) + 1
    else:  # continuous
        data_feature_dim_or_num_classes = data_np_for_algo.shape[2]

    if verbose_eval: print(f"  Data loaded: {data_np_orig.shape[0]} samples, {n_nodes} variables.")

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
        # Correctly get parameters for the specific algorithm
        algo_params_config_root = dataset_config.get("algo_params_override", {})
        current_algo_params = algo_params_config_root.get(algorithm_name.lower(), {})

        algorithm_instance = get_algorithm(
            algorithm_name=algorithm_name,
            data_type=dataset_config["data_type"],
            parameters_override=current_algo_params
        )

        best_model_path = current_algo_params.get("best_model_path")
        if algorithm_name.lower() == "dag-gnn" and best_model_path and os.path.exists(best_model_path):
            if verbose_eval: print(f"  Loading pre-tuned DAG-GNN model from: {best_model_path}")
            algorithm_instance.load_trained_model(
                best_model_path,
                n_nodes,
                data_feature_dim_or_num_classes,
                current_algo_params  # Pass the params that were used to train this model
            )
            execution_time = current_algo_params.get("best_model_exec_time", 0)
        else:
            if verbose_eval: print(f"  Running {algorithm_name.upper()} with params: {current_algo_params}")
            algorithm_instance.learn_structure(data_np_for_algo, variable_names=variable_names)
            execution_time = algorithm_instance.get_execution_time()

        learned_cpdag_adj_raw = algorithm_instance.get_learned_cpdag_adj_matrix()
        learned_strict_dag_adj = algorithm_instance.get_learned_strict_dag_adj_matrix()

    except Exception as e:
        print(f"  Error running {algorithm_name.upper()} on {dataset_id_str}: {e}")
        import traceback
        traceback.print_exc()
        return None

    metrics = {"execution_time_sec": execution_time}
    if true_adj_matrix_dag is not None and learned_strict_dag_adj is not None and learned_cpdag_adj_raw is not None:
        if verbose_eval: print("\n  Evaluating Learned Graph against Ground Truth...")
        eval_metrics = evaluate_dag_metrics(
            learned_strict_dag_adj, learned_cpdag_adj_raw, true_adj_matrix_dag, verbose=verbose_eval
        )
        metrics.update(eval_metrics)

    if viz_enabled and current_run_viz_folder:
        from visualization.graph_visualization import visualize_true_graph, visualize_learned_graph_nx, \
            visualize_comparison_graphs
        from visualization.plotting_utils import get_shared_layout
        viz_file_basename = f"{get_dataset_name(dataset_config['data_path'])}"
        layout_matrices = []
        if true_adj_matrix_dag is not None: layout_matrices.append(true_adj_matrix_dag)
        if learned_cpdag_adj_raw is not None: layout_matrices.append(learned_cpdag_adj_raw)
        shared_pos = {}
        if layout_matrices and variable_names: shared_pos = get_shared_layout(layout_matrices, variable_names)
        if true_adj_matrix_dag is not None:
            visualize_true_graph(true_adj_matrix_dag, variable_names, f"1_true_dag_{viz_file_basename}.png",
                                 pos=shared_pos, dataset_name_str=viz_file_basename, viz_folder=current_run_viz_folder)
        if learned_cpdag_adj_raw is not None:
            visualize_learned_graph_nx(learned_cpdag_adj_raw, variable_names,
                                       f"2_{algorithm_name.lower()}_learned_cpdag_{viz_file_basename}.png",
                                       pos=shared_pos,
                                       dataset_name_str=f"{viz_file_basename} ({algorithm_name.upper()})",
                                       viz_folder=current_run_viz_folder)
        if learned_cpdag_adj_raw is not None and true_adj_matrix_dag is not None:
            visualize_comparison_graphs(learned_cpdag_adj_raw, true_adj_matrix_dag, variable_names,
                                        f"3_comparison_{algorithm_name.lower()}_vs_true_{viz_file_basename}.png",
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
    N_OPTUNA_TRIALS_PER_DATASET = 5
    # Keep low for testing, increase for thorough search

    if PERFORM_OPTUNA_TUNING and ALGORITHM_TO_TUNE.lower() == "dag-gnn":
        ensure_visualization_folder(OPTUNA_OUTPUT_BASE_FOLDER)
        for dataset_name_to_tune in DATASETS_FOR_DAGNN_TUNING:
            dataset_config_for_tuning = next(
                (item for item in DATASETS_CONFIG if item["name_id"] == dataset_name_to_tune), None
            )
            if not dataset_config_for_tuning:
                print(f"Config for dataset '{dataset_name_to_tune}' not found. Skipping Optuna tuning for it.")
                continue

            # Load n_samples once for this dataset to inform batch_size choices
            try:
                temp_data_for_nsamples, _, _ = load_data(dataset_config_for_tuning["data_path"])
                dataset_config_for_tuning['n_samples_for_batch'] = temp_data_for_nsamples.shape[0]
            except Exception as e:
                print(
                    f"Warning: Could not load data for {dataset_name_to_tune} to determine n_samples for batch_size. Error: {e}")
                dataset_config_for_tuning['n_samples_for_batch'] = 256  # Fallback

            print(f"\n--- Starting Optuna Tuning for DAG-GNN on {dataset_name_to_tune} (Optimizing SHD) ---")
            study_name = f"dagnn_shd_tuning_{dataset_name_to_tune}"
            study = optuna.create_study(direction="minimize",
                                        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3))  # n_warmup_steps

            study.optimize(
                lambda trial: dagnn_objective(trial, dataset_config_for_tuning, OPTUNA_OUTPUT_BASE_FOLDER),
                n_trials=N_OPTUNA_TRIALS_PER_DATASET,
            )

            print(f"\n--- Optuna Tuning Finished for DAG-GNN on {dataset_name_to_tune} ---")
            if not study.trials or study.best_trial is None:  # Check if any trial completed
                print(f"  No successful trials completed for {dataset_name_to_tune}. Cannot determine best parameters.")
            else:
                best_trial_overall = study.best_trial
                print(
                    f"  Best SHD: {best_trial_overall.value:.4f} (F1 Skeleton: {best_trial_overall.user_attrs.get('f1_skeleton', 'N/A')})")
                print("  Best Params from this study: ")
                for key, value in best_trial_overall.params.items():
                    print(f"    {key}: {value}")

                for i, config_item in enumerate(DATASETS_CONFIG):
                    if config_item["name_id"] == dataset_name_to_tune:
                        if "algo_params_override" not in DATASETS_CONFIG[i]: DATASETS_CONFIG[i][
                            "algo_params_override"] = {}
                        if ALGORITHM_TO_TUNE.lower() not in DATASETS_CONFIG[i]["algo_params_override"]:
                            DATASETS_CONFIG[i]["algo_params_override"][ALGORITHM_TO_TUNE.lower()] = {}

                        DATASETS_CONFIG[i]["algo_params_override"][ALGORITHM_TO_TUNE.lower()].update(
                            best_trial_overall.params)

                        saved_model_path = best_trial_overall.user_attrs.get("model_path")
                        if saved_model_path:
                            DATASETS_CONFIG[i]["algo_params_override"][ALGORITHM_TO_TUNE.lower()][
                                "best_model_path"] = saved_model_path
                            DATASETS_CONFIG[i]["algo_params_override"][ALGORITHM_TO_TUNE.lower()][
                                "best_model_exec_time"] = best_trial_overall.user_attrs.get("execution_time")
                            print(f"  Will use saved model from: {saved_model_path}")
                        else:
                            print(
                                f"  Warning: No 'best_model_path' found in best_trial user_attrs. Will retrain with best params.")
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
            experiment_key = f"{d_config['name_id']}_{algo_name.lower()}"
            result_metrics = run_single_experiment(d_config, algorithm_name=algo_name, viz_enabled=True,
                                                   verbose_eval=True)
            if result_metrics: all_run_results[experiment_key] = result_metrics

    print("\n\n--- Overall Experiment Results Summary ---")
    # ... (results summary printing code from before) ...
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
