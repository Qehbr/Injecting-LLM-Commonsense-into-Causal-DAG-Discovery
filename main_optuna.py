# project/main_optuna.py
import os
import numpy as np
import pandas as pd
import optuna
import time
import torch
# import os # Already imported
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # For PyTorch debugging if needed

from utils.data_processing import (
    load_data, load_ground_truth_edges,
    convert_edges_to_adj_matrix, get_dataset_name
)
from utils.evaluation import evaluate_dag_metrics
from visualization.graph_visualization import (
    ensure_visualization_folder
)
from algorithms.algorithm_factory import get_algorithm

# --- Standard Parameters for DAG-GNN (Base for Optuna & variants) ---
DAG_GNN_BASE_PARAMS = {
    "epochs": 300, "lr": 1e-3,  # Optuna will likely override these in its suggestions
    "lambda_A": 0.0,
    "c_A": 1.0,
    "k_max_iter": 100,
    "graph_threshold": 0.3,
    "h_tol": 1e-8,
    "tau_A": 0.0,
    "encoder_hidden": 64,
    "decoder_hidden": 64,
    "z_dims": 3,  # This should ideally be dataset-specific or tuned
    "seed": 42,
    "variance": 0.1,  # Default variance for NLL
    # "llm_prior_init_value" will be added by Optuna for relevant variants
    # or taken from here as a default if not tuned.
    "llm_prior_init_value": 0.3  # Default strength if not tuned by Optuna
}

# --- Configuration for Datasets ---
DATASETS_CONFIG = [
    {
        "name_id": "asia_n2000",
        "data_path": "datasets/asia_N2000.csv",
        "gt_path": "datasets/asia_ground_truth_edges.txt",
        "data_type": "discrete",
        "n_samples_for_batch": 2000,
        "n_nodes_approx": 8,  # Approximate number of nodes for z_dims default
        "algo_params_override": {
            "ges": {"bdeu_sample_prior": 10.0},
            "pc": {"alpha": 0.05, "indep_test": "gsq"},
            "dag-gnn": {**DAG_GNN_BASE_PARAMS, 'z_dims': 4},  # Override z_dims for asia
            "dag-gnn-gemini": {
                **DAG_GNN_BASE_PARAMS, 'z_dims': 4,
                "llm_prior_edges_path": "datasets/gemini/asia_N2000_causal_edges.csv"
            },
            "dag-gnn-claude": {
                **DAG_GNN_BASE_PARAMS, 'z_dims': 4,
                "llm_prior_edges_path": "datasets/claude/asia_N2000_causal_edges.csv"
            }
        }
    },
    {
        "name_id": "sachs_continuous",
        "data_path": "datasets/sachs_observational_continuous.csv",
        "gt_path": "datasets/sachs_ground_truth_edges_17.txt",
        # Ensure this GT file matches the 11-node data if that's what sachs_observational_continuous.csv is
        "data_type": "continuous",
        "n_samples_for_batch": 853,
        "n_nodes_approx": 11,  # For sachs
        "algo_params_override": {
            "ges": {"penalty_discount": 2.0},
            "pc": {"alpha": 0.05},
            "dag-gnn": {**DAG_GNN_BASE_PARAMS, "tau_A": 0.01, 'z_dims': 5},  # Override z_dims for sachs
            "dag-gnn-gemini": {
                **DAG_GNN_BASE_PARAMS, "tau_A": 0.01, 'z_dims': 5,
                "llm_prior_edges_path": "datasets/gemini/sachs_observational_continuous_causal_edges.csv"
            },
            "dag-gnn-claude": {
                **DAG_GNN_BASE_PARAMS, "tau_A": 0.01, 'z_dims': 5,
                "llm_prior_edges_path": "datasets/claude/sachs_observational_continuous_causal_edges.csv"
            }
        }
    }
]
MAIN_VIZ_OUTPUT_FOLDER = "visualizations_output_project"  # For final runs after tuning
OPTUNA_OUTPUT_BASE_FOLDER = "optuna_study_results"  # For Optuna trial artifacts


def dagnn_objective(trial, dataset_config_entry, algorithm_name_being_tuned, base_optuna_artifacts_folder):
    dataset_id_str = dataset_config_entry['name_id']
    trial_num = trial.number
    num_nodes_approx = dataset_config_entry.get('n_nodes_approx', 10)  # Get n_nodes for dynamic z_dims

    print(f"\n--- Optuna Trial #{trial_num} for {algorithm_name_being_tuned.upper()} on {dataset_id_str} ---")

    # Define hyperparameters to be tuned by Optuna
    suggested_params = {
        "epochs": trial.suggest_int("epochs", 100, 400, step=50),  # Adjusted range
        "lr": trial.suggest_float("lr", 5e-4, 1e-2, log=True),  # Adjusted range
        "batch_size": trial.suggest_categorical("batch_size", [64, 100, 128, min(256, dataset_config_entry.get(
            'n_samples_for_batch', 256))]),
        # lambda_A is often 0 when using the augmented Lagrangian approach with c_A and h_A
        "lambda_A": 0.0,  # Kept fixed as per common DAG-GNN setup
        "c_A": trial.suggest_float("c_A", 0.01, 20.0, log=True),  # Wider range for c_A
        "k_max_iter": trial.suggest_int("k_max_iter", 20, 100, step=10),  # Adjusted range
        "graph_threshold": trial.suggest_float("graph_threshold", 0.1, 0.4, step=0.05),
        "encoder_hidden": trial.suggest_categorical("encoder_hidden", [32, 64, 128]),
        "decoder_hidden": trial.suggest_categorical("decoder_hidden", [32, 64, 128]),
        "z_dims": trial.suggest_int("z_dims", max(1, num_nodes_approx // 3), num_nodes_approx),  # Dynamic z_dims
        "h_tol": trial.suggest_float("h_tol", 1e-9, 1e-5, log=True),
        "tau_A": trial.suggest_float("tau_A", 0.0, 0.05, step=0.005),  # Allow 0, up to a bit more L1
        "variance": trial.suggest_float("variance", 0.05, 0.5, log=True),
        "seed": 42  # Keep seed fixed for reproducibility of a single trial's result
    }

    # Conditionally add llm_prior_init_value to tuning if it's an LLM variant
    if "gemini" in algorithm_name_being_tuned or "claude" in algorithm_name_being_tuned:
        suggested_params["llm_prior_init_value"] = trial.suggest_float("llm_prior_init_value", 0.05, 0.7, step=0.05)

    print(f"Optuna Suggested params for trial {trial_num}: {suggested_params}")

    # --- Prepare config for this specific trial run ---
    # Start with the base config for this algorithm variant from DATASETS_CONFIG
    # This ensures static parts like llm_prior_edges_path are included
    trial_algo_params = dataset_config_entry["algo_params_override"].get(algorithm_name_being_tuned, {}).copy()

    # Update with Optuna's suggested parameters for this trial
    trial_algo_params.update(suggested_params)

    # --- Execute the algorithm with these parameters ---
    trial_artifacts_folder = os.path.join(base_optuna_artifacts_folder, dataset_id_str, algorithm_name_being_tuned,
                                          f"trial_{trial_num}")
    ensure_visualization_folder(trial_artifacts_folder)  # For saving models if needed

    try:
        data_np, variable_names, _ = load_data(dataset_config_entry["data_path"])
        true_edges_list = load_ground_truth_edges(dataset_config_entry["gt_path"])
        true_adj_matrix_dag = convert_edges_to_adj_matrix(true_edges_list, variable_names)
    except Exception as e:
        print(f"  Error loading data/GT for trial #{trial_num} on {dataset_id_str}: {e}. Pruning.")
        raise optuna.exceptions.TrialPruned()

    if true_adj_matrix_dag is None:
        print(f"  Ground truth not available for trial #{trial_num} on {dataset_id_str}. Pruning.")
        raise optuna.exceptions.TrialPruned()

    # Prepare data shape for DAG-GNN (expects 3D)
    if data_np.ndim == 2:
        data_np_for_algo = np.expand_dims(data_np, axis=2)
    elif data_np.ndim == 3 and data_np.shape[2] != 1 and dataset_config_entry["data_type"] == "discrete":
        # This case needs careful handling for discrete multivariate, but DAG-GNN module assumes univariate discrete
        print(
            f"Warning: Discrete data has multiple features per node ({data_np.shape[2]}). DAG-GNN's discrete mode expects 1.")
        data_np_for_algo = data_np[..., [0]]  # Take only the first feature for discrete
    else:
        data_np_for_algo = data_np

    learn_results = None
    learned_strict_dag_adj = None
    learned_cpdag_adj_raw = None
    learn_time = -1.0

    try:
        algorithm_instance = get_algorithm(
            algorithm_name=algorithm_name_being_tuned,  # Use the specific variant name
            data_type=dataset_config_entry["data_type"],
            parameters_override=trial_algo_params  # Pass the combined (base + Optuna) params
        )

        print(f"  Starting {algorithm_name_being_tuned.upper()} learn_structure for trial {trial_num}...")
        start_learn_time = time.time()
        learn_results = algorithm_instance.learn_structure(data_np_for_algo, variable_names=variable_names)
        learn_time = time.time() - start_learn_time
        print(f"  Finished learn_structure for trial {trial_num} in {learn_time:.2f}s.")

        learned_cpdag_adj_raw = algorithm_instance.get_learned_cpdag_adj_matrix()
        learned_strict_dag_adj = algorithm_instance.get_learned_strict_dag_adj_matrix()

    except Exception as e:
        print(
            f"  Trial {trial_num} for {algorithm_name_being_tuned.upper()} on {dataset_id_str} failed during algorithm execution: {e}")
        import traceback
        traceback.print_exc()
        # For Optuna, it's better to return a very bad score than prune if some result can be salvaged
        # However, if execution fails completely, pruning or returning a penalty value is common.
        # Let's return a clear failure metric (e.g., negative F1 or high SHD).
        # For maximization (F1), return 0.0 or a very small number.
        return 0.0  # Or raise optuna.exceptions.TrialPruned()

    if learned_strict_dag_adj is not None and learned_cpdag_adj_raw is not None:
        eval_metrics = evaluate_dag_metrics(
            learned_strict_dag_adj, learned_cpdag_adj_raw, true_adj_matrix_dag, verbose=False
            # Keep Optuna logs concise
        )
        # Choose your primary metric for Optuna to optimize
        # Example: F1 score for strictly directed edges
        metric_to_optimize = eval_metrics.get("f1_strict_directed", 0.0)

        shd = eval_metrics.get("shd_custom_cpdag_vs_dag", float('inf'))
        f1_skeleton = eval_metrics.get("f1_skeleton", 0.0)

        print(
            f"  Trial {trial_num} ({algorithm_name_being_tuned}) on {dataset_id_str} - F1 Strict: {metric_to_optimize:.4f}, SHD: {shd}, F1 Skel: {f1_skeleton:.4f}, Time: {learn_time:.2f}s")

        # Store other relevant metrics or artifacts if needed
        trial.set_user_attr("shd_custom_cpdag_vs_dag", shd)
        trial.set_user_attr("f1_skeleton", f1_skeleton)
        trial.set_user_attr("execution_time_sec", learn_time)

        if learn_results and "encoder_state" in learn_results and hasattr(learn_results["encoder_state"],
                                                                          'keys'):  # Check if it's a state dict
            model_save_path = os.path.join(trial_artifacts_folder, f"trial_{trial_num}_model_state.pt")
            try:
                torch.save(learn_results["encoder_state"], model_save_path)
                trial.set_user_attr("model_path", model_save_path)
                print(f"  Saved model state for trial {trial_num} to {model_save_path}")
            except Exception as e_save:
                print(f"  Warning: Could not save model state for trial {trial_num}: {e_save}")

        return metric_to_optimize  # Optuna will try to maximize this
    else:
        print(
            f"  Trial {trial_num} ({algorithm_name_being_tuned}) on {dataset_id_str} - Learned graph is None. Returning low optimization metric (0.0).")
        return 0.0


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

    # --- START: Corrected Data Reshaping Block ---
    data_np_for_algo = None
    if algorithm_name.lower() == "dag-gnn":
        # DAG-GNN expects a 3D array: (samples, nodes, features_per_node)
        if data_np_orig.ndim == 2:
            data_np_for_algo = np.expand_dims(data_np_orig, axis=2)
        else:
            data_np_for_algo = data_np_orig
    else:
        # GES and PC expect a 2D array: (samples, nodes)
        if data_np_orig.ndim == 3 and data_np_orig.shape[2] == 1:
            data_np_for_algo = np.squeeze(data_np_orig, axis=2)
        elif data_np_orig.ndim == 2:
            data_np_for_algo = data_np_orig
        else:
            # Handle unexpected shapes if necessary
            raise ValueError(f"Unsupported data shape {data_np_orig.shape} for {algorithm_name.upper()}")
    # --- END: Corrected Data Reshaping Block ---

    n_nodes = data_np_for_algo.shape[1]

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
        if algorithm_name.lower() == "dag-gnn":
            # Calculate this ONLY for DAG-GNN
            if dataset_config["data_type"] == 'discrete':
                # Ensure data_np_for_algo is appropriate for np.max (it will be 3D [N, nodes, 1] for discrete DAG-GNN)
                temp_data_for_max = data_np_for_algo
                if temp_data_for_max.ndim == 3 and temp_data_for_max.shape[2] == 1:
                    temp_data_for_max = np.squeeze(data_np_for_algo, axis=2)  # Use squeezed version for np.max
                data_feature_dim_or_num_classes = int(np.max(temp_data_for_max)) + 1
            else:  # continuous for DAG-GNN (data_np_for_algo is 3D: [N, nodes, features_per_node])
                data_feature_dim_or_num_classes = data_np_for_algo.shape[2]

            if best_model_path and os.path.exists(best_model_path):
                if verbose_eval: print(f"  Loading pre-tuned DAG-GNN model from: {best_model_path}")
                algorithm_instance.load_trained_model(
                    best_model_path,
                    n_nodes,
                    data_feature_dim_or_num_classes,  # Correctly used here
                    current_algo_params
                )
                execution_time = current_algo_params.get("best_model_exec_time", 0)
            else:
                # If training DAG-GNN from scratch, it will use data_feature_dim_or_num_classes internally
                # The _setup_hyperparameters method in DAG_GNN_Algorithm should handle this
                if verbose_eval: print(f"  Running {algorithm_name.upper()} with params: {current_algo_params}")
                algorithm_instance.learn_structure(data_np_for_algo, variable_names=variable_names)
                execution_time = algorithm_instance.get_execution_time()
        else:  # For GES, PC
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
# --- Main Execution Block ---
if __name__ == "__main__":
    PERFORM_OPTUNA_TUNING = True
    # Define which DAG-GNN variants to tune
    # Ensure your LLM_commonsense.py script has run and created the prior files for these.
    DAG_GNN_VARIANTS_TO_TUNE = ["dag-gnn", "dag-gnn-gemini", "dag-gnn-claude"]
    # Or focus: DAG_GNN_VARIANTS_TO_TUNE = ["dag-gnn-gemini", "dag-gnn-claude"]

    DATASETS_FOR_TUNING = ["asia_n2000", "sachs_continuous"]
    N_OPTUNA_TRIALS_PER_DATASET_VARIANT = 1  # Number of trials for each (dataset, algo_variant) combination

    # Create a directory for all Optuna outputs if it doesn't exist
    ensure_visualization_folder(OPTUNA_OUTPUT_BASE_FOLDER)

    if PERFORM_OPTUNA_TUNING:
        for algo_variant_name in DAG_GNN_VARIANTS_TO_TUNE:
            print(f"\n===== Starting Optuna Tuning for Algorithm Variant: {algo_variant_name.upper()} =====")
            for dataset_name_to_tune in DATASETS_FOR_TUNING:
                dataset_config_for_tuning = next(
                    (item for item in DATASETS_CONFIG if item["name_id"] == dataset_name_to_tune), None
                )
                if not dataset_config_for_tuning:
                    print(f"Config for dataset '{dataset_name_to_tune}' not found. Skipping Optuna for it.")
                    continue

                # Check if LLM prior file exists if tuning an LLM variant
                if "gemini" in algo_variant_name or "claude" in algo_variant_name:
                    prior_path_key = "llm_prior_edges_path"
                    # Get the specific parameter block for this algorithm variant
                    variant_params = dataset_config_for_tuning["algo_params_override"].get(algo_variant_name, {})
                    actual_prior_path = variant_params.get(prior_path_key)

                    if not actual_prior_path:
                        print(
                            f"'{prior_path_key}' not defined for {algo_variant_name} on {dataset_name_to_tune}. Skipping Optuna.")
                        continue
                    if not os.path.exists(actual_prior_path):
                        print(
                            f"LLM prior file NOT FOUND for {algo_variant_name} on {dataset_name_to_tune} at path '{actual_prior_path}'. Ensure your LLM script has run. Skipping Optuna for this combination.")
                        continue
                    print(
                        f"Found LLM prior file for {algo_variant_name} on {dataset_name_to_tune}: {actual_prior_path}")

                # Load n_samples and n_nodes for this dataset to inform Optuna search space
                try:
                    temp_data, temp_var_names, _ = load_data(dataset_config_for_tuning["data_path"])
                    dataset_config_for_tuning['n_samples_for_batch'] = temp_data.shape[0]
                    dataset_config_for_tuning['n_nodes_approx'] = len(temp_var_names)
                except Exception as e:
                    print(
                        f"Warning: Could not load data for {dataset_name_to_tune} to get n_samples/n_nodes. Using defaults. Error: {e}")
                    dataset_config_for_tuning['n_samples_for_batch'] = 256  # Fallback
                    dataset_config_for_tuning['n_nodes_approx'] = 10  # Fallback

                print(
                    f"\n--- Optuna for {algo_variant_name.upper()} on {dataset_name_to_tune} (Optimizing F1 Strict Directed) ---")
                study_name = f"{algo_variant_name}_f1_strict_{dataset_name_to_tune}"
                # Use a unique database for each study to store results persistently
                study_db_path = os.path.join(OPTUNA_OUTPUT_BASE_FOLDER, f"{study_name}.db")
                study = optuna.create_study(
                    study_name=study_name,
                    storage=f"sqlite:///{study_db_path}",  # Store results in SQLite DB
                    load_if_exists=True,  # Resume study if it exists
                    direction="maximize",
                    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
                    # Adjusted pruner
                )

                # Pass algo_variant_name to the objective function
                study.optimize(
                    lambda trial: dagnn_objective(trial, dataset_config_for_tuning, algo_variant_name,
                                                  OPTUNA_OUTPUT_BASE_FOLDER),
                    n_trials=N_OPTUNA_TRIALS_PER_DATASET_VARIANT,
                    # timeout=600 # Optional: set a timeout for each study.optimize call
                )

                print(f"\n--- Optuna Tuning Finished for {algo_variant_name.upper()} on {dataset_name_to_tune} ---")
                if not study.trials or not hasattr(study, 'best_trial') or study.best_trial is None:
                    print(
                        f"  No successful trials or best_trial is None for {study_name}. Cannot determine best parameters.")
                else:
                    best_trial_overall = study.best_trial
                    print(f"  Study: {study_name}")
                    print(f"  Best Trial Number: {best_trial_overall.number}")
                    print(f"  Best F1 Strict Directed: {best_trial_overall.value:.4f}")

                    # Safely print execution time
                    exec_time_val = best_trial_overall.user_attrs.get('execution_time_sec', 'N/A')
                    exec_time_str = f"{exec_time_val:.2f}s" if isinstance(exec_time_val, (int, float)) else str(
                        exec_time_val)

                    print(f"  User Attrs: SHD: {best_trial_overall.user_attrs.get('shd_custom_cpdag_vs_dag', 'N/A')}, "
                          f"F1 Skeleton: {best_trial_overall.user_attrs.get('f1_skeleton', 'N/A')}, "
                          f"Exec Time: {exec_time_str}")  # Use exec_time_str

                    print(f"  Best Params for {algo_variant_name} on {dataset_name_to_tune}:")
                    for key, value in best_trial_overall.params.items():
                        print(f"    {key}: {value}")

                    # Update the main DATASETS_CONFIG with the best found parameters for this variant
                    for i, config_item in enumerate(DATASETS_CONFIG):
                        if config_item["name_id"] == dataset_name_to_tune:
                            if "algo_params_override" not in DATASETS_CONFIG[i]:
                                DATASETS_CONFIG[i]["algo_params_override"] = {}

                            # Ensure the specific algo_variant_name key exists
                            if algo_variant_name not in DATASETS_CONFIG[i]["algo_params_override"]:
                                DATASETS_CONFIG[i]["algo_params_override"][algo_variant_name] = {}

                            # Update with Optuna's best params
                            DATASETS_CONFIG[i]["algo_params_override"][algo_variant_name].update(
                                best_trial_overall.params)

                            # Also save model path and execution time if available
                            saved_model_path = best_trial_overall.user_attrs.get("model_path")
                            if saved_model_path:
                                DATASETS_CONFIG[i]["algo_params_override"][algo_variant_name][
                                    "best_model_path"] = saved_model_path
                                DATASETS_CONFIG[i]["algo_params_override"][algo_variant_name][
                                    "best_model_exec_time"] = best_trial_overall.user_attrs.get("execution_time_sec")
                                print(f"  Best model for this study saved at: {saved_model_path}")
                            else:
                                print("  Warning: No 'model_path' found in best_trial user_attrs for DAG-GNN variant.")
                            print(
                                f"  Updated main DATASETS_CONFIG for '{dataset_name_to_tune}' with best Optuna params for '{algo_variant_name}'.")
                            break  # Found and updated the dataset config
            print(f"\n===== Finished Optuna Tuning for Algorithm Variant: {algo_variant_name.upper()} =====")
        print("\nCompleted all Optuna tuning sessions specified.")

    print("\n--- Starting Regular Experiment Runs (Potentially with Tuned Params) ---")
    all_run_results = {}

    # Define all algorithms to run in the final evaluation, including tuned LLM variants
    final_algorithms_to_run = ["ges", "pc"] + DAG_GNN_VARIANTS_TO_TUNE

    for d_config in DATASETS_CONFIG:
        if not os.path.exists(d_config["data_path"]):
            print(
                f"Critical Error: Data file {d_config['data_path']} for '{d_config['name_id']}' not found. Skipping dataset.")
            continue
        if "gt_path" not in d_config or not os.path.exists(d_config["gt_path"]):
            print(
                f"Warning: Ground truth file path missing or file {d_config.get('gt_path')} for '{d_config['name_id']}' not found. Evaluation will be limited.")

        for algo_name_final_run in final_algorithms_to_run:
            experiment_key = f"{d_config['name_id']}_{algo_name_final_run.lower()}"
            print(f"\nRunning final experiment for: {experiment_key}")

            # The d_config should now contain the tuned parameters under algo_params_override[algo_name_final_run]
            result_metrics = run_single_experiment(
                dataset_config=d_config,  # d_config now contains tuned HPs
                algorithm_name=algo_name_final_run,
                viz_enabled=True,
                verbose_eval=True
            )
            if result_metrics:
                all_run_results[experiment_key] = result_metrics
            else:
                print(f"No metrics returned for {experiment_key}.")

    print("\n\n--- Overall Experiment Results Summary (After Tuning and Final Runs) ---")
    # ... (your existing results summary printing code) ...
    if not all_run_results:
        print("No experiments were successfully run or no metrics were collected in the final run phase.")
    else:
        results_summary_df = pd.DataFrame.from_dict(all_run_results, orient='index')
        # Define your preferred column order for the summary table
        preferred_column_order = [
            'execution_time_sec', 'shd_custom_cpdag_vs_dag', 'f1_skeleton',
            'precision_skeleton', 'recall_skeleton', 'tp_skeleton', 'fp_skeleton', 'fn_skeleton',
            'f1_strict_directed', 'precision_strict_directed', 'recall_strict_directed',
            'tp_strict_directed', 'fp_strict_directed', 'fn_strict_directed'
        ]
        # Filter for columns that actually exist in the DataFrame
        display_columns = [col for col in preferred_column_order if col in results_summary_df.columns]

        if not results_summary_df.empty and display_columns:
            print(results_summary_df[display_columns].to_string(float_format="%.4f"))
            summary_csv_path = os.path.join(MAIN_VIZ_OUTPUT_FOLDER,
                                            "experiment_results_summary_final.csv")  # Changed name
            try:
                os.makedirs(os.path.dirname(summary_csv_path), exist_ok=True)
                results_summary_df[display_columns].to_csv(summary_csv_path, float_format="%.4f")
                print(f"\nResults summary also saved to: {summary_csv_path}")
            except Exception as e:
                print(f"\nCould not save results summary CSV: {e}")
        elif not results_summary_df.empty:
            print("Results DataFrame has columns, but preferred display columns might be missing. Printing all:")
            print(results_summary_df.to_string(float_format="%.4f"))
        else:
            print("Results DataFrame is empty.")

    print("\n--- Causal Discovery Pipeline Finished ---")
