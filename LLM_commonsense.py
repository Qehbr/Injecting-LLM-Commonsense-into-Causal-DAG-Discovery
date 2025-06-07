import itertools
import time
import os
from llm_utils.data_processing import get_feature_names, write_causal_edges

from llm_utils.llm_query import init_llm, query_llm_for_causality

API_CALL_DELAY_SECONDS = 2
CHOSEN_LLM = 'claude'
GEMINI_MODEL_NAME = 'gemini-2.0-flash'
CLAUDE_MODEL_NAME = 'claude-sonnet-4-20250514'
LLM_OUTPUT_BASE_DIR = os.path.join("datasets", CHOSEN_LLM)

DATASETS_TO_PROCESS = [
    {
        "filename": "sachs_observational_continuous.csv",
        "domain": "cellular signaling pathways in human immune system cells",
    },
    {
        "filename": "asia_N2000.csv",
        "domain": "medical diagnosis related to lung conditions",
    },
]


def process_dataset_with_llm(llm_client,
                             llm_type_chosen,
                             dataset_info,
                             base_output_dir,
                             api_delay_seconds,
                             claude_model_for_query=None):
    """
    Orchestrates querying an LLM to generate causal priors for a dataset.

    This function iterates through all ordered pairs of features in a given
    dataset, queries a Large Language Model (LLM) to determine if a direct
    causal relationship exists, and saves the identified causal edges to a
    CSV file.

    Parameters
    ----------
    llm_client : object
        An initialized client object for the target LLM (e.g., from `init_llm`).
    llm_type_chosen : str
        The type of LLM being used, e.g., 'gemini' or 'claude'.
    dataset_info : dict
        A dictionary containing metadata for the dataset, requiring keys
        'filename' (str) and 'domain' (str).
    base_output_dir : str
        The path to the directory where the output CSV of causal edges will be
        saved.
    api_delay_seconds : int or float
        The number of seconds to wait between consecutive API calls to the LLM.
    claude_model_for_query : str, optional
        The specific model name required when `llm_type_chosen` is 'claude'.
        Default is None.
    """
    filename_with_ext = dataset_info["filename"]
    domain = dataset_info["domain"]
    dataset_name = os.path.splitext(filename_with_ext)[0]
    output_csv_filename = f"{dataset_name}_causal_edges.csv"
    output_csv_path = os.path.join(base_output_dir, output_csv_filename)

    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)

    csv_filepath = os.path.join("./datasets", filename_with_ext)
    feature_names = get_feature_names(csv_filepath)

    if not feature_names:
        return

    causal_edges_found = []
    total_pairs = len(feature_names) * (len(feature_names) - 1)

    if total_pairs <= 0:
        return

    current_pair_count = 0
    for feature_a, feature_b in itertools.permutations(feature_names, 2):
        current_pair_count += 1
        is_causal = query_llm_for_causality(
            llm_client=llm_client,
            llm_type=llm_type_chosen,
            feature_a=feature_a,
            feature_b=feature_b,
            dataset_domain=domain,
            claude_model_name=claude_model_for_query if llm_type_chosen == 'claude' else None
        )
        if is_causal:
            causal_edges_found.append((feature_a, feature_b))

        if current_pair_count < total_pairs:
            time.sleep(api_delay_seconds)

    write_causal_edges(output_csv_path, causal_edges_found)


if __name__ == "__main__":
    """
    Main execution block to run the LLM-based causal prior generation process.
    """
    if not os.path.exists(LLM_OUTPUT_BASE_DIR):
        os.makedirs(LLM_OUTPUT_BASE_DIR)

    llm_client_instance = None
    if CHOSEN_LLM == 'gemini':
        llm_client_instance = init_llm(llm_type='gemini', gemini_model_name=GEMINI_MODEL_NAME)
    elif CHOSEN_LLM == 'claude':
        llm_client_instance = init_llm(llm_type='claude')
    else:
        exit(1)

    if not llm_client_instance:
        exit(1)

    for dataset_config in DATASETS_TO_PROCESS:
        process_dataset_with_llm(
            llm_client=llm_client_instance,
            llm_type_chosen=CHOSEN_LLM,
            dataset_info=dataset_config,
            base_output_dir=LLM_OUTPUT_BASE_DIR,
            api_delay_seconds=API_CALL_DELAY_SECONDS,
            claude_model_for_query=CLAUDE_MODEL_NAME if CHOSEN_LLM == 'claude' else None
        )
