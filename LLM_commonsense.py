# main.py
import itertools
import time
import os

import anthropic

from llm_utils.data_processing import get_feature_names, write_causal_edges

# Removed ensure_dummy_datasets_exist as it wasn't used, can be re-added if needed
from llm_utils.llm_query import init_llm, query_llm_for_causality

# --- Configuration ---
API_CALL_DELAY_SECONDS = 2

# LLM Choice and Configuration
CHOSEN_LLM = 'claude'  # Options: 'gemini', 'claude'
GEMINI_MODEL_NAME = 'gemini-2.0-flash'  # Used if CHOSEN_LLM is 'gemini'
CLAUDE_MODEL_NAME = 'claude-sonnet-4-20250514'  # Used if CHOSEN_LLM is 'claude'

LLM_OUTPUT_BASE_DIR = os.path.join("datasets", CHOSEN_LLM)  # Dynamic output directory

DATASETS_TO_PROCESS = [
    {
        "filename": "sachs_observational_continuous.csv",
        "domain": "cellular signaling pathways in human immune system cells",
    },
    {
        "filename": "asia_N2000.csv",
        "domain": "medical diagnosis related to lung conditions",
    },
    # {
    #     "filename": "test.csv",
    #     "domain": "medical diagnosis related to lung conditions",
    # }
]


def process_dataset_with_llm(llm_client,
                             llm_type_chosen,
                             dataset_info,
                             base_output_dir,
                             api_delay_seconds,
                             claude_model_for_query=None):  # Pass Claude model name for querying
    """
    Processes a single dataset: reads features, queries LLM for all pairs,
    and writes asserted causal links to an output file in the specified directory.
    """
    filename_with_ext = dataset_info["filename"]
    domain = dataset_info["domain"]
    dataset_name = os.path.splitext(filename_with_ext)[0]
    output_csv_filename = f"{dataset_name}_causal_edges.csv"
    output_csv_path = os.path.join(base_output_dir, output_csv_filename)

    print(f"\n>>> Processing dataset: {filename_with_ext} with {llm_type_chosen.upper()} for domain: '{domain}' <<<")
    print(f"Output will be saved to: {output_csv_path}")

    if not os.path.exists(base_output_dir):
        try:
            os.makedirs(base_output_dir)
            print(f"Created LLM output directory: {base_output_dir}")
        except OSError as e:
            print(
                f"Error creating directory {base_output_dir}: {e}. Skipping dataset processing for {filename_with_ext}.")
            return

    csv_filepath = os.path.join("./datasets", filename_with_ext)
    feature_names = get_feature_names(csv_filepath)

    if not feature_names:
        print(f"Could not retrieve feature names for {filename_with_ext}. Skipping.")
        return

    print(f"Features found: {feature_names}")
    causal_edges_found = []
    total_pairs = len(feature_names) * (len(feature_names) - 1)

    if total_pairs <= 0:
        print("Not enough features to form pairs. Skipping.")
        return

    current_pair_count = 0
    for feature_a, feature_b in itertools.permutations(feature_names, 2):
        current_pair_count += 1
        print(f"\nProcessing pair {current_pair_count}/{total_pairs} for {filename_with_ext}:")

        try:
            is_causal = query_llm_for_causality(
                llm_client=llm_client,
                llm_type=llm_type_chosen,
                feature_a=feature_a,
                feature_b=feature_b,
                dataset_domain=domain,
                claude_model_name=claude_model_for_query if llm_type_chosen == 'claude' else None
                # max_retries and retry_delay_seconds can be passed here if needed
            )
            if is_causal:
                causal_edges_found.append((feature_a, feature_b))
                print(f"Causal link asserted by {llm_type_chosen.upper()}: {feature_a} -> {feature_b}")

        except (RuntimeError, ValueError, EnvironmentError,
                anthropic.APIError if anthropic else Exception) as e:  # Catch specific and general errors
            print(f"ERROR processing pair {feature_a} -> {feature_b} with {llm_type_chosen.upper()}: {e}")
            print("Skipping this pair and continuing with the next.")
            # Optionally log this error
        except Exception as e:  # Catch any other unexpected errors
            print(f"UNEXPECTED ERROR processing pair {feature_a} -> {feature_b} with {llm_type_chosen.upper()}: {e}")
            print("Skipping this pair and continuing with the next.")

        if current_pair_count < total_pairs:
            print(f"Waiting for {api_delay_seconds} seconds before next API call...")
            time.sleep(api_delay_seconds)

    write_causal_edges(output_csv_path, causal_edges_found)
    print(
        f"Total causal edges found for {filename_with_ext} using {llm_type_chosen.upper()}: {len(causal_edges_found)}")


if __name__ == "__main__":
    print(f"--- Starting LLM Commonsense Causality Discovery using {CHOSEN_LLM.upper()} ---")

    if not os.path.exists(LLM_OUTPUT_BASE_DIR):
        try:
            os.makedirs(LLM_OUTPUT_BASE_DIR)
            print(f"Created base LLM output directory: {LLM_OUTPUT_BASE_DIR}")
        except OSError as e:
            print(f"Critical error creating base LLM output directory {LLM_OUTPUT_BASE_DIR}: {e}. Exiting.")
            exit(1)  # Exit with an error code

    # Initialize the chosen LLM
    llm_client_instance = None
    try:
        if CHOSEN_LLM == 'gemini':
            llm_client_instance = init_llm(llm_type='gemini', gemini_model_name=GEMINI_MODEL_NAME)
        elif CHOSEN_LLM == 'claude':
            llm_client_instance = init_llm(llm_type='claude')  # API key from env, model name is per-call
        else:
            print(f"Error: Invalid CHOSEN_LLM '{CHOSEN_LLM}'. Must be 'gemini' or 'claude'.")
            exit(1)

    except (ImportError, EnvironmentError, RuntimeError) as e:
        print(f"LLM initialization failed: {e}")
        exit(1)  # Exit if LLM cannot be initialized

    if not llm_client_instance:  # Should be caught by exceptions above, but as a safeguard
        print("LLM client initialization returned None unexpectedly. Exiting.")
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

    print("\n--- All processing complete. ---")
