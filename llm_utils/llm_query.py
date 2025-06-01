# llm_utils/llm_query.py
import os
import time
from dotenv import load_dotenv

import google.generativeai as genai
import anthropic


PROMPT_TEMPLATE = """I am working on a project about causal discovery in Bayesian networks. I am currently examining potential direct causal relationships between variables.

Consider the following two features:
* Feature A: {feature_a}
* Feature B: {feature_b}

These features are from a dataset related to {dataset_domain}.

Please first reason step-by-step about whether Feature A could be a direct cause of Feature B in the real world, considering their nature and typical interactions. Explain your thought process.

After your reasoning, conclude with a final verdict on a new line in the exact format:
FINAL VERDICT: YES
or
FINAL VERDICT: NO

Important constraints for your reasoning and verdict:
* Focus solely on whether Feature A is a **direct cause** of Feature B.
* Base your assessment on your general commonsense and scientific understanding, **not** on any specific knowledge you might have about the 'ASIA' or 'SACHS' benchmark datasets or their established ground truth graphs.

For example:
If Feature A was 'Smoking' and Feature B was 'Lung Cancer', your response might look like:
Reasoning: Smoking introduces carcinogens into the lungs. These carcinogens can damage DNA in lung cells, leading to mutations that can result in uncontrolled cell growth, which is characteristic of cancer. This is a well-established direct biological pathway.
FINAL VERDICT: YES

If Feature A was 'Ice Cream Sales' and Feature B was 'Shark Attacks', your response might look like:
Reasoning: Ice cream sales and shark attacks might both increase in warmer weather as more people go to the beach and buy ice cream, and more people swim in the ocean. However, the act of selling or eating ice cream does not directly cause sharks to attack people. There is no direct mechanism linking the two.
FINAL VERDICT: NO

Now, analyze the direct causal relationship between {feature_a} and {feature_b}.
"""


def init_llm(llm_type='gemini',
             gemini_model_name='gemini-2.0-flash',
             gemini_api_key_env_var='GENAI_API_KEY',
             claude_api_key_env_var='ANTHROPIC_API_KEY'):
    """
    Initializes and returns the specified LLM client.
    llm_type can be 'gemini' or 'claude'.
    Raises EnvironmentError or RuntimeError on failure.
    """
    load_dotenv()  # Load .env file for API keys

    if llm_type == 'gemini':
        api_key = os.getenv(gemini_api_key_env_var)
        if not api_key:
            raise EnvironmentError(
                f"{gemini_api_key_env_var} not found. Please set it in your environment or .env file."
            )
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(gemini_model_name)
            print(f"Gemini model '{gemini_model_name}' initialized successfully.")
            return model
        except Exception as e:
            raise RuntimeError(f"Error initializing Gemini model '{gemini_model_name}': {e}") from e

    elif llm_type == 'claude':
        api_key = os.getenv(claude_api_key_env_var)
        if not api_key:
            raise EnvironmentError(
                f"{claude_api_key_env_var} not found. Please set it in your environment or .env file."
            )
        try:
            client = anthropic.Anthropic(api_key=api_key)
            print("Anthropic Claude client initialized successfully.")
            return client  # Returns the client, model is specified per call
        except Exception as e:
            raise RuntimeError(f"Error initializing Anthropic Claude client: {e}") from e

    else:
        raise ValueError(f"Unsupported llm_type: '{llm_type}'. Choose 'gemini' or 'claude'.")


def query_llm_for_causality(llm_client,
                            llm_type,
                            feature_a,
                            feature_b,
                            dataset_domain,
                            claude_model_name=None,  # e.g., 'claude-3-haiku-20240307'
                            max_retries=3,
                            retry_delay_seconds=2,
                            max_tokens_claude=1500):  # Max tokens for Claude response
    """
    Queries the initialized LLM about causality. Retries if a clear verdict isn't found.
    Returns True if causality is asserted ('FINAL VERDICT: YES'), False otherwise.
    Raises exceptions for API errors, blocked content, or if max retries are exhausted.
    """

    prompt = PROMPT_TEMPLATE.format(
        feature_a=feature_a,
        feature_b=feature_b,
        dataset_domain=dataset_domain
    )

    for attempt in range(max_retries):
        print(
            f"--- Querying {llm_type.capitalize()} (Attempt {attempt + 1}/{max_retries}): {feature_a} -> {feature_b} (Domain: {dataset_domain}) ---")

        llm_response_text = ""
        api_call_succeeded = False

        try:
            if llm_type == 'gemini':
                response = llm_client.generate_content(prompt)
                api_call_succeeded = True
                try:
                    llm_response_text = response.text
                except ValueError as ve:  # Blocked content etc.
                    error_message = f"Could not extract text from Gemini response (Attempt {attempt + 1}) for {feature_a} -> {feature_b}. Possibly blocked. Feedback: {getattr(response, 'prompt_feedback', 'N/A')}"
                    print(f"Warning: {error_message}")
                    if attempt == max_retries - 1: raise RuntimeError(error_message) from ve
                    llm_response_text = ""  # Ensure it's empty for retry logic

            elif llm_type == 'claude':
                # System prompts are generally for overall instructions, this prompt structure as a user message is fine.
                response = llm_client.messages.create(
                    model=claude_model_name,
                    max_tokens=max_tokens_claude,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                api_call_succeeded = True
                if response.content and len(response.content) > 0 and hasattr(response.content[0], 'text'):
                    llm_response_text = response.content[0].text
                else:  # Handle cases where Claude response structure is unexpected or empty
                    error_message = f"Unexpected or empty content in Claude response (Attempt {attempt + 1}) for {feature_a} -> {feature_b}. Response: {response}"
                    print(f"Warning: {error_message}")
                    if attempt == max_retries - 1: raise RuntimeError(error_message)
                    llm_response_text = ""

            if api_call_succeeded and llm_response_text:  # Only print if we got something
                print(f"LLM Raw Response (Attempt {attempt + 1}):\n{llm_response_text}\n")

        except anthropic.APIError as e:  # Catch Claude specific API errors
            error_message = f"Claude API Error (Attempt {attempt + 1}) for {feature_a} -> {feature_b}: {e}"
            print(error_message)
            if attempt == max_retries - 1: raise RuntimeError(error_message) from e
        except Exception as e:  # Catch other errors (e.g., Gemini API errors, general issues)
            error_message = f"Error calling {llm_type.capitalize()} API (Attempt {attempt + 1}) for {feature_a} -> {feature_b}: {e}"
            print(error_message)
            if attempt == max_retries - 1: raise RuntimeError(error_message) from e

        if not api_call_succeeded or not llm_response_text:
            if attempt < max_retries - 1:
                print(f"API call failed or response was empty/unreadable. Retrying in {retry_delay_seconds} seconds...")
                time.sleep(retry_delay_seconds)
                continue
            else:  # Last attempt failed to get a response text
                # If it was an API error, it would have been raised already.
                # This means it was likely a blocked Gemini response or empty Claude content on the last try.
                raise RuntimeError(
                    f"API call successful but response was empty/unreadable after {max_retries} attempts for {feature_a} -> {feature_b}.")

        # Case-insensitive verdict strings for comparison
        verdict_yes_str_lower = "final verdict: yes"
        verdict_no_str_lower = "final verdict: no"
        found_verdict_value = None

        lines = llm_response_text.strip().split('\n')
        for line in lines:
            cleaned_line_lower = line.strip().lower()
            if cleaned_line_lower == verdict_yes_str_lower:
                found_verdict_value = True
                break
            elif cleaned_line_lower == verdict_no_str_lower:
                found_verdict_value = False
                break

        if found_verdict_value is not None:
            print(
                f"Verdict for {feature_a} -> {feature_b} (Attempt {attempt + 1}): {'YES' if found_verdict_value else 'NO'}")
            return found_verdict_value

        warning_message = (
            f"Could not find clear 'FINAL VERDICT: YES/NO' in attempt {attempt + 1}/{max_retries} "
            f"for {feature_a} -> {feature_b}."
        )
        print(f"Warning: {warning_message}")
        if attempt < max_retries - 1:
            print(f"Retrying for format in {retry_delay_seconds} seconds...")
            time.sleep(retry_delay_seconds)
        else:
            raise RuntimeError(
                f"Max retries ({max_retries}) reached for {feature_a} -> {feature_b}. "
                "No clear 'FINAL VERDICT' format found in response. "
                f"Last response:\n{llm_response_text}"
            )

    raise RuntimeError(
        f"Query loop completed without verdict or error for {feature_a} -> {feature_b}, which is unexpected if max_retries > 0.")