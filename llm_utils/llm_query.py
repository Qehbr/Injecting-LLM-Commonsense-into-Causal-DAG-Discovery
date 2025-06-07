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
    Initializes and returns a client for a specified Large Language Model.

    This function loads the required API key from environment variables and
    configures the client for either Google Gemini or Anthropic Claude.

    Parameters
    ----------
    llm_type : str, optional
        The type of LLM to initialize. Must be 'gemini' or 'claude'.
        Default is 'gemini'.
    gemini_model_name : str, optional
        The specific Gemini model to use. Default is 'gemini-1.5-flash'.
    gemini_api_key_env_var : str, optional
        The name of the environment variable containing the Gemini API key.
        Default is 'GENAI_API_KEY'.
    claude_api_key_env_var : str, optional
        The name of the environment variable containing the Anthropic API key.
        Default is 'ANTHROPIC_API_KEY'.

    Returns
    -------
    genai.GenerativeModel or anthropic.Anthropic
        An initialized client object for the specified LLM.

    Raises
    ------
    EnvironmentError
        If the required API key environment variable is not set.
    ValueError
        If an unsupported `llm_type` is provided.
    """
    load_dotenv()

    if llm_type == 'gemini':
        api_key = os.getenv(gemini_api_key_env_var)
        if not api_key:
            raise EnvironmentError(
                f"{gemini_api_key_env_var} not found. Please set it in your environment or .env file."
            )
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(gemini_model_name)
        return model
    elif llm_type == 'claude':
        api_key = os.getenv(claude_api_key_env_var)
        if not api_key:
            raise EnvironmentError(
                f"{claude_api_key_env_var} not found. Please set it in your environment or .env file."
            )
        client = anthropic.Anthropic(api_key=api_key)
        return client
    else:
        raise ValueError(f"Unsupported llm_type: '{llm_type}'. Choose 'gemini' or 'claude'.")


def query_llm_for_causality(llm_client,
                            llm_type,
                            feature_a,
                            feature_b,
                            dataset_domain,
                            claude_model_name=None,
                            max_retries=3,
                            retry_delay_seconds=2,
                            max_tokens_claude=1500):
    """
    Queries an LLM to assess the direct causal link from feature_a to feature_b.

    This function formats a detailed prompt, sends it to the specified LLM,
    and parses the response to extract a "YES" or "NO" verdict. It includes
    a retry mechanism to handle transient API issues or malformed responses.

    Parameters
    ----------
    llm_client : object
        The initialized LLM client (from `init_llm`).
    llm_type : str
        The type of LLM being used ('gemini' or 'claude').
    feature_a : str
        The name of the potential causal feature.
    feature_b : str
        The name of the potential effect feature.
    dataset_domain : str
        A brief description of the dataset's domain for context.
    claude_model_name : str, optional
        The specific model name for Claude (e.g., 'claude-3-opus-20240229').
        Required if `llm_type` is 'claude'.
    max_retries : int, optional
        The maximum number of times to retry the query on failure. Default is 3.
    retry_delay_seconds : int, optional
        The number of seconds to wait between retries. Default is 2.
    max_tokens_claude : int, optional
        The maximum number of tokens to generate in the Claude response.
        Default is 1500.

    Returns
    -------
    bool
        True if the LLM's final verdict is 'YES', False if it is 'NO'.

    Raises
    ------
    RuntimeError
        If a clear verdict cannot be obtained after all retry attempts.
    """
    prompt = PROMPT_TEMPLATE.format(
        feature_a=feature_a,
        feature_b=feature_b,
        dataset_domain=dataset_domain
    )

    for attempt in range(max_retries):
        llm_response_text = ""
        api_call_succeeded = False

        if llm_type == 'gemini':
            response = llm_client.generate_content(prompt)
            api_call_succeeded = True
            llm_response_text = response.text  # Will raise ValueError if blocked, per user request to crash
        elif llm_type == 'claude':
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
            else:
                llm_response_text = ""  # Will be caught by logic below

        if not api_call_succeeded or not llm_response_text:
            if attempt < max_retries - 1:
                time.sleep(retry_delay_seconds)
                continue
            else:
                raise RuntimeError(
                    f"API call successful but response was empty/unreadable after {max_retries} attempts for {feature_a} -> {feature_b}.")

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
            return found_verdict_value

        if attempt < max_retries - 1:
            time.sleep(retry_delay_seconds)
        else:
            raise RuntimeError(
                f"Max retries ({max_retries}) reached for {feature_a} -> {feature_b}. "
                "No clear 'FINAL VERDICT' format found in response. "
                f"Last response:\n{llm_response_text}"
            )
    raise RuntimeError(
        f"Query loop completed without verdict or error for {feature_a} -> {feature_b}, which is unexpected if max_retries > 0.")
