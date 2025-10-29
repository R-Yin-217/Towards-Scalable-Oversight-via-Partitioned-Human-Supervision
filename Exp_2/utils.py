import random
import string
from collections import namedtuple

import numpy as np

Example = namedtuple('Example', ['question', 'choice1', 'choice2', 'choice3', 'choice4', 'correct_index'])

def generate_multichoice_prompt(options, dataset_name):
    """
    Generate an industry classification prompt in a clean format.
    
    Parameters:
        options : list of str
            List of industry categories in Japanese.
    """
    # turn list into bullet points
    if isinstance(options, list):
        options_str = "\n".join([f"- {opt}" for opt in options])
    else:
        options_str = str(options)
    if(dataset_name == 'Dataset/EDINET_Bench_16.csv'or dataset_name == 'Dataset/EDINET_Bench_Extended_16.csv'):
        base = f"""
        Based on the following financial report, classify the company into one of the Japanese industry categories.
        You must classify the company into exactly one of these categories:

        {options_str}

        Notes:
        - The input is extracted from a Japanese company's securities report.
        - Some information may be missing due to parsing errors.
        - Answer with exactly one category name (no explanations, no extra text).

        The current year's extracted securities report is as follows:

        {{Input}}
        """
    elif(dataset_name == 'Dataset/Medical_Abstract_5.csv'):
        base = f"""
        Based on the following medical research abstract, classify the paper into one of the medical research categories.
        You must classify the abstract into exactly one of these categories:

        {options_str}

        Notes:
        - The input is extracted from a medical research paper abstract.
        - Some information may be missing or noisy due to parsing errors.
        - Answer with exactly one category name (no explanations, no extra text).

        The research abstract is as follows:

        {{Abstract}}
        """
        
    else:
        raise ValueError("You got the wrong Dataset for utils!")
    return base.strip()


def format_multichoice_question(options, row, dataset_name):
    QUERY_TEMPLATE_MULTICHOICE = generate_multichoice_prompt(options, dataset_name)
    return QUERY_TEMPLATE_MULTICHOICE.format(**row)


def random_id(length=4):
    characters = string.ascii_letters + string.digits  # includes both upper/lower case letters and numbers
    random_id = ''.join(random.choices(characters, k=length))
    return random_id


def bootstrap_confidence_interval(
    data,
    num_bootstrap_samples=10000,
    confidence_level=0.95,
    seed=None
):
    """
    Calculate accuracy, variance, and bootstrap confidence interval 
    for the mean of 1D accuracy data.

    Args:
        data (list or array of float): 1D list or array of data points (0/1 or transformed scores).
        num_bootstrap_samples (int): Number of bootstrap samples.
        confidence_level (float): The desired confidence level (e.g., 0.95 for 95%).
        seed (int or None): Random seed for reproducibility.

    Returns:
        dict: {
            "accuracy": float,           # mean of data
            "variance": float,           # unbiased sample variance / n
            "ci_lower": float,           # lower bound of CI
            "ci_upper": float,           # upper bound of CI
            "median": float              # median of bootstrap means
        }
    """
    rng = np.random.default_rng(seed)
    data = np.array(data)

    # acc and var
    accuracy = np.mean(data)
    variance = np.var(data, ddof=1) / len(data) if len(data) > 1 else 0.0

    # bootstrap
    bootstrap_means = [
        np.mean(rng.choice(data, size=len(data), replace=True))
        for _ in range(num_bootstrap_samples)
    ]
    bootstrap_means = np.array(bootstrap_means)

    lower_percentile = (1.0 - confidence_level) / 2.0
    upper_percentile = 1.0 - lower_percentile
    ci_lower = np.percentile(bootstrap_means, lower_percentile * 100)
    ci_upper = np.percentile(bootstrap_means, upper_percentile * 100)
    median = np.median(bootstrap_means)

    return {
        "accuracy": float(accuracy),
        "variance": float(variance),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "median": float(median)
    }