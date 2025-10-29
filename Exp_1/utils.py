import random
import string
from collections import namedtuple

import numpy as np

Example = namedtuple('Example', ['question', 'choice1', 'choice2', 'choice3', 'choice4', 'correct_index'])

def generate_multichoice_prompt(n_choices=4):
    assert 1 <= n_choices <= 26, "[--num_multiple] Only supports 1 to 26 choices (A–Z)"
    
    letters = string.ascii_uppercase  # 'A' to 'Z'
    base = "Answer the following multiple choice question.\n\n{Question}\n\n"
    
    options = [f"({letters[i]}) {{{letters[i]}}}" for i in range(n_choices)]
    options_str = "\n".join(options)
    
    return (base + options_str).strip()


def format_multichoice_question(n, row):
    QUERY_TEMPLATE_MULTICHOICE = generate_multichoice_prompt(n)
    return QUERY_TEMPLATE_MULTICHOICE.format(**row)

def generate_or_string(n):
    """
    Generate a string like "A or B or C ... or N"
    
    Parameters:
        n : int
            Number of options (1 ≤ n ≤ 26 typically).
    """
    if n < 1 or n > 26:
        raise ValueError("n must be between 1 and 26")
    
    letters = [chr(ord("A") + i) for i in range(n)]
    return " or ".join(letters)


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