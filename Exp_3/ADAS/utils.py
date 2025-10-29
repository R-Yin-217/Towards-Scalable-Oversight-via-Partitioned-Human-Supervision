import random
import string
from collections import namedtuple

import numpy as np

Example = namedtuple('Example', ['question', 'choice1', 'choice2', 'choice3', 'choice4', 'correct_index'])

def generate_multichoice_prompt(options, n_choices=4, dataset_name='Dataset/GPQA_Extended_4.csv'):
    if(dataset_name in ['Dataset/GPQA_Extended_4.csv', 'Dataset/MATH_MC_5.csv']):
        assert 1 <= n_choices <= 26, "[--num_multiple] Only supports 1 to 26 choices (A–Z)"

        letters = string.ascii_uppercase  # 'A' to 'Z'
        base = "Answer the following multiple choice question.\n\n{Question}\n\n"

        options = [f"({letters[i]}) {{{letters[i]}}}" for i in range(n_choices)]
        options_str = "\n".join(options)

        return (base + options_str).strip()
    
    elif(dataset_name == 'Dataset/Medical_Abstract_5.csv'):
        if isinstance(options, list):
            options_str = "\n".join([f"- {opt}" for opt in options])
        else:
            options_str = str(options)
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
        return base.strip()
    else:
        raise ValueError("You got the wrong Dataset for utils!")

def format_multichoice_question(options, n, row, dataset_name):
    QUERY_TEMPLATE_MULTICHOICE = generate_multichoice_prompt(options, n, dataset_name)
    return QUERY_TEMPLATE_MULTICHOICE.format(**row)


def random_id(length=4):
    characters = string.ascii_letters + string.digits  # includes both upper/lower case letters and numbers
    random_id = ''.join(random.choices(characters, k=length))
    return random_id


def bootstrap_confidence_interval(data, num_bootstrap_samples=100000, confidence_level=0.95, seed=None, CL=False, K=4):
    """
    Calculate the bootstrap confidence interval for the mean of 1D accuracy data.
    Also returns the median of the bootstrap means.
    
    Args:
    - data (list or array of float): 1D list or array of data points.
    - num_bootstrap_samples (int): Number of bootstrap samples.
    - confidence_level (float): The desired confidence level (e.g., 0.95 for 95%).
    
    Returns:
    - str: Formatted string with 95% confidence interval and median as percentages with one decimal place.
    """
    rng = np.random.default_rng(seed)
    data = np.array(data)
    
    # Convert data to a numpy array for easier manipulation
    data = np.array(data)

    # List to store the means of bootstrap samples
    bootstrap_means = []

    # Generate bootstrap samples and compute the mean for each sample
    for _ in range(num_bootstrap_samples):
        # Resample with replacement
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        # Compute the mean of the bootstrap sample
        bootstrap_mean = np.mean(bootstrap_sample)
        bootstrap_means.append(bootstrap_mean)

    # Convert bootstrap_means to a numpy array for percentile calculation
    bootstrap_means = np.array(bootstrap_means)

    # Compute the lower and upper percentiles for the confidence interval
    lower_percentile = (1.0 - confidence_level) / 2.0
    upper_percentile = 1.0 - lower_percentile
    ci_lower = np.percentile(bootstrap_means, lower_percentile * 100)
    ci_upper = np.percentile(bootstrap_means, upper_percentile * 100)

    # Compute the median of the bootstrap means
    median = np.median(bootstrap_means)
    
    if(CL):
        ci_lower = (K-1) * ci_lower - (K-2)
        ci_upper = (K-1) * ci_upper - (K-2)
        median = (K-1) * median - (K-2)
    
    # Convert to percentages and format to one decimal place
    ci_lower_percent = ci_lower * 100
    ci_upper_percent = ci_upper * 100
    median_percent = median * 100
        

    # Return the formatted string with confidence interval and median
    return f"95% Bootstrap Confidence Interval: ({ci_lower_percent:.1f}%, {ci_upper_percent:.1f}%), Median: {median_percent:.1f}%"


