import numpy as np

def calculate_ecs(verdicts):
    """
    Calculates the Evaluator Conflict Score (ECS).
    1 - Agreement(v_1, ..., v_n)
    where Agreement is the fraction of evaluators that agree with the majority.
    """
    if not verdicts:
        return 0.0
    
    outcomes = [v['verdict'] for v in verdicts]
    if len(outcomes) == 0:
        return 0.0

    counts = {}
    for o in outcomes:
        counts[o] = counts.get(o, 0) + 1
    
    max_agreement = max(counts.values())
    agreement_ratio = max_agreement / len(outcomes)
    return 1.0 - agreement_ratio

def compute_correlation_matrix(error_matrix):
    """
    Given a binary error matrix (num_samples x num_evaluators) where 1 indicates an error
    and 0 indicates correct, compute the pairwise Pearson correlation matrix.
    """
    # np.corrcoef expects variables as rows, so we transpose
    return np.corrcoef(error_matrix, rowvar=False)

def calculate_independence_weights(correlation_matrix):
    """
    Simplistic implementation of independence-aware weights.
    w = argmin w^T C w, subject to w >= 0 and sum(w) = 1.
    For demonstration, we use inverse of the sum of absolute correlations per evaluator,
    normalized to sum to 1. In a real scenario, use quadratic programming.
    """
    C = np.array(correlation_matrix)
    # Sum of correlations for each evaluator
    corr_sums = np.sum(np.abs(C), axis=1)
    
    # Invert to penalize highly correlated evaluators
    inv_sums = 1.0 / (corr_sums + 1e-9)
    
    # Normalize
    weights = inv_sums / np.sum(inv_sums)
    return weights
