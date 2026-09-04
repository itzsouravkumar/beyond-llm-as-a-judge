from src.metrics import calculate_ecs, compute_correlation_matrix
import numpy as np

def test_calculate_ecs():
    verdicts_agree = [{"verdict": "pass"}, {"verdict": "pass"}, {"verdict": "pass"}]
    assert calculate_ecs(verdicts_agree) == 0.0
    
    verdicts_disagree = [{"verdict": "pass"}, {"verdict": "fail"}, {"verdict": "pass"}]
    # 2 out of 3 agree with majority (pass). ECS = 1 - 2/3 = 1/3
    ecs = calculate_ecs(verdicts_disagree)
    assert abs(ecs - (1 - 2/3)) < 1e-6

def test_compute_correlation():
    # 2 evaluators, identical errors -> correlation 1.0
    error_matrix = np.array([
        [1, 1],
        [0, 0],
        [1, 1],
        [0, 0]
    ])
    corr = compute_correlation_matrix(error_matrix)
    assert abs(corr[0, 1] - 1.0) < 1e-6
