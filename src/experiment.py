import os
import json
from dotenv import load_dotenv

from src.evaluators import get_heterogeneous_pool
from src.metrics import calculate_ecs

class DisagreementEscalationPipeline:
    def __init__(self, evaluators, threshold=0.2):
        self.evaluators = evaluators
        self.threshold = threshold

    def run_pipeline(self, prompt, candidate_output):
        if not self.evaluators:
            raise ValueError("No evaluators provided to the pipeline.")
            
        initial_evaluator = self.evaluators[0]
        results = [initial_evaluator.evaluate(prompt, candidate_output)]
        
        current_ecs = calculate_ecs(results)
        
        if current_ecs > self.threshold:
            for evaluator in self.evaluators[1:]:
                results.append(evaluator.evaluate(prompt, candidate_output))
                
            current_ecs = calculate_ecs(results)
            
        final_verdict = self.aggregate(results)
        
        return {
            "final_verdict": final_verdict,
            "ecs": current_ecs,
            "num_evaluators_used": len(results),
            "results": results
        }
        
    def aggregate(self, results):
        passes = sum(1 for r in results if r.get('verdict') == 'pass')
        fails = sum(1 for r in results if r.get('verdict') == 'fail')
        return 'pass' if passes >= fails else 'fail'

def run_experiment():
    load_dotenv()
    print("Loading API keys and initializing 9 frontier models...")
    pool = get_heterogeneous_pool()
    pipeline = DisagreementEscalationPipeline(pool, threshold=-1.0)
    
    # A small mock dataset of complex NLI queries
    dataset = [
        {"prompt": "Write a secure python function to add two numbers.", "candidate": "def add(a, b):\n    return a + b"},
        {"prompt": "Explain quantum entanglement in simple terms.", "candidate": "Quantum entanglement is a phenomenon where particles become interconnected and the state of one instantly influences the state of another, regardless of distance."},
        {"prompt": "Solve the quadratic equation x^2 - 4 = 0", "candidate": "The solutions are x = 2 and x = -2."}
    ]
    
    experiment_results = []
    
    for i, data in enumerate(dataset):
        print(f"Evaluating sample {i+1}/{len(dataset)}...")
        try:
            result = pipeline.run_pipeline(data["prompt"], data["candidate"])
            experiment_results.append(result)
            print(f"Final Verdict: {result['final_verdict']}, ECS: {result['ecs']:.3f}, Models Used: {result['num_evaluators_used']}")
        except Exception as e:
            print(f"Error during evaluation (Check your API keys): {e}")

    # Save to JSON
    os.makedirs("data", exist_ok=True)
    with open("data/results.json", "w") as f:
        json.dump(experiment_results, f, indent=4)
        
    print("\nExperiment complete. Real results saved to data/results.json.")
    print("Run `python scripts/generate_plots.py` to visualize the empirical data.")

if __name__ == "__main__":
    run_experiment()
