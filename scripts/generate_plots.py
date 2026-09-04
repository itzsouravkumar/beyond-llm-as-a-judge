import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Make sure diagrams directory exists
os.makedirs("diagrams", exist_ok=True)

def generate_kish_plot():
    # By default, use the predicted theoretical values from the 2026 paper's findings.
    # If the user has run experiment.py with their own API keys, we will attempt to 
    # incorporate the real empirical effective sample sizes.
    k = [1, 3, 5, 7, 9]
    n_eff = [1.00, 1.65, 2.05, 2.15, 2.18]
    
    # Try to load live data if experiment was run
    if os.path.exists("data/results.json"):
        try:
            with open("data/results.json", "r") as f:
                results = json.load(f)
            if results:
                # Calculate average ECS from the live API responses
                avg_ecs = sum(r['ecs'] for r in results) / len(results)
                
                # Replace the final data point (k=9) with the true live data average if the user used all 9 models
                if len(results[0].get('results', [])) >= 9:
                    n_eff[-1] = avg_ecs
                else:
                    # Dynamically adjust the plot based on how many models actually ran
                    max_models = len(results[0].get('results', []))
                    if max_models in k:
                        idx = k.index(max_models)
                        n_eff[idx] = avg_ecs
        except Exception as e:
            print(f"Could not parse live empirical data: {e}. Falling back to predicted values.")
    
    plt.figure(figsize=(8, 5))
    plt.plot(k, n_eff, marker='o', linewidth=2, markersize=8, color='#2c7bb6')
    plt.axhline(y=2.6, color='r', linestyle='--', alpha=0.7, label='Theoretical Max (\\sim 2.6)')
    
    plt.xlabel('Panel Size ($k$)', fontsize=12)
    plt.ylabel('Effective Votes ($n_{eff}$)', fontsize=12)
    plt.title('Effective Sample Size vs Panel Size (MNLI, $\\bar{\\phi} = 0.391$)', fontsize=14)
    plt.xticks(k)
    plt.ylim(0, 3)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig("diagrams/effective_sample_size.png", dpi=300)
    plt.close()
    print("Generated effective_sample_size.png")

def generate_judgebench_plot():
    # Data from Table 3: JudgeBench accuracy (theoretical baseline)
    categories = ["Knowledge", "Reasoning", "Mathematics", "Coding", "Overall"]
    
    data = {
        "Random Guessing": [50.00, 50.00, 50.00, 50.00, 50.00],
        "PandaLM": [21.05, 15.31, 2.27, 11.90, 13.14],
        "JudgeLM-7B": [42.11, 28.57, 9.09, 23.81, 25.14],
        "VertexAI": [48.68, 52.04, 40.91, 28.57, 44.57],
        "Vanilla GPT-4o": [43.42, 48.98, 61.36, 53.57, 50.86],
        "Arena-Hard Judge": [50.65, 54.08, 75.00, 59.52, 56.57]
    }
    
    x = np.arange(len(categories))
    width = 0.12
    multiplier = 0
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#cccccc', '#d7191c', '#fdae61', '#abd9e9', '#2c7bb6', '#053061']
    
    for i, (attribute, measurement) in enumerate(data.items()):
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute, color=colors[i])
        multiplier += 1
        
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('JudgeBench Accuracy by Evaluator and Category', fontsize=14)
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.axhline(y=50, color='gray', linestyle=':', alpha=0.8)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig("diagrams/judgebench_accuracy.png", dpi=300)
    plt.close()
    print("Generated judgebench_accuracy.png")

if __name__ == "__main__":
    generate_kish_plot()
    generate_judgebench_plot()
