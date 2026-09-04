# Beyond LLM-as-a-Judge: Overcoming Correlated Errors in Generative AI Evaluation

## Overview
This repository contains the source code, experimental logic, and visualization scripts supporting the research paper on the limitations of the "LLM-as-a-Judge" paradigm. The project demonstrates how correlated errors among frontier language models undermine the statistical independence required for reliable AI evaluation panels, and it proposes adaptive, hybrid verification frameworks as a solution.

## Architecture and Live API Evaluation
This evaluation framework natively integrates with live frontier models via their respective APIs to dynamically aggregate evaluations and compute Effective Sample Sizes based on real-world correlated errors.

To run the experiments, you will need to provide your API keys.

## Repository Structure
- `src/`: Contains the core experimental logic.
  - `evaluators.py`: Base interfaces and implementations for `OpenAIJudge`, `GeminiFreeJudge`, and `OpenRouterJudge`.
  - `experiment.py`: Logic for running trials through the `DisagreementEscalationPipeline` and calculating consensus.
  - `metrics.py`: Statistical functions, including the Kish Effective Sample Size calculation.
- `scripts/`: Automation scripts for visualization.
  - `generate_plots.py`: Generates the data-driven graphs based on empirical evaluation data.
  - `generate_diagrams.py`: Generates the architectural flowcharts.
- `diagrams/`: Output directory for the generated `.png` assets used in the paper.
- `requirements.txt`: Python package dependencies.

## How to Run the Project

1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your API keys. Rename `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
# Edit .env with your favorite editor
```

4. Run the live evaluation experiment:
```bash
python scripts/experiment.py
```
*(Note: Ensure your PYTHONPATH is set if running from the root directory: `PYTHONPATH=. python src/experiment.py`)*

5. Generate the data plots and architectural diagrams from your live results:
```bash
python scripts/generate_plots.py
python scripts/generate_diagrams.py
```
The resulting visualizations will be saved to the `diagrams/` directory. If you run the plotting scripts without running the experiment first, they will automatically default to rendering the theoretical baseline values established in the paper.

## References
This project and its implementations build upon the following verified research:

1. Zheng, L., Chiang, W.-L., Sheng, Y., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS. https://arxiv.org/abs/2306.05685
2. Boodhun, B., et al. (2024). JudgeBench: A Benchmark for Evaluating LLM-based Judges. https://github.com/ScalerLab/JudgeBench

## Contact
For inquiries or discussions regarding this research, please contact Sourav Kumar at sk9453@srmist.edu.in.
