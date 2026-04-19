# Multimodal Grounding Evaluations

This project evaluates and compares LLM responses (Anthropic and Google GenAI) for multimodal grounding tasks.

## Project Structure

* `eval.py`: The core experiment script. It iterates through dataset samples, calls the LLM APIs, and saves the raw grounding responses.
* `analysis.ipynb`: A Jupyter Notebook used to process the results, calculate metrics, and generate visualizations (plots/tables).
* `requirements.txt`: List of necessary Python dependencies.
* `.env`: (User-provided) Contains API keys for Anthropic and Google.

## Setup Instructions

### 1. Clone and Environment
First, clone the repository and create a virtual environment:
```bash
git clone <your-repo-url>
cd multimodal-grounding-evals
python3 -m venv .venv
source .venv/bin/activate