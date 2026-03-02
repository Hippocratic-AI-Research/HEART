# Model Evaluation Pipeline

This repository contains scripts for evaluating new models on emotional support dialogue tasks. The evaluation pipeline consists of three main steps:

## Setup

### 1. Create a Virtual Environment

First, create a new virtual environment in the project directory:

```bash
python3 -m venv venv
```

### 2. Activate the Virtual Environment

**On macOS/Linux:**

```bash
source venv/bin/activate
```

After activation, you should see `(venv)` in your terminal prompt.

### 3. Install Dependencies

Install all required packages from `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Important Notes

- **Always activate the virtual environment** before running any scripts
- Use `python` (not `python3`) when the virtual environment is activated
- The virtual environment isolates project dependencies from your system Python
- If you encounter import errors, make sure the venv is activated and dependencies are installed

## Overview

1. **Generate Completions**: Use `add_new_model.py` to generate model completions for dialogue datasets
2. **Pairwise Evaluations**: Run `run_all_pairwise_evals.py` to generate pairwise comparisons between models
3. **Bradley-Terry Ranking**: Use `bradley-terry.py` to compute model rankings from pairwise evaluations

---

## Step 1: Generate Model Completions

Run `add_new_model.py` to generate completions for your new model on both the regular and adversarial dialogue datasets.

**Note:** Make sure your virtual environment is activated before running these commands (see Setup section above).

### For Regular Dialogues

```bash
source venv/bin/activate  # Activate venv first
python add_new_model.py \
  --model-name "your-model-name" \
  --model-type "openai" \
  --api-key "your-api-key" \
  --base-file "data/dialogues_regular.json" \
  --output-file "all_model_completions_regular.json"
```

### For Adversarial Dialogues

```bash
source venv/bin/activate  # Activate venv first
python add_new_model.py \
  --model-name "your-model-name" \
  --model-type "openai" \
  --api-key "your-api-key" \
  --base-file "data/dialogues_adversarial.json" \
  --output-file "all_model_completions_adversarial.json"
```

### Model Types

The `--model-type` argument supports:

- `openai`: For OpenAI models (requires `OPENAI_API_KEY` if `--api-key` not provided)
- `claude`: For Anthropic Claude models (requires `ANTHROPIC_API_KEY` if `--api-key` not provided)
- `gemini`: For Google Gemini models (requires `GOOGLE_API_KEY` if `--api-key` not provided)
- `api`: For custom API endpoints (requires `--api-url` and optionally `--api-key`)

### Additional Options

- `--model-name-save`: Name to save files as (if different from `--model-name`)
- `--parallel-workers N`: Number of parallel workers (default: 6)
- `--batch-size N`: Batch size for processing (default: 20)
- `--skip-completions`: Skip completion generation if already done

**Note**: The script will generate a JSON file containing all dialogue entries with the new model's completions added.

---

## Step 2: Run Pairwise Evaluations

After generating completions, run pairwise evaluations to compare all models:

```bash
source venv/bin/activate  # Activate venv first
python run_all_pairwise_evals.py \
  --completions-file "all_model_completions_regular.json" \
  --output-folder "pairwise-evals" \
  --batch-size 50
```

### Arguments

- `--completions-file`: Path to the completions JSON file generated in Step 1
- `--output-folder`: Folder where pairwise evaluation files will be written (default: `pairwise-evals`)
- `--batch-size`: Initial batch size for API calls (default: 50)
- `--models`: Optional list of specific models to evaluate (default: evaluates all models)

**Note**: This script creates pairwise comparison files for all model pairs, generating evaluations using Claude, OpenAI (o1), and Gemini evaluators. The process may take significant time depending on the number of models and dialogues.

---

## Step 3: Bradley-Terry Ranking

Finally, run the Bradley-Terry ranking algorithm on the pairwise evaluation results:

```bash
source venv/bin/activate  # Activate venv first
python bradley-terry.py \
  --folder "pairwise-evals" \
  --no-load \
  --no-save \
  --reset
```

### Arguments

- `--folder`: Folder containing the pairwise evaluation JSONL files (output from Step 2)
- `--no-load`: Don't load existing Bradley-Terry state (start fresh)
- `--no-save`: Don't save Bradley-Terry state after processing

### Additional Options

- `--pattern`: File pattern to match (default: `**/*.json*`)
- `--leaderboard-output PATH`: Optional path to save leaderboard JSON for UI
- `--metadata`: Show metadata analysis (emotion/problem type breakdown)
- `--no-analysis`: Skip detailed analysis output

**Note**: The `--no-load` and `--no-save` flags ensure that the evaluation starts from scratch and doesn't save intermediate state, which is recommended for independent runs.

---

## Complete Example Workflow

```bash
# Activate virtual environment first
source venv/bin/activate

# Step 1: Generate completions for regular dialogues
python add_new_model.py \
  --model-name "gpt-4-turbo" \
  --model-type "openai" \
  --base-file "data/dialogues_regular.json" \
  --output-file "all_model_completions_regular.json"

# Step 2: Generate pairwise evaluations
python run_all_pairwise_evals.py \
  --completions-file "all_model_completions_regular.json" \
  --output-folder "pairwise-evals"

# Step 3: Compute Bradley-Terry rankings
python bradley-terry.py \
  --folder "pairwise-evals" \
  --no-load \
  --no-save
```

---

## Output Files

- **Step 1**: `all_model_completions_regular.json` or `all_model_completions_adversarial.json` - Contains dialogue data with model completions
- **Step 2**: `pairwise-evals/` folder - Contains JSONL files with pairwise evaluations for each model pair
- **Step 3**: Console output showing model rankings and optionally a leaderboard JSON file (if `--leaderboard-output` is specified)

---

## Requirements

All required packages are listed in `requirements.txt`. After setting up the virtual environment and installing dependencies (see Setup section above), make sure you have API keys configured:

- **OpenAI**: Set `OPENAI_API_KEY` environment variable or pass via `--api-key` argument
- **Anthropic**: Set `ANTHROPIC_API_KEY` environment variable or pass via `--api-key` argument
- **Google Gemini**: Set `GOOGLE_API_KEY` environment variable or pass via `--api-key` argument
- **Custom API**: Pass both `--api-url` and optionally `--api-key` arguments

---

If you would like the completions generated by each of the models that are on our leaderboard, please reach out to the corresponding author.
