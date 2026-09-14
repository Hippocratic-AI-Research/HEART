# Model Evaluation Pipeline

Evaluate models on emotional support dialogues with two scripts:

1. `add_new_model.py` generates completions, creates comparisons against existing model columns, and runs the three judges.
2. `bradley-terry.py` computes rankings from the resulting pairwise evaluations.

There is no separate `run_all_pairwise_evals.py` command. Pairwise evaluation runs by default in `add_new_model.py`; use `--skip-pairwise` for completions only.

## Setup

Use Python 3.10 or later. Run commands from the repository root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For pairwise evaluation, configure all three judge credentials in your environment:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
```

The judges configured in the script are `gpt-4o`, `claude-sonnet-4-20250514`, and `gemini-2.5-flash`. Your accounts must have access to these models. The candidate model's `--api-key` does not replace these judge environment variables. API calls incur provider charges.

## Step 1: Generate completions and pairwise evaluations

```bash
python add_new_model.py \
  --model-name "your-model-name" \
  --model-name-save "candidate_regular" \
  --model-type "openai" \
  --base-file "data/dialogues_regular.json" \
  --output-file "all_model_completions_regular.json"
```

The input must be a nonempty JSON array of dialogue objects with `dialogue_history` and existing model completion columns. The supplied dialogue files include `vanilla_completion` as a comparison baseline. To compare against more models, pass a completions file containing their responses for the same dialogue rows as `--base-file`.

The command writes:

- `all_model_completions_regular.json`: input rows with the candidate response under `candidate_regular`.
- `candidate_regular-pairwise/`: one file per opponent and judge, named `candidate_regular-vs-<opponent>-<judge>.json`. These files contain **JSON Lines**, despite their `.json` extension.

Only the candidate is compared against existing model columns; this does not recompute every pair among existing models. Human completion fields are excluded. Keep unrelated metadata out of model columns because opponent detection uses the first row's keys.

For adversarial dialogues, use `data/dialogues_adversarial.json`, a distinct save name such as `candidate_adversarial`, and a distinct output file. Pairwise folders are derived from `--model-name-save` in the current working directory, so reusing the same save name overwrites generated pair files.

### Model types and options

| Option | Behavior |
| --- | --- |
| `--model-type openai` | Candidate key from `OPENAI_API_KEY` or `--api-key` |
| `--model-type claude` | Candidate key from `ANTHROPIC_API_KEY` or `--api-key` |
| `--model-type gemini` | Candidate key from `GOOGLE_API_KEY` or `--api-key` |
| `--model-type api` | Custom endpoint; requires `--api-url` and `--api-key` |
| `--model-name-save NAME` | Response column and output folder prefix; use a filename-safe name without slashes |
| `--parallel-workers N` | Completion workers / concurrent judge files (default: 6) |
| `--batch-size N` | Entries per batch (default: 20) |
| `--skip-pairwise` | Generate completions without creating pairs or calling judges |
| `--skip-completions` | Read existing candidate responses from `--base-file`, then generate and judge pairs |
| `--eval-only` | Resume judging existing pair files without regenerating them |

To create pairs from previously generated completions, rerun the Step 1 command with `--skip-completions` and set `--base-file` to the filled completions file. Keep the same candidate identifier. This regenerates pair files; use `--eval-only` to resume an interrupted judge run instead:

```bash
python add_new_model.py \
  --eval-only \
  --model-name-save "candidate_regular" \
  --parallel-workers 6 \
  --batch-size 20
```

Resume mode skips records with an existing `overall_eq` result and no evaluation error. Inspect the generated records for errors or missing evaluations before ranking; a completed process does not guarantee every provider request succeeded.

## Step 2: Compute Bradley-Terry rankings

```bash
python bradley-terry.py \
  --folder "candidate_regular-pairwise" \
  --no-load \
  --no-save \
  --reset \
  --leaderboard-output "leaderboard_regular.json"
```

`--no-load` and `--reset` ignore previously saved state; `--no-save` prevents writing state. `--leaderboard-output` is optional and exports JSON for a leaderboard UI. Rank the folder produced by Step 1, or a directory containing the intended collection of pairwise results. Rankings depend on which comparisons are included.

Additional options:

- `--pattern`: file pattern (default: `**/*.json*`).
- `--metadata`: show emotion/problem type breakdowns.
- `--no-analysis`: omit detailed analysis output.
- `--random-seed`: set evaluation ordering seed.

Use `python add_new_model.py --help` and `python bradley-terry.py --help` for all supported arguments.

If you would like the completions generated by each of the models that are on our leaderboard, please reach out to the corresponding author.
