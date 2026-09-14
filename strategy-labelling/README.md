# Counseling strategy classifier

This optional analysis labels individual sentences with probabilities for 15 counseling strategies. It is separate from pairwise judging and Bradley–Terry ranking.

## Run one example

From the repository root:

```bash
python -m pip install -r strategy-labelling/requirements.txt
cd strategy-labelling
# Set OPENAI_API_KEY in your environment before running the classifier.
python -c "import pandas as pd; pd.DataFrame({'response': ['It makes sense that you feel overwhelmed.']}).to_excel('example.xlsx', index=False)"
python label_strategies_llm.py --in example.xlsx --out example_labels.xlsx --start-col A --end-col A --strategies strategy_metadata.json --model gpt-4o-mini --threshold 0.8 --batch-size 1 --max-concurrent 1
```

This sends the example sentence and strategy definitions to OpenAI and incurs API usage. The script reads Excel input even when `--format csv` is selected; that option controls output only. For your own workbook, set the inclusive start and end column letters to the response columns. The default range is D–O. Use `--sheet` to select a named sheet; omit it for the first sheet.

## Prompts and outputs

`SYSTEM_PROMPT` and `USER_PROMPT_TEMPLATE` in the script contain the exact prompts. `strategy_metadata.json` contains the strategy definitions supplied to the model. The requested JSON structure is:

```json
{"probs": {"Affirmation": 0.2, "Clarification": 0.8}, "labels_above_threshold": ["Clarification"]}
```

The example above is illustrative; the prompt requests a probability for every strategy. These are model-estimated probabilities, not validated confidence estimates. Calls use temperature 0.0. The default threshold is 0.8 and the default model is `gpt-4o-mini`; these defaults do not establish the settings used for the manuscript.

Excel output includes input data, sentence probabilities, sentence labels, cell-level unions of sentence labels, and a run summary. `--format csv` or `--format both` also exports CSV files. Checkpoints support resume; use a distinct checkpoint or `--no-resume` when changing inputs or settings.

## Provenance and limitations

Imported from `kriti-hippo/benchproj`, commit `dc0b65092d22fb1aeda13f8c99145294ac194b99`: `strategy-labelling/old/label_strategies_llm_clean.py` and `strategy-labelling/strategy_metadata.json`. Only the script filename and its usage example were changed. The source repository is private; the runnable source and definitions are included here. Historical completion data and classification outputs are not included.

This is the recovered classifier implementation. Its correspondence to the final published analysis has not been independently verified. The upstream parser can replace malformed responses or exhausted API failures with all-zero probabilities, and it can retain model-supplied threshold labels that disagree with probabilities. Inspect errors and outputs before analysis; zero scores alone do not distinguish absence from failure. This implementation requests JSON but does not enforce a formal JSON Schema.

Validation covers a synthetic one-sentence input with a mocked API response, Excel/CSV export, and CLI help. A fresh live classification has not been run.
