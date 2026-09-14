#!/usr/bin/env python3
"""
LLM-based, sentence-level multi-label classification of counseling strategies (columns D–O).

Enhancements:
- FAST async batch processing with parallelization for major speed improvements.
- Fixed freezing issues with proper timeout handling, rate limiting, and progress tracking.
- Automatic checkpointing and resume functionality.
- Supports --format {xlsx,csv,both}. CSV writes multiple files (one per "sheet").

Usage example (fast async mode):
  python label_strategies_llm.py \
    --in "Completions Data.xlsx" \
    --out "completions_llm_labels.xlsx" \
    --format xlsx \
    --threshold 0.8 \
    --model "gpt-4o-mini" \
    --batch-size 10 \
    --max-concurrent 5 \
    --sheet "Sheet1" \
    --start-col "D" --end-col "O"
"""

from __future__ import annotations
import os, re, json, time, argparse, sys
import asyncio
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
from pathlib import Path

# OpenAI client
try:
    from openai import AsyncOpenAI, RateLimitError, APITimeoutError, APIError
except Exception as e:
    print("Please 'pip install openai' (>=1.0).", file=sys.stderr)
    raise

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs): return x

def validate_api_key():
    """Validate OpenAI API key is available"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set!", file=sys.stderr)
        print("Please set your OpenAI API key:", file=sys.stderr)
        print("  export OPENAI_API_KEY='your-api-key-here'", file=sys.stderr)
        sys.exit(1)
    return api_key

def col_letter_to_index(letter: str) -> int:
    letter = letter.upper().strip()
    result = 0
    for ch in letter:
        if not ('A' <= ch <= 'Z'):
            raise ValueError(f"Invalid column letter: {letter}")
        result = result * 26 + (ord(ch) - ord('A') + 1)
    return result - 1

def split_sentences(text: str) -> List[str]:
    if not isinstance(text, str):
        text = str(text if text is not None else "")
    s = text.strip()
    if not s:
        return []
    s = re.sub(r'[\r\n]+', '\n', s)
    parts = []
    for block in s.split("\n"):
        block = block.strip()
        if not block:
            continue
        chunks = re.split(r'(?<=[\.\?\!])\s+(?=[A-Z0-9\"\'])', block)
        for c in chunks:
            c = c.strip()
            if c:
                parts.append(c)
    return parts

SYSTEM_PROMPT = """\
You are a precise annotation model. Your task: for a SINGLE given sentence,
estimate the probability (0.0–1.0) that EACH counseling strategy is being used
in the sentence. Decide ONLY from the sentence itself (do not infer what a good
reply should do). Strategies are defined by their descriptions below.

IMPORTANT: You MUST respond with ONLY valid JSON. No additional text before or after.

Return strict JSON with keys:
- "probs": object mapping strategy name -> float in [0,1].
- "labels_above_threshold": array of strategy names that are >= the given threshold.

Use calibrated probabilities reflecting how clearly the sentence matches the strategy.
If no strategies are present, use low probabilities (e.g., <= 0.1).

Example format:
{"probs": {"Affirmation": 0.2, "Clarification": 0.8}, "labels_above_threshold": ["Clarification"]}
"""

USER_PROMPT_TEMPLATE = """\
Threshold: {threshold}

Sentence:
{sentence}

Strategies (name: description):
{strategies_block}

Respond with JSON only.
"""

def make_strategies_block(strat_map: Dict[str, str]) -> str:
    return "\n".join([f"- {name}: {desc}" for name, desc in strat_map.items()])

class AsyncRateLimiter:
    """Async rate limiter to prevent API throttling"""
    def __init__(self, max_requests_per_minute: int = 60):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.lock = asyncio.Lock()
    
    async def wait_if_needed(self):
        async with self.lock:
            now = time.time()
            # Remove requests older than 1 minute
            self.requests = [t for t in self.requests if now - t < 60]
            
            if len(self.requests) >= self.max_requests:
                # Wait until the oldest request is more than 1 minute old
                sleep_time = 60 - (now - self.requests[0]) + 0.1
                if sleep_time > 0:
                    print(f"Rate limit reached, waiting {sleep_time:.1f} seconds...")
                    await asyncio.sleep(sleep_time)
                    # Refresh the list after waiting
                    now = time.time()
                    self.requests = [t for t in self.requests if now - t < 60]
            
            self.requests.append(now)

@dataclass
class AsyncBatchModelCaller:
    """Async batched version of ModelCaller for better performance"""
    model: str
    threshold: float
    strategies: Dict[str, str]
    rate_limiter: Optional[AsyncRateLimiter] = None
    max_retries: int = 5
    timeout: int = 30
    batch_size: int = 10
    max_concurrent: int = 5

    def __post_init__(self):
        self.client = AsyncOpenAI(timeout=self.timeout)
        if self.rate_limiter is None:
            # Adjust rate limit for concurrent requests
            adjusted_rate = max(10, 60 // self.max_concurrent)  
            self.rate_limiter = AsyncRateLimiter(adjusted_rate)
        
        # Track API usage
        self.total_requests = 0
        self.failed_requests = 0
        self._lock = asyncio.Lock()

    async def _update_stats(self, failed: bool = False):
        """Thread-safe statistics update"""
        async with self._lock:
            self.total_requests += 1
            if failed:
                self.failed_requests += 1

    async def score_sentence_async(self, sentence: str) -> Dict[str, Any]:
        """Score a single sentence asynchronously"""
        await self._update_stats()
        
        for attempt in range(self.max_retries + 1):
            try:
                # Apply rate limiting
                await self.rate_limiter.wait_if_needed()
                
                user_prompt = USER_PROMPT_TEMPLATE.format(
                    threshold=self.threshold,
                    sentence=sentence.strip(),
                    strategies_block=make_strategies_block(self.strategies),
                )
                
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content

                # Parse response
                data = None
                if content:
                    try:
                        data = json.loads(content.strip())
                    except json.JSONDecodeError:
                        # Try to extract JSON from response
                        json_patterns = [
                            r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",  # Simple nested JSON
                            r"\{.*?\}",  # Basic JSON block
                        ]
                        for pattern in json_patterns:
                            matches = re.findall(pattern, content, re.S)
                            for match in matches:
                                try:
                                    data = json.loads(match)
                                    break
                                except json.JSONDecodeError:
                                    continue
                            if data:
                                break
                        
                        if not data:
                            # Last resort: create empty response structure
                            data = {"probs": {strat: 0.0 for strat in self.strategies.keys()}, "labels_above_threshold": []}
                
                if not data:
                    raise ValueError(f"Could not parse any valid JSON from response: {content}")

                # Validate and fix response structure
                probs = data.get("probs", {})
                if not isinstance(probs, dict):
                    probs = {}
                
                # Ensure all strategies have probability values
                for strat in self.strategies.keys():
                    if strat not in probs:
                        probs[strat] = 0.0
                    elif not isinstance(probs[strat], (int, float)):
                        probs[strat] = 0.0
                
                data["probs"] = probs
                labels = [k for k, v in probs.items() if isinstance(v, (int, float)) and v >= self.threshold]
                data["labels_above_threshold"] = data.get("labels_above_threshold", labels)
                
                return data
                
            except (RateLimitError, APITimeoutError) as e:
                wait_time = min(2 ** attempt, 60)  # Exponential backoff, max 60 seconds
                print(f"API error on attempt {attempt + 1}/{self.max_retries + 1}: {e}")
                print(f"Waiting {wait_time} seconds before retry...")
                await asyncio.sleep(wait_time)
                continue
                
            except APIError as e:
                if "rate limit" in str(e).lower():
                    wait_time = min(2 ** attempt, 60)
                    print(f"Rate limit error on attempt {attempt + 1}/{self.max_retries + 1}: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    print(f"API Error: {e}")
                    break
                    
            except Exception as e:
                print(f"Unexpected error on attempt {attempt + 1}/{self.max_retries + 1}: {e}")
                if attempt == self.max_retries:
                    break
                await asyncio.sleep(1)
                continue
        
        # If all retries failed, return empty result
        await self._update_stats(failed=True)
        print(f"Failed to score sentence after {self.max_retries + 1} attempts: '{sentence[:100]}...'")
        return {"probs": {strat: 0.0 for strat in self.strategies.keys()}, "labels_above_threshold": []}

    async def score_batch_async(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """Score a batch of sentences concurrently"""
        # Limit concurrent requests to prevent overwhelming the API
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def score_with_semaphore(sentence):
            async with semaphore:
                return await self.score_sentence_async(sentence)
        
        # Process all sentences in batch concurrently
        tasks = [score_with_semaphore(sentence) for sentence in sentences]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Error processing sentence {i}: {result}")
                processed_results.append({
                    "probs": {strat: 0.0 for strat in self.strategies.keys()}, 
                    "labels_above_threshold": []
                })
            else:
                processed_results.append(result)
        
        return processed_results

def save_checkpoint(checkpoint_path: str, rows_sent: List[Dict], progress_info: Dict):
    """Save progress checkpoint to resume if interrupted"""
    checkpoint_data = {
        "rows_sent": rows_sent,
        "progress_info": progress_info,
        "timestamp": time.time()
    }
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint_data, f, indent=2)

def load_checkpoint(checkpoint_path: str) -> Tuple[List[Dict], Dict]:
    """Load progress checkpoint"""
    try:
        with open(checkpoint_path, "r") as f:
            data = json.load(f)
        return data["rows_sent"], data["progress_info"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return [], {}

async def annotate_file_async(
    in_path: str,
    sheet: str,
    start_col_letter: str,
    end_col_letter: str,
    threshold: float,
    model: str,
    strategies_path: str,
    checkpoint_path: Optional[str] = None,
    resume: bool = True,
    batch_size: int = 10,
    max_concurrent: int = 5,
):
    """Async version of annotate_file with batching and parallelization"""
    df = pd.read_excel(in_path, sheet_name=sheet)
    start_idx = col_letter_to_index(start_col_letter)
    end_idx = col_letter_to_index(end_col_letter) + 1
    target_cols = list(df.columns[start_idx:end_idx])

    with open(strategies_path, "r") as f:
        strategies = json.load(f)

    # Initialize async batch caller
    rate_limiter = AsyncRateLimiter(max_requests_per_minute=60)
    caller = AsyncBatchModelCaller(
        model=model, 
        threshold=threshold, 
        strategies=strategies, 
        rate_limiter=rate_limiter,
        batch_size=batch_size,
        max_concurrent=max_concurrent
    )

    # Set up checkpointing
    if checkpoint_path is None:
        checkpoint_path = f"checkpoint_{Path(in_path).stem}.json"
    
    rows_sent = []
    processed_items = set()
    
    # Try to resume from checkpoint
    if resume and os.path.exists(checkpoint_path):
        print(f"Found checkpoint file: {checkpoint_path}")
        try:
            rows_sent, progress_info = load_checkpoint(checkpoint_path)
            processed_items = set((r["row_index"], r["column"], r["sentence_index"]) for r in rows_sent)
            print(f"Resuming from checkpoint: {len(rows_sent)} sentences already processed")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            rows_sent = []
            processed_items = set()

    # Collect all sentences to process
    sentence_items = []
    for ridx, row in df.iterrows():
        for col in target_cols:
            cell = row[col] if col in df.columns else None
            if pd.isna(cell) or str(cell).strip() == "":
                continue
            sents = split_sentences(str(cell))
            for sidx, sent in enumerate(sents):
                item_key = (ridx, col, sidx)
                if item_key not in processed_items:
                    sentence_items.append((ridx, col, sidx, sent))
    
    total_items = len(processed_items) + len(sentence_items)
    print(f"Total items to process: {total_items}")
    print(f"Already processed: {len(processed_items)}")
    print(f"Remaining: {len(sentence_items)}")
    print(f"Batch size: {batch_size}, Max concurrent: {max_concurrent}")

    # Process in batches
    processed_count = len(processed_items)
    last_checkpoint = time.time()
    checkpoint_interval = 60  # Save checkpoint every minute
    
    try:
        with tqdm(total=len(sentence_items), desc="Processing sentence batches") as pbar:
            # Process in batches
            for i in range(0, len(sentence_items), batch_size):
                batch = sentence_items[i:i + batch_size]
                sentences = [item[3] for item in batch]  # Extract sentences
                
                # Process batch asynchronously
                results = await caller.score_batch_async(sentences)
                
                # Process results
                for j, (ridx, col, sidx, sent) in enumerate(batch):
                    scored = results[j] if j < len(results) else {
                        "probs": {strat: 0.0 for strat in strategies.keys()}, 
                        "labels_above_threshold": []
                    }
                    
                    record = {
                        "row_index": ridx,
                        "column": col,
                        "sentence_index": sidx,
                        "sentence": sent,
                    }
                    probs = scored.get("probs", {})
                    for strat in strategies.keys():
                        record[f"prob::{strat}"] = float(probs.get(strat, 0.0))
                    record["labels>=threshold"] = "; ".join(scored.get("labels_above_threshold", []))
                    rows_sent.append(record)
                    processed_items.add((ridx, col, sidx))
                    processed_count += 1
                
                pbar.update(len(batch))
                
                # Save checkpoint periodically
                if time.time() - last_checkpoint > checkpoint_interval:
                    progress_info = {
                        "processed_count": processed_count,
                        "total_count": total_items,
                        "batch_progress": i + len(batch),
                        "api_stats": {
                            "total_requests": caller.total_requests,
                            "failed_requests": caller.failed_requests
                        }
                    }
                    save_checkpoint(checkpoint_path, rows_sent, progress_info)
                    print(f"Checkpoint saved. Progress: {processed_count}/{total_items} ({100*processed_count/total_items:.1f}%)")
                    last_checkpoint = time.time()

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving final checkpoint...")
        progress_info = {
            "processed_count": processed_count,
            "total_count": total_items,
            "interrupted": True,
            "api_stats": {
                "total_requests": caller.total_requests,
                "failed_requests": caller.failed_requests
            }
        }
        save_checkpoint(checkpoint_path, rows_sent, progress_info)
        print(f"Checkpoint saved to {checkpoint_path}")
        print("You can resume processing by running the script again with the same parameters.")
        raise  # Re-raise to be handled by caller

    # Clean up checkpoint file if completed successfully
    if processed_count >= total_items and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Processing completed successfully. Checkpoint file removed.")

    # Print final statistics
    print(f"\nFinal API Statistics:")
    print(f"  Total requests: {caller.total_requests}")
    print(f"  Failed requests: {caller.failed_requests}")
    if caller.total_requests > 0:
        success_rate = 100 * (1 - caller.failed_requests / caller.total_requests)
        print(f"  Success rate: {success_rate:.1f}%")

    sent_df = pd.DataFrame(rows_sent)

    # Cell-level aggregation (union of sentence labels)
    cell_rows = []
    if not sent_df.empty:
        grp = sent_df.groupby(["row_index", "column"], dropna=False)
        for (ridx, col), g in grp:
            labels = sorted({lab.strip() for labs in g["labels>=threshold"] for lab in str(labs).split(";") if lab.strip()})
            cell_rows.append({
                "row_index": ridx,
                "column": col,
                "labels>=threshold": "; ".join(labels),
                "num_sentences": int(g.shape[0]),
            })
    cell_df = pd.DataFrame(cell_rows)

    summary = pd.DataFrame({
        "input_rows": [len(df)],
        "sentence_rows": [len(sent_df)],
        "cell_rows": [len(cell_df)],
        "threshold": [threshold],
        "model": [model],
        "columns": [", ".join(target_cols)],
        "batch_size": [batch_size],
        "max_concurrent": [max_concurrent],
    })

    return df, sent_df, cell_df, summary

def write_outputs_xlsx(out_path: str, df, sent_df, cell_df, summary):
    with pd.ExcelWriter(out_path, engine="openpyxl") as wr:
        df.to_excel(wr, sheet_name="Data", index=False)
        if not sent_df.empty:
            sent_df.to_excel(wr, sheet_name="Sentence_Probs", index=False)
            compact = sent_df[["row_index","column","sentence_index","sentence","labels>=threshold"]]
            compact.to_excel(wr, sheet_name="Sentence_Labels", index=False)
        if not cell_df.empty:
            cell_df.to_excel(wr, sheet_name="Cell_Labels", index=False)
        summary.to_excel(wr, sheet_name="Summary", index=False)

def write_outputs_csv(prefix: str, df, sent_df, cell_df, summary, include_data=False):
    # Note: CSV cannot store multiple "sheets", so we write multiple files using a common prefix.
    if include_data:
        df.to_csv(f"{prefix}_data.csv", index=False)
    if not sent_df.empty:
        sent_df.to_csv(f"{prefix}_sentence_probs.csv", index=False)
        compact = sent_df[["row_index","column","sentence_index","sentence","labels>=threshold"]]
        compact.to_csv(f"{prefix}_sentence_labels.csv", index=False)
    if not cell_df.empty:
        cell_df.to_csv(f"{prefix}_cell_labels.csv", index=False)
    summary.to_csv(f"{prefix}_summary.csv", index=False)

def main():
    ap = argparse.ArgumentParser(description="LLM-based strategy labeling with async batch processing")
    ap.add_argument("--in", dest="in_path", required=True, help="Path to the Excel file (e.g., 'Completions Data.xlsx')")
    ap.add_argument("--out", dest="out_path", default="completions_llm_labels.xlsx", help="Output XLSX path when using --format xlsx/both")
    ap.add_argument("--out-prefix", dest="out_prefix", default=None, help="Prefix for CSV outputs (default: stem of --out)")
    ap.add_argument("--sheet", default=0, help="Sheet name or index (default: first sheet)")
    ap.add_argument("--start-col", dest="start_col", default="D")
    ap.add_argument("--end-col", dest="end_col", default="O")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--strategies", dest="strategies_path", default="strategy_metadata.json")
    ap.add_argument("--format", dest="fmt", choices=["xlsx","csv","both"], default="xlsx")
    ap.add_argument("--include-data-csv", action="store_true", help="Also export the input Data sheet as CSV when --format csv/both")
    ap.add_argument("--checkpoint", dest="checkpoint_path", default=None, help="Custom checkpoint file path")
    ap.add_argument("--no-resume", action="store_true", help="Don't resume from checkpoint (start fresh)")
    ap.add_argument("--batch-size", type=int, default=10, help="Batch size for parallel processing (default: 10)")
    ap.add_argument("--max-concurrent", type=int, default=5, help="Max concurrent requests (default: 5)")
    args = ap.parse_args()

    # Validate API key at startup
    print("Validating API key...")
    api_key = validate_api_key()
    print("✓ API key found")

    # Validate input files
    if not os.path.exists(args.in_path):
        print(f"ERROR: Input file not found: {args.in_path}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(args.strategies_path):
        print(f"ERROR: Strategies file not found: {args.strategies_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Input file: {args.in_path}")
    print(f"Strategies: {args.strategies_path}")
    print(f"Model: {args.model} (threshold: {args.threshold})")
    print(f"🚀 Using ASYNC BATCH mode for better performance!")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max concurrent: {args.max_concurrent}")
    print(f"  Expected speedup: {args.max_concurrent}x faster")
    
    try:
        df, sent_df, cell_df, summary = asyncio.run(annotate_file_async(
            in_path=args.in_path,
            sheet=args.sheet,
            start_col_letter=args.start_col,
            end_col_letter=args.end_col,
            threshold=args.threshold,
            model=args.model,
            strategies_path=args.strategies_path,
            checkpoint_path=args.checkpoint_path,
            resume=not args.no_resume,
            batch_size=args.batch_size,
            max_concurrent=args.max_concurrent,
        ))
    except KeyboardInterrupt:
        print("\nScript interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Decide prefixes and write
    out_stem = args.out_path.rsplit(".", 1)[0]
    prefix = args.out_prefix or out_stem

    if args.fmt in ("xlsx", "both"):
        write_outputs_xlsx(args.out_path, df, sent_df, cell_df, summary)
    if args.fmt in ("csv", "both"):
        write_outputs_csv(prefix, df, sent_df, cell_df, summary, include_data=args.include_data_csv)

    print("Done.")
    if args.fmt in ("xlsx", "both"):
        print(f"  Wrote XLSX -> {args.out_path}")
    if args.fmt in ("csv", "both"):
        print(f"  Wrote CSVs -> {prefix}_sentence_probs.csv, {prefix}_sentence_labels.csv, {prefix}_cell_labels.csv, {prefix}_summary.csv")
        if args.include_data_csv:
            print(f"  (Plus) {prefix}_data.csv")

if __name__ == "__main__":
    main()