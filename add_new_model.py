#!/usr/bin/env python3
"""
Generalized script for adding new models to the evaluation system.

Usage:
    python add_new_model.py --model-name "new_model" --model-type "api" --api-url "https://api.example.com" --api-key "key"
    python add_new_model.py --model-name "gpt4_turbo" --model-type "openai" --api-key "sk-..."
    python add_new_model.py --model-name "claude3_opus" --model-type "claude" --api-key "sk-ant-..."
    python add_new_model.py --model-name "gemini-2.0-flash-exp" --model-type "gemini" --api-key "..."

This script will:
1. Generate combined_with_model_completions_{modelname}.json
2. Update combined_with_all_model_completions_X.json with sequential numbering
3. Generate pairwise evaluations for the new model vs all existing models
4. Save pairwise files in pairwise-human-completion-leaderboard-X folder
"""

import json
import os
import argparse
import glob
import shutil
from pathlib import Path
import sys
import concurrent.futures
import threading
import time
import asyncio
from openai import OpenAI
try:
    import anthropic
except ImportError:
    anthropic = None
try:
    import google.generativeai as genai
except ImportError:
    genai = None

# Internal model names for classification
INTERNAL_MODELS = ['i60', 'i60l300b', 'i102l300b', 'i102l4maverick', 'i104l4maverick', 'i107l4maverick', 'hippoai', 'gptoss']

def is_internal_model(model_name):
    """Check if a model is internal based on its name."""
    model_lower = model_name.lower()
    for internal_model in INTERNAL_MODELS:
        if internal_model in model_lower:
            return True
    return False

MODELS = [
    {
         "name": "kimik2",
         "url": "https://endpoint/kimik2/openai/v1",
         "api_key": "API_KEY_HERE",
         "model_name": "moonshotai/kimi-k2-instruct-0905",
         "eval_mode": "external"
    }
]

def find_next_sequential_number(base_path, pattern):
    """Find the next sequential number for a file pattern."""
    existing_files = glob.glob(f"{base_path}/{pattern}*")
    if not existing_files:
        return 2  # Start with 2 since 1 is usually the original
    
    numbers = []
    for file in existing_files:
        # Extract number from filename
        parts = os.path.basename(file).split('_')
        for part in parts:
            if part.replace('.json', '').isdigit():
                numbers.append(int(part.replace('.json', '')))
    
    if not numbers:
        return 2
    return max(numbers) + 1

def load_base_data(base_file):
    """Load the base completions file."""
    if not os.path.exists(base_file):
        raise FileNotFoundError(f"Base file {base_file} not found")
    
    with open(base_file, 'r') as f:
        return json.load(f)

def build_prompt(dialogue_history, prev_speakers=None, eval_mode="external"):
    """Build the user prompt for the model - same as the generation script"""
    # If no prev_speakers provided, alternate between seeker and supporter
    if prev_speakers is None:
        prev_speakers = ["seeker" if i % 2 == 0 else "supporter" for i in range(len(dialogue_history))]
    
    # Annotate the transcript with Seeker: and Supporter: labels
    annotated_transcript = []
    for turn, speaker in zip(dialogue_history, prev_speakers):
        if speaker == "seeker":
            annotated_transcript.append(f"Seeker: {turn}")
        else:  # speaker == "supporter"
            annotated_transcript.append(f"Supporter: {turn}")
    
    transcript = "\n".join(annotated_transcript)
    
    if eval_mode == "internal":
        return (
            f"This is a roleplay, you playing an emotionally intelligent human who is, essentially, yourself. Treat it like a real situation. Always respond in first person."
            f"You are the Supporter in a conversation with a Seeker. "
            f"Guidelines:\\n"
            f"- Respond with empathy and understanding\\n"
            f"- Keep your response conversational and natural\\n"
            f"- Build on what the person just said\\n"
            f"- Do NOT repeat the conversation\\n"
            f"- Write **EXACTLY ONE** Supporter reply and nothing else. The response should be a maximum of 40 words.\\n"
            f"- Do NOT include any Seeker text, speaker labels, or quotation marks\\n\\n"
            f"Conversation so far:\\n{transcript}\\n"
            f"Supporter:"
        )
    
    return (
        f"This is a roleplay, you playing an emotionally intelligent human who is, essentially, yourself. Treat it like a real situation. Always respond in first person."
        f"You are the Supporter in a conversation with a Seeker. "
        f"The Seeker is someone who needs emotional support and guidance. "
        f"The Supporter (you) provides empathy, understanding, and helpful responses. "
        f"Guidelines:\\n"
        f"- Respond with empathy and understanding\\n"
        f"- Keep your response conversational and natural (1-3 sentences)\\n"
        f"- Build on what the person just said\\n"
        f"- Do NOT repeat the conversation\\n"
        f"- Write **EXACTLY ONE** Supporter reply and nothing else. The response should be a maximum of 40 words.\\n"
        f"- Do NOT include any Seeker text, speaker labels, or quotation marks\\n\\n"
        f"Conversation so far:\\n{transcript}\\n"
        f"Supporter:"
    )

def clean_response(response):
    """Clean the model response - same as the generation script"""
    response = response.strip()
    
    # Remove any obvious response labels
    response = response.replace('Your response:', '').replace('Response:', '')
    response = response.replace('My response:', '').replace('RESPONSE:', '')
    
    # Remove speaker labels if the model accidentally includes them
    response = response.replace('Supporter:', '').replace('Seeker:', '')
    response = response.replace('supporter:', '').replace('seeker:', '')
    
    response = response.strip()
    return response

# Track warning prints across calls so we don't spam per entry
_WARNED_MAX_TOKENS_MODELS = set()
_WARNED_TEMPERATURE_MODELS = set()

def query_gpt_model(prompt, system_msg, model="gpt-4o", api_key=None, base_url=None, use_completions=False, retries=5):
    """Query OpenAI GPT model"""
    client_kwargs = {}
    if api_key:
        client_kwargs['api_key'] = api_key
    elif 'OPENAI_API_KEY' in os.environ:
        client_kwargs['api_key'] = os.environ['OPENAI_API_KEY']
    if base_url:
        # Fix: Remove trailing /openai/v1 if present, as OpenAI client will add the endpoint
        # This prevents double paths like /openai/v1/chat/completions
        if base_url.endswith('/openai/v1'):
            base_url = base_url.rsplit('/openai/v1', 1)[0]
        client_kwargs['base_url'] = base_url
    
    client = OpenAI(**client_kwargs)
    
    model_l = model.lower()
    # Default flags – newer models (o3, gpt-5, gpt-4.1) start with stricter defaults
    is_new_model = model_l.startswith(("o3", "gpt-5", "gpt-4.1"))
    is_gpt5 = model_l == "gpt-5"
    is_o3 = model_l.startswith("o3")
    use_max_completion_tokens = is_new_model  # new models default to new parameter
    # gpt-5: let API default temperature; others near-deterministic if allowed
    use_temperature = False if is_gpt5 else not is_new_model
    temperature_value = 0.7            # Prefer near-deterministic responses when allowed
    # Start with a generous cap; for newer models this uses max_completion_tokens/max_output_tokens
    if is_gpt5:
        max_tokens_value = 4000
    elif is_o3:
        max_tokens_value = 4000
    else:
        max_tokens_value = 256
    warned_max_tokens = False
    warned_temperature = False
    retried_empty = False

    for attempt in range(retries):
        try:
            # For gpt-5, use the responses API to set max_output_tokens explicitly
            if is_gpt5:
                resp = client.responses.create(
                    model=model,
                    input=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ],
                    max_output_tokens=max_tokens_value,
                )
                completion = resp.output_text or ""
            elif use_completions:
                # Use completions endpoint
                full_prompt = f"{system_msg}\n\n{prompt}"
                comp_kwargs = {
                    "model": model,
                    "prompt": full_prompt,
                    "max_tokens": max_tokens_value,
                }
                if use_temperature:
                    comp_kwargs["temperature"] = temperature_value

                response = client.completions.create(**comp_kwargs)
                completion = response.choices[0].text
            else:
                # Use chat completions endpoint
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ]
                chat_kwargs = {
                    "model": model,
                    "messages": messages,
                }
                if use_temperature:
                    chat_kwargs["temperature"] = temperature_value
                # Newer models require max_completion_tokens instead of max_tokens
                if use_max_completion_tokens:
                    chat_kwargs["max_completion_tokens"] = max_tokens_value
                else:
                    chat_kwargs["max_tokens"] = max_tokens_value

                response = client.chat.completions.create(**chat_kwargs)
                content = response.choices[0].message.content
                # Handle str, list-of-dicts/objects, or objects with .text
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text" and "text" in part:
                            parts.append(part["text"])
                        elif hasattr(part, "text"):
                            parts.append(part.text)
                        elif isinstance(part, str):
                            parts.append(part)
                    completion = "\n".join(parts)
                elif hasattr(content, "text"):
                    completion = content.text
                else:
                    completion = content

            # Clean and validate completion
            completion_text = clean_response(completion or "")
            if not completion_text:
                if not retried_empty and not is_gpt5:
                    # One retry with the same (or slightly larger) cap, no temp
                    print("[warn] GPT returned empty completion; retrying once without temperature adjustments")
                    retried_empty = True
                    use_temperature = False
                    if max_tokens_value < 256:
                        max_tokens_value = 256
                    continue
                raise ValueError("Empty completion text")
            return completion_text
        except Exception as exc:
            error_str = str(exc).lower()
            if 'max_tokens' in error_str and 'max_completion_tokens' in error_str and not use_max_completion_tokens:
                if not warned_max_tokens and model not in _WARNED_MAX_TOKENS_MODELS:
                    print("[warn] GPT API requires max_completion_tokens; switching to that parameter")
                    warned_max_tokens = True
                    _WARNED_MAX_TOKENS_MODELS.add(model)
                use_max_completion_tokens = True
                continue
            if 'temperature' in error_str and 'unsupported' in error_str and use_temperature:
                if not warned_temperature and model not in _WARNED_TEMPERATURE_MODELS:
                    print("[warn] GPT model does not accept custom temperature; reverting to default")
                    warned_temperature = True
                    _WARNED_TEMPERATURE_MODELS.add(model)
                use_temperature = False
                continue
            if 'max_tokens' in error_str and 'output limit was reached' in error_str:
                # Reduce max tokens and retry
                new_val = max(64, max_tokens_value // 2)
                if new_val < max_tokens_value:
                    print(f"[warn] GPT hit output limit; reducing max tokens to {new_val} and retrying")
                    max_tokens_value = new_val
                    continue
            # Special handling for rate limits
            if 'RateLimitError' in exc.__class__.__name__ or 'rate_limit' in str(exc).lower():
                # For rate limits, wait longer and extract suggested wait time if available
                wait_time = 2 ** attempt
                
                # Try to extract wait time from error message
                import re
                error_str = str(exc)
                wait_match = re.search(r'try again in (\d+\.?\d*)s', error_str)
                if wait_match:
                    suggested_wait = float(wait_match.group(1))
                    wait_time = max(wait_time, suggested_wait + 1)  # Add 1 second buffer
                
                print(f"[warn] GPT {exc.__class__.__name__}: Rate limit hit — waiting {wait_time:.1f}s before retry")
                time.sleep(wait_time)
            else:
                wait = 2 ** attempt
                print(f"[warn] GPT {exc.__class__.__name__}: {exc} — retrying in {wait}s")
                time.sleep(wait)
    raise RuntimeError("OpenAI API failed after several retries")

def query_claude_model(prompt, system_msg, model="claude-3-5-sonnet-20241022", api_key=None, retries=3, max_tokens=400):
    """Query Anthropic Claude model"""
    if anthropic is None:
        raise ImportError("anthropic package is required. Install with: pip install anthropic")
    
    client_kwargs = {}
    if api_key:
        client_kwargs['api_key'] = api_key
    elif 'ANTHROPIC_API_KEY' in os.environ:
        client_kwargs['api_key'] = os.environ['ANTHROPIC_API_KEY']
    
    client = anthropic.Anthropic(**client_kwargs)
    
    for attempt in range(retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_msg,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return clean_response(response.content[0].text) if max_tokens <= 400 else response.content[0].text.strip()
        except Exception as exc:
            # Check for Cloudflare errors in the response
            if hasattr(exc, 'response') and exc.response and hasattr(exc.response, 'text'):
                if '522' in str(exc.response.text) or '525' in str(exc.response.text):
                    wait = 5 * (attempt + 1)  # Longer wait for Cloudflare errors
                    print(f"[warn] Cloudflare error detected, waiting {wait}s before retry...")
                    time.sleep(wait)
                    continue
            
            wait = 2 ** attempt
            print(f"[warn] Claude {exc.__class__.__name__}: {exc} — retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("Claude API failed after several retries")

def query_gemini_model(prompt, system_msg, model="gemini-2.0-flash-exp", api_key=None, retries=3, max_tokens=400, delay_seconds=12.0):
    """Query Google Gemini model - will retry indefinitely on rate limits"""
    if genai is None:
        raise ImportError("google-generativeai package is required. Install with: pip install google-generativeai")

    # Normalize common aliases to API model ids
    model_aliases = {
        "gemini25flash": "gemini-2.5-flash",
        "gemini25pro": "gemini-2.5-pro",
        "gemini3": "gemini-3-pro-preview",
    }
    model_id = model_aliases.get(model, model)
    
    # Configure API key
    if api_key:
        genai.configure(api_key=api_key)
    elif 'GOOGLE_API_KEY' in os.environ:
        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
    else:
        raise ValueError("No Gemini API key provided")
    
    # Add rate limit delay. Increase to reduce quota errors.
    time.sleep(delay_seconds)
    
    attempt = 0
    while True:  # Retry indefinitely until success
        try:
            # Create model
            gemini_model = genai.GenerativeModel(model_id)
            
            # Combine system message and prompt
            full_prompt = f"{system_msg}\n\n{prompt}"
            
            # Generate response
            # Note: max_output_tokens removed due to bug in gemini-2.5-flash/pro that returns empty responses
            response = gemini_model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                )
            )
            
            # Check if response has valid content
            if not response.candidates:
                raise ValueError("No candidates in response")
            
            candidate = response.candidates[0]
            
            # Check finish reason
            # 0=UNSPECIFIED, 1=STOP (good), 2=MAX_TOKENS, 3=SAFETY, 4=RECITATION, 5=OTHER
            if hasattr(candidate, 'finish_reason'):
                if candidate.finish_reason == 2:  # MAX_TOKENS
                    # Try to get partial content if available
                    if candidate.content and candidate.content.parts:
                        text = candidate.content.parts[0].text
                        print(f"[warn] Response hit max_tokens, using partial: {text[:50]}...")
                        return clean_response(text)
                    else:
                        raise ValueError("Response hit MAX_TOKENS with no content")
                elif candidate.finish_reason == 3:  # SAFETY
                    raise ValueError("Response blocked by safety filters")
                elif candidate.finish_reason == 4:  # RECITATION
                    raise ValueError("Response blocked by recitation filters")
                elif candidate.finish_reason not in [0, 1]:  # Not STOP or UNSPECIFIED
                    raise ValueError(f"Response stopped with finish_reason={candidate.finish_reason}")
            
            # Try to get the text
            if hasattr(response, 'text') and response.text:
                return clean_response(response.text)
            elif candidate.content and candidate.content.parts:
                text = candidate.content.parts[0].text
                return clean_response(text)
            else:
                raise ValueError("No valid text in response")
            
        except Exception as exc:
            attempt += 1
            error_str = str(exc).lower()
            
            # Check if it's a daily quota issue (needs to wait until next day)
            if 'quota' in error_str and ('day' in error_str or 'daily' in error_str):
                print(f"[ERROR] Daily quota exceeded! Gemini free tier has 32,000 tokens/day limit.")
                print(f"        You've likely hit the daily token quota due to dialogue length.")
                print(f"        Options:")
                print(f"          1. Wait until tomorrow (quota resets at midnight UTC)")
                print(f"          2. Enable billing on your Gemini API key")
                print(f"          3. Use a different API key")
                raise RuntimeError(f"Gemini daily quota exceeded - cannot continue")
            
            # Handle safety/content filter blocks - these won't succeed on retry
            elif 'safety' in error_str or 'blocked' in error_str:
                print(f"[ERROR] Content blocked by safety filters: {exc}")
                print(f"        This dialogue cannot be processed by Gemini due to content policy.")
                raise RuntimeError(f"Content blocked - skipping this entry")
            
            # Handle per-minute rate limits - wait and retry forever
            elif 'quota' in error_str or 'rate' in error_str or 'resource_exhausted' in error_str:
                # Check for specific TPM (tokens per minute) vs RPM limits
                if 'token' in error_str:
                    wait_time = min(600, 180 * (attempt + 1))  # back off harder than 60s
                    print(f"[warn] Gemini TOKEN rate limit hit — waiting {wait_time}s (attempt {attempt}); error: {exc}")
                else:
                    wait_time = min(600, 180 * (attempt + 1))  # back off harder than 60s
                    print(f"[warn] Gemini REQUEST rate limit hit — waiting {wait_time}s (attempt {attempt}); error: {exc}")
                time.sleep(wait_time)
                # Continue to next iteration (retry)
            
            # Handle response validation errors (max_tokens, etc) - retry with longer output
            elif 'max_tokens' in error_str or 'finish_reason' in error_str:
                print(f"[warn] Response generation issue: {exc} — retrying (attempt {attempt})")
                # Don't increase max_tokens since we're already at 400 which should be enough
                # Just retry - might be a transient issue
                time.sleep(2)
                # Continue to next iteration (retry)
                
            # Handle other errors - retry with exponential backoff, but don't give up
            else:
                # Cap the wait time at 60 seconds for non-rate-limit errors
                wait = min(2 ** min(attempt, 6), 60)  # Max 60s wait
                print(f"[warn] Gemini error: {exc} — retrying in {wait}s (attempt {attempt})")
                time.sleep(wait)
                # Continue to next iteration (retry)
        
        # Note: This function will never raise an error - it keeps retrying until successful

def process_single_entry(entry, model_config, system_msg, entry_index):
    """Process a single entry to generate completion."""
    # Create a copy of the entry
    new_entry = entry.copy()
    save_key = model_config['save_name']
    # Skip if we already have a non-empty, non-error completion
    existing = new_entry.get(save_key)
    if isinstance(existing, str) and existing.strip() and not existing.strip().lower().startswith("error:"):
        return new_entry, entry_index
    if existing not in (None, "", float("nan")):
        # For non-string types, if present and truthy, keep as is
        try:
            import math
            if isinstance(existing, float) and math.isnan(existing):
                pass  # treat NaN as missing
            else:
                return new_entry, entry_index
        except Exception:
            return new_entry, entry_index
    
    # Generate completion for this dialogue
    dialogue_history = entry['dialogue_history']
    prev_speakers = entry.get('prev_speakers', None)
    eval_mode = model_config.get('eval_mode', 'external')
    
    # Build the prompt using the same logic as the generation script
    prompt = build_prompt(dialogue_history, prev_speakers, eval_mode)
    
    try:
        # Generate completion based on model type
        if model_config['type'] == 'openai':
            completion = query_gpt_model(
                prompt=prompt,
                system_msg=system_msg,
                model=model_config.get('model_name', 'gpt-4o'),
                api_key=model_config['api_key'],
                use_completions=model_config.get('use_completions_endpoint', False)
            )
        elif model_config['type'] == 'claude':
            completion = query_claude_model(
                prompt=prompt,
                system_msg=system_msg,
                model=model_config.get('model_name', 'claude-3-5-sonnet-20241022'),
                api_key=model_config['api_key']
            )
        elif model_config['type'] == 'gemini':
            completion = query_gemini_model(
                prompt=prompt,
                system_msg=system_msg,
                model=model_config.get('model_name', 'gemini-2.0-flash-exp'),
                api_key=model_config.get('api_key'),
                delay_seconds=model_config.get('gemini_delay_seconds', 12.0)
            )
        elif model_config['type'] == 'api':
            # Custom API endpoint (assume OpenAI-compatible)
            completion = query_gpt_model(
                prompt=prompt,
                system_msg=system_msg,
                model=model_config.get('model_name', 'default'),
                api_key=model_config['api_key'],
                base_url=model_config['api_url'],
                use_completions=model_config.get('use_completions_endpoint', False)
            )
        else:
            raise ValueError(f"Unknown model type: {model_config['type']}")
        
        # Take only the first line of the response (like the generation script)
        completion_lines = completion.split("\n")
        final_completion = completion_lines[0].strip()
        if not final_completion:
            raise ValueError("Empty completion text")

        # Add the completion to the entry
        new_entry[model_config['save_name']] = final_completion
        # if model_config['type'] == 'gemini':
        #     print(f"[info] Gemini success for entry {entry_index}: {final_completion[:80]!r}")

    except Exception as e:
        print(f"Error generating completion for entry {entry_index}: {e}")
        new_entry[model_config['save_name']] = f"Error: {str(e)}"
    
    return new_entry, entry_index

def generate_model_completions(data, model_config, parallel_workers=6, batch_size=20):
    """Generate completions for the new model using the base data with parallel processing."""
    print(f"Generating completions for {model_config['name']} with {parallel_workers} workers, batch size {batch_size}...")
    
    # Default system message
    system_msg = (
        "You are an empathetic, compassionate listener skilled at resolving conflicts. "
        "Given a conversation, you deeply analyze what both parties are feeling and "
        "generate a thoughtful, kind, and constructive response that aims to resolve "
        "the situation. Return only the response as plain text, no JSON, no formatting."
    )
    
    # Process data in batches with parallel workers
    updated_data = [None] * len(data)
    total_entries = len(data)
    processed = 0
    
    for batch_start in range(0, total_entries, batch_size):
        batch_end = min(batch_start + batch_size, total_entries)
        batch_data = data[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}/{(total_entries + batch_size - 1)//batch_size} ({batch_start}-{batch_end-1})...")
        
        # Process batch in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            # Submit all tasks in this batch
            future_to_index = {
                executor.submit(process_single_entry, entry, model_config, system_msg, batch_start + i): batch_start + i
                for i, entry in enumerate(batch_data)
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_index):
                original_index = future_to_index[future]
                try:
                    result_entry, _ = future.result()
                    updated_data[original_index] = result_entry
                    processed += 1
                    if processed % 50 == 0:
                        print(f"  Completed {processed}/{total_entries} entries...")
                except Exception as e:
                    print(f"  Error processing entry {original_index}: {e}")
                    # Create error entry
                    error_entry = data[original_index].copy()
                    error_entry[model_config['save_name']] = f"Error: {str(e)}"
                    updated_data[original_index] = error_entry
                    processed += 1
    
    print(f"Completed {processed}/{total_entries} entries")
    return updated_data

def save_model_completions(data, save_name):
    """Save the completions to model-specific file."""
    filename = f"combined_with_model_completions_{save_name}.json"
    # with open(filename, 'w') as f:
    #     json.dump(data, f, indent=2)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open('w') as f:
        json.dump(updated_data, f, indent=2)
    print(f"Saved model completions to {filename}")
    return filename

def update_combined_files(data, model_save_name, is_internal, regular_file, with_internal_file):
    """
    Update both combined files:
    1. combined_with_all_model_completions.json (non-internal models only)
    2. combined_with_all_model_completions_with_internal.json (all models)

    Files are REPLACED, not incremented.
    """
    # Always update the with_internal file (it contains all models)
    with open(with_internal_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Updated: {with_internal_file}")

    # Update the regular file (non-internal only)
    if not is_internal:
        # If the new model is not internal, update the regular file too
        with open(regular_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Updated: {regular_file}")
    else:
        # If the new model is internal, filter it out from the regular file
        print(f"Model {model_save_name} is internal - updating only combined_with_all_model_completions_with_internal.json")

        # Load existing regular file and add only non-internal model data
        if os.path.exists(regular_file):
            # The regular file should not have this model's data
            # (it's already correct, no need to update)
            print(f"{regular_file} remains unchanged (internal model excluded)")
        else:
            # Create regular file by filtering out internal models
            filtered_data = []
            for dialogue in data:
                new_dialogue = {}
                for key, value in dialogue.items():
                    # Check if this is a model-specific field
                    if key.endswith('_completion') or key.endswith('_model_used'):
                        # Extract model name
                        if key.endswith('_completion'):
                            model_name = key.replace('_completion', '')
                        else:
                            model_name = key.replace('_model_used', '')

                        # Only include if not internal
                        if not is_internal_model(model_name):
                            new_dialogue[key] = value
                    else:
                        # Keep all non-model fields
                        new_dialogue[key] = value

                filtered_data.append(new_dialogue)

            with open(regular_file, 'w') as f:
                json.dump(filtered_data, f, indent=2)
            print(f"Created: {regular_file} (filtered to exclude internal models)")

def get_existing_models():
    """Get list of existing models from the base data."""
    try:
        data = load_base_data()
        if not data:
            return []
        
        # Get all keys from first entry except standard fields and metadata fields
        metadata_fields = {
            'dialogue_history', 'emotion_type', 'problem_type', 'turn', 'prev_speakers',
            'situation', 'human_completion_1', 'human_completion_2', 'human_completion_3', 
            'human_completion_4', 'human_completion_5', 'chosen_likert', 'reason', 
            'rating_user', 'timestamp', 'source', 'evaluation'
        }
        
        sample_entry = data[0]
        models = [key for key in sample_entry.keys() if key not in metadata_fields]
        
        # Additional filtering: only include fields that contain model completions
        # Model completion fields typically end with '_completion' or are simple model names
        # Exclude '_model_used' fields as they are metadata
        actual_models = []
        for model in models:
            # Keep if it's a completion field or a simple model name, but exclude metadata
            if (model.endswith('_completion') or 
                (not model.endswith('_model_used') and 
                 not any(metadata_word in model.lower() for metadata_word in ['human_completion', 'likert', 'rating', 'timestamp', 'source', 'reason']))):
                actual_models.append(model)
        
        return actual_models
    except:
        return []

def generate_pairwise_evaluations(save_name, existing_models):
    """Generate pairwise evaluation files for the new model vs all existing models."""
    print(f"Generating pairwise evaluations for {save_name} vs all existing models...")
    
    # Create model-specific pairwise folder
    pairwise_folder = f"{save_name}-pairwise"
    
    # Create the folder
    os.makedirs(pairwise_folder, exist_ok=True)
    print(f"Created pairwise folder: {pairwise_folder}")
    
    # Load the model completions file
    model_file = f"combined_with_model_completions_{save_name}.json"
    if not os.path.exists(model_file):
        print(f"Error: Model completions file {model_file} not found")
        return
    
    with open(model_file, 'r') as f:
        data = json.load(f)
    
    # Generate pairwise files for this model vs each existing model
    evaluators = ['openai', 'claude']
    
    for other_model in existing_models:
        if other_model == save_name:
            continue
            
        print(f"  Generating {save_name} vs {other_model}...")
        
        for evaluator in evaluators:
            # Create pairwise comparison data with dialogue_history
            pairwise_data = []
            
            for i, entry in enumerate(data):
                # Get completion for save_name model
                if save_name not in entry:
                    continue
                    
                # Get completion for other model (handle both _completion and direct names)
                other_completion = None
                if other_model in entry:
                    other_completion = entry[other_model]
                elif f"{other_model}_completion" in entry:
                    other_completion = entry[f"{other_model}_completion"]
                
                if not other_completion:
                    continue
                
                # Randomize model presentation order
                import random
                if random.choice([True, False]):
                    # Model A = new model, Model B = other model  
                    presentation_order = {"A0493": "model_a", "A0488": "model_b"}
                    conv_a_model = save_name
                    conv_a_response = entry[save_name]
                    conv_b_model = other_model
                    conv_b_response = other_completion
                else:
                    # Model A = other model, Model B = new model
                    presentation_order = {"A0493": "model_b", "A0488": "model_a"} 
                    conv_a_model = other_model
                    conv_a_response = other_completion
                    conv_b_model = save_name
                    conv_b_response = entry[save_name]
            
                # Create pairwise comparison entry with dialogue_history
                pairwise_entry = {
                    "row_index": i,
                    "dialogue_history": entry.get("dialogue_history", []),  # Include dialogue history
                    "presentation_order": presentation_order,
                    "conversation_a_metadata": {
                        "emotion_type": entry.get('emotion_type', ''),
                        "problem_type": entry.get('problem_type', ''),
                        "final_turn": conv_a_response,
                        "turn": entry.get('turn', 1),
                        "model_name": conv_a_model
                    },
                    "conversation_b_metadata": {
                        "emotion_type": entry.get('emotion_type', ''),
                        "problem_type": entry.get('problem_type', ''),
                        "final_turn": conv_b_response,
                        "turn": entry.get('turn', 1),
                        "model_name": conv_b_model
                    },
                    "evaluation": {}  # Will be filled by evaluation process
                }
                pairwise_data.append(pairwise_entry)
            
            # Save pairwise file
            pairwise_filename = f"{pairwise_folder}/{save_name}-vs-{other_model}-{evaluator}.json"
            with open(pairwise_filename, 'w') as f:
                for entry in pairwise_data:
                    f.write(json.dumps(entry) + '\n')
            
            print(f"    Saved {len(pairwise_data)} comparisons to {pairwise_filename}")
    
    print(f"Pairwise evaluation files saved in {pairwise_folder}")
    return pairwise_folder

async def async_evaluate_batch(entries_with_indices, evaluator):
    """Evaluate a batch of entries asynchronously in parallel."""
    tasks = []
    
    if evaluator == 'claude':
        # Create async Claude client
        client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        for entry, idx in entries_with_indices:
            task = async_evaluate_claude(client, entry, idx)
            tasks.append(task)
    else:  # openai
        # Create async OpenAI client
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        for entry, idx in entries_with_indices:
            task = async_evaluate_openai(client, entry, idx)
            tasks.append(task)
    
    # Run all evaluations in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"      Error in batch: {result}")
            # Keep original entry on error
            processed_results.append((entries_with_indices[i][0], entries_with_indices[i][1]))
        else:
            processed_results.append(result)
    
    return processed_results

async def async_evaluate_claude(client, entry, idx):
    """Evaluate a single entry using async Claude API."""
    try:
        # Skip if already evaluated
        existing_eval = entry.get('evaluation', {})
        if existing_eval and not existing_eval.get('error') and existing_eval.get('overall_eq'):
            return (entry, idx)
        
        # Build evaluation prompt
        dialogue_history = entry.get('dialogue_history', [])
        record_a = {
            'dialogue_history': dialogue_history,
            'final_turn': entry['conversation_a_metadata']['final_turn'],
            'emotion_type': entry['conversation_a_metadata'].get('emotion_type', ''),
            'problem_type': entry['conversation_a_metadata'].get('problem_type', ''),
            'model': entry['conversation_a_metadata']['model_name']
        }
        record_b = {
            'dialogue_history': dialogue_history,
            'final_turn': entry['conversation_b_metadata']['final_turn'],
            'emotion_type': entry['conversation_b_metadata'].get('emotion_type', ''),
            'problem_type': entry['conversation_b_metadata'].get('problem_type', ''),
            'model': entry['conversation_b_metadata']['model_name']
        }
        
        eval_prompt, presentation_mapping = build_evaluation_prompt_fixed(record_a, record_b)
        eval_system_msg = get_evaluation_system_message()
        
        # Make async API call
        response = await client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            system=eval_system_msg,
            messages=[{"role": "user", "content": eval_prompt}]
        )
        
        evaluation_text = response.content[0].text.strip()
        evaluation_dict = parse_evaluation_result(evaluation_text)
        
        # Replace model names
        model_a_name = entry['conversation_a_metadata']['model_name']
        model_b_name = entry['conversation_b_metadata']['model_name']
        evaluation_dict = replace_model_names_in_evaluation(
            evaluation_dict, model_a_name, model_b_name, presentation_mapping
        )
        
        # Update entry
        entry['presentation_order'] = presentation_mapping
        entry['evaluation'] = evaluation_dict
        
        return (entry, idx)
        
    except Exception as e:
        print(f"      Error evaluating entry {idx}: {e}")
        entry['evaluation'] = {"error": str(e)}
        return (entry, idx)

async def async_evaluate_openai(client, entry, idx):
    """Evaluate a single entry using async OpenAI API."""
    try:
        # Skip if already evaluated
        existing_eval = entry.get('evaluation', {})
        if existing_eval and not existing_eval.get('error') and existing_eval.get('overall_eq'):
            return (entry, idx)
        
        # Build evaluation prompt
        dialogue_history = entry.get('dialogue_history', [])
        record_a = {
            'dialogue_history': dialogue_history,
            'final_turn': entry['conversation_a_metadata']['final_turn'],
            'emotion_type': entry['conversation_a_metadata'].get('emotion_type', ''),
            'problem_type': entry['conversation_a_metadata'].get('problem_type', ''),
            'model': entry['conversation_a_metadata']['model_name']
        }
        record_b = {
            'dialogue_history': dialogue_history,
            'final_turn': entry['conversation_b_metadata']['final_turn'],
            'emotion_type': entry['conversation_b_metadata'].get('emotion_type', ''),
            'problem_type': entry['conversation_b_metadata'].get('problem_type', ''),
            'model': entry['conversation_b_metadata']['model_name']
        }
        
        eval_prompt, presentation_mapping = build_evaluation_prompt_fixed(record_a, record_b)
        eval_system_msg = get_evaluation_system_message()
        
        # Make async API call
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": eval_system_msg},
                {"role": "user", "content": eval_prompt}
            ]
        )
        
        evaluation_text = response.choices[0].message.content.strip()
        evaluation_dict = parse_evaluation_result(evaluation_text)
        
        # Replace model names
        model_a_name = entry['conversation_a_metadata']['model_name']
        model_b_name = entry['conversation_b_metadata']['model_name']
        evaluation_dict = replace_model_names_in_evaluation(
            evaluation_dict, model_a_name, model_b_name, presentation_mapping
        )
        
        # Update entry
        entry['presentation_order'] = presentation_mapping
        entry['evaluation'] = evaluation_dict
        
        return (entry, idx)
        
    except Exception as e:
        print(f"      Error evaluating entry {idx}: {e}")
        entry['evaluation'] = {"error": str(e)}
        return (entry, idx)

async def process_single_file_async(pairwise_file, batch_size):
    """Process evaluations for a single pairwise file asynchronously."""
    print(f"📄 Starting {os.path.basename(pairwise_file)}...")
    
    # Load the pairwise data
    with open(pairwise_file, 'r') as f:
        pairwise_data = [json.loads(line) for line in f]
    
    # Count how many need evaluation
    needs_eval = []
    for i, entry in enumerate(pairwise_data):
        existing_eval = entry.get('evaluation', {})
        # Skip if already has valid evaluation (not empty and not error)
        if not existing_eval or existing_eval.get('error') or not existing_eval.get('overall_eq'):
            needs_eval.append(i)
    
    if not needs_eval:
        print(f"  ✅ All evaluations complete for {os.path.basename(pairwise_file)}")
        return True
    
    print(f"  📊 {len(needs_eval)} entries need evaluation (of {len(pairwise_data)} total)")
    
    # Determine evaluator from filename
    if '-openai.json' in pairwise_file:
        evaluator = 'openai'
    elif '-claude.json' in pairwise_file:
        evaluator = 'claude'
    else:
        print(f"  ⚠️ Skipping {pairwise_file}: Unknown evaluator")
        return False
    
    # Copy original data
    updated_data = pairwise_data.copy()
    processed = 0
    
    # Process only entries that need evaluation in batches
    for batch_start in range(0, len(needs_eval), batch_size):
        batch_end = min(batch_start + batch_size, len(needs_eval))
        batch_indices = needs_eval[batch_start:batch_end]
        
        # Prepare batch entries
        batch_entries = [(pairwise_data[idx], idx) for idx in batch_indices]
        
        # Run async evaluations in parallel within batch
        batch_results = await async_evaluate_batch(batch_entries, evaluator)
        
        # Update results
        for entry, idx in batch_results:
            updated_data[idx] = entry
            processed += 1
        
        # Save after each batch
        with open(pairwise_file, 'w') as f:
            for entry in updated_data:
                f.write(json.dumps(entry) + '\n')
    
    print(f"  ✅ Completed {os.path.basename(pairwise_file)} - {processed} evaluations")
    return True

async def run_evaluations_on_pairwise_files_async(save_name, parallel_workers=6, batch_size=20):
    """Run evaluations on existing pairwise files for a model - processing multiple files in parallel."""
    pairwise_folder = f"{save_name}-pairwise"
    
    if not os.path.exists(pairwise_folder):
        print(f"Error: Pairwise folder {pairwise_folder} not found")
        return False
    
    print(f"🚀 Running evaluations on pairwise files in {pairwise_folder}...")
    print(f"⚙️ Settings: {parallel_workers} parallel files, batch size {batch_size}")
    
    # Find all pairwise files
    pairwise_files = glob.glob(f"{pairwise_folder}/*.json")
    
    if not pairwise_files:
        print(f"No pairwise files found in {pairwise_folder}")
        return False
    
    print(f"📁 Found {len(pairwise_files)} pairwise files to process")
    
    # Create semaphore to limit concurrent file processing
    semaphore = asyncio.Semaphore(parallel_workers)
    
    async def process_file_with_semaphore(file_path):
        async with semaphore:
            return await process_single_file_async(file_path, batch_size)
    
    # Create tasks for all files
    tasks = []
    for pairwise_file in pairwise_files:
        task = process_file_with_semaphore(pairwise_file)
        tasks.append(task)
    
    # Process all files with controlled parallelism
    print(f"\n🏁 Starting evaluation of {len(tasks)} files with up to {parallel_workers} in parallel...")
    start_time = time.time()
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Count results
    successful = sum(1 for r in results if r is True)
    failed = sum(1 for r in results if isinstance(r, Exception))
    skipped = len(results) - successful - failed
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n✨ All evaluations completed in {duration:.1f} seconds!")
    print(f"📊 Results: ✅ {successful} successful, ❌ {failed} failed, ⏭️ {skipped} skipped")
    
    return successful > 0

def evaluate_single_pairwise(entry, evaluator, entry_index):
    """Evaluate a single pairwise comparison entry."""
    # Skip if already evaluated and not an error
    existing_eval = entry.get('evaluation', {})
    if existing_eval and not existing_eval.get('error'):
        return entry, entry_index
    
    try:
        # Build structured records for evaluation
        dialogue_history = entry.get('dialogue_history', [])
        
        # Create record_a and record_b in the format expected by the evaluation functions
        record_a = {
            'dialogue_history': dialogue_history,
            'final_turn': entry['conversation_a_metadata']['final_turn'],
            'emotion_type': entry['conversation_a_metadata'].get('emotion_type', ''),
            'problem_type': entry['conversation_a_metadata'].get('problem_type', ''),
            'model': entry['conversation_a_metadata']['model_name']
        }
        
        record_b = {
            'dialogue_history': dialogue_history,
            'final_turn': entry['conversation_b_metadata']['final_turn'],
            'emotion_type': entry['conversation_b_metadata'].get('emotion_type', ''),
            'problem_type': entry['conversation_b_metadata'].get('problem_type', ''),
            'model': entry['conversation_b_metadata']['model_name']
        }
        
        # Build evaluation prompt using the same logic as the reference script
        eval_prompt, presentation_mapping = build_evaluation_prompt_fixed(record_a, record_b)
        eval_system_msg = get_evaluation_system_message()
        
        # Run evaluation
        if evaluator == 'openai':
            evaluation_result = query_gpt_model(
                prompt=eval_prompt,
                system_msg=eval_system_msg,
                model="gpt-4o",
                api_key=None,  # Will use environment variable
                use_completions=False
            )
        elif evaluator == 'claude':
            evaluation_result = query_claude_model(
                prompt=eval_prompt,
                system_msg=eval_system_msg,
                model="claude-3-5-sonnet-20241022",
                api_key=None,  # Will use environment variable
                max_tokens=4096  # Evaluations need more tokens
            )
        else:
            raise ValueError(f"Unknown evaluator: {evaluator}")
        
        # Parse evaluation result and update entry
        evaluation_dict = parse_evaluation_result(evaluation_result)
        
        # Replace A0493/A0488 with actual model names in the evaluation
        model_a_name = entry['conversation_a_metadata']['model_name']
        model_b_name = entry['conversation_b_metadata']['model_name']
        evaluation_dict = replace_model_names_in_evaluation(
            evaluation_dict, model_a_name, model_b_name, presentation_mapping
        )
        
        # Update presentation order in the entry
        entry['presentation_order'] = presentation_mapping
        entry['evaluation'] = evaluation_dict
        
    except Exception as e:
        print(f"Error evaluating entry {entry_index}: {e}")
        entry['evaluation'] = {"error": str(e)}
    
    return entry, entry_index

def build_evaluation_prompt_fixed(record_a, record_b, randomize_order=True):
    """Build the evaluation prompt using the same format as the reference Claude script."""
    import random
    
    # Get dialogue history (should be same for both)
    dialogue_history = record_a.get('dialogue_history', [])
    
    # Format dialogue as conversation turns
    conversation_parts = []
    for i, turn in enumerate(dialogue_history):
        if i % 2 == 0:
            conversation_parts.append(f"Seeker: {turn}")
        else:
            conversation_parts.append(f"Supporter: {turn}")
    
    # Add final turns
    conversation_a = "\n".join(conversation_parts) + f"\nFinal Response: {record_a['final_turn']}"
    conversation_b = "\n".join(conversation_parts) + f"\nFinal Response: {record_b['final_turn']}"
    
    # Randomize presentation order
    if randomize_order and random.choice([True, False]):
        # Swap - present B as A0493 and A as A0488
        conversation_history_A = conversation_b
        conversation_history_B = conversation_a
        final_turn_a = record_b['final_turn']
        final_turn_b = record_a['final_turn']
        presentation_mapping = {"A0493": "model_b", "A0488": "model_a"}
        emotion_type = record_b.get('emotion_type', '') or record_a.get('emotion_type', '')
    else:
        # Keep original order - A as A0493 and B as A0488
        conversation_history_A = conversation_a
        conversation_history_B = conversation_b
        final_turn_a = record_a['final_turn']
        final_turn_b = record_b['final_turn']
        presentation_mapping = {"A0493": "model_a", "A0488": "model_b"}
        emotion_type = record_a.get('emotion_type', '') or record_b.get('emotion_type', '')
    
    # Create scenario notes
    scenario_notes = "Final responses from each model:\n"
    scenario_notes += f"A0493 Final Turn: {final_turn_a}\n"
    scenario_notes += f"A0488 Final Turn: {final_turn_b}"
    
    if emotion_type:
        scenario_notes += f"\n\nContext: This conversation addresses {emotion_type} support."
    
    # Build the full prompt using the reference template
    prompt = f"""[RESPONDENT A0493]
{conversation_history_A}

[RESPONDENT A0488]
{conversation_history_B}

You will compare the relative ability of each respondent for these evaluation criteria.
Categories and subcategories to help guide your judgment:

Empathic Responsiveness
• Empathy / Warmth / Affirmation
• Acknowledgement & Validation
• Perspective-taking
• Empowering language (respect for autonomy)
• Gentle Challenge

Emotional Handling & Insight
• Emotional & social reasoning
• Tone and frustration de-escalation
• Handling of complex or mixed emotions

Personalization & Contextual Adaptation
• Message tailoring (content + tone)
• Contextual awareness & memory retention
• Emotion-informed action planning
• Self-disclosure
• Likeability & Engagement

Conversational Fluency & Naturalness
• Human-like Conversation
• Turn-taking, timing, interactive flow
• Response variation & sentence structure
• Groundedness, clarity, conversational fillers when appropriate

Instruction Following & Safety
• Instruction following
• Boundary setting
• Giving Information

Notes on the scenario to assist judging:
{scenario_notes}

Judging Instructions:

For each criterion, you must always pick a winner (A0493 or A0488; never a tie).
For each, rate the magnitude of the win using a plus-based scale: "+" / "++" / "+++" / "++++" / "+++++" Example: "A0493++" = A0493 is somewhat stronger; "A0488++++" = A0488 is far stronger.
Judgments are based only on assistant messages in the dialogue above—ignore the user and any "debrief" content.
Use your discretion; good emotional intelligence is context-sensitive and doesn't always mean being analytical or verbose.
The output must be valid JSON, structured exactly as below (escape all strings correctly):
{{
  "chain_of_thought_reasoning": "detailed chain of thought reasoning about the coming scoring decisions. Include exactly one line of reasoning for each of the 6 categories: (1) empathic_responsiveness, (2) emotional_handling_and_insight, (3) personalization_and_contextual_adaptation, (4) conversational_fluency_and_naturalness, (5) instruction_following_and_safety, and (6) overall_eq",
  "empathic_responsiveness": "winner & disparity rating",
  "emotional_handling_and_insight": "winner & disparity rating",
  "personalization_and_contextual_adaptation": "winner & disparity rating",
  "conversational_fluency_and_naturalness": "winner & disparity rating",
  "instruction_following_and_safety": "winner & disparity rating",
  "overall_eq": "winner & disparity rating"
}}
Judge solely on what is present in the assistant dialogues for each respondent, referencing the above criteria and subcategories."""
    
    return prompt, presentation_mapping

def build_evaluation_prompt(dialogue_history, response_a, response_b):
    """Build the evaluation prompt for comparing two responses."""
    # Convert dialogue to text with proper formatting
    if dialogue_history:
        dialogue_text = "\n".join(f"Turn {i+1}: {turn}" for i, turn in enumerate(dialogue_history))
    else:
        dialogue_text = "No previous dialogue context available."
    
    return f"""Given this emotional support conversation:

{dialogue_text}

Please evaluate these two responses:

Response A: {response_a}

Response B: {response_b}

Which response is better for providing emotional support? Consider empathy, helpfulness, appropriateness, and emotional intelligence.

You must respond with a JSON object containing detailed evaluations. Here's the required format:

{{
  "chain_of_thought_reasoning": "Your detailed reasoning for the comparison...",
  "empathic_responsiveness": "A0493+" or "A0488+" (with appropriate + symbols),
  "emotional_handling_and_insight": "A0493+" or "A0488+" (with appropriate + symbols),
  "personalization_and_contextual_adaptation": "A0493+" or "A0488+" (with appropriate + symbols),
  "conversational_fluency_and_naturalness": "A0493+" or "A0488+" (with appropriate + symbols),
  "instruction_following_and_safety": "A0493+" or "A0488+" (with appropriate + symbols),
  "overall_eq": "A0493+" or "A0488+" (with appropriate + symbols)
}}

Use + symbols to indicate confidence level: + = somewhat better, ++ = clearly better, +++ = much better."""

def get_evaluation_system_message():
    """Get the system message for evaluations - matching the reference script."""
    return (
        "You are an expert evaluator of emotional support conversations. "
        "You will compare two conversations across multiple criteria and provide "
        "detailed evaluations. Always pick a winner for each criterion and rate "
        "the magnitude of the win using the specified scale."
    )

def replace_model_names_in_evaluation(evaluation_result, model_a_name, model_b_name, presentation_mapping):
    """Replace A0493/A0488 with actual model names in evaluation results."""
    import re
    
    # Determine which identifier maps to which model
    if presentation_mapping["A0493"] == "model_a":
        identifier_to_model = {"A0493": model_a_name, "A0488": model_b_name}
    else:
        identifier_to_model = {"A0493": model_b_name, "A0488": model_a_name}
    
    # Create a new evaluation dict with replacements
    updated_eval = {}
    for key, value in evaluation_result.items():
        if isinstance(value, str):
            # Replace identifiers with model names
            updated_value = value
            for identifier, model_name in identifier_to_model.items():
                # Replace identifier followed by + symbols
                pattern = f"{identifier}(\\+*)"
                replacement = f"{model_name}\\1"
                updated_value = re.sub(pattern, replacement, updated_value)
            updated_eval[key] = updated_value
        else:
            updated_eval[key] = value
    
    return updated_eval

def parse_evaluation_result(evaluation_text):
    """Parse the evaluation result into a structured format."""
    evaluation_text = evaluation_text.strip()
    
    try:
        # Try to parse as JSON first
        result = json.loads(evaluation_text)
        if isinstance(result, dict) and 'overall_eq' in result:
            return result
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from the text
    json_start = evaluation_text.find('{')
    json_end = evaluation_text.rfind('}')
    
    if json_start != -1 and json_end != -1:
        try:
            json_text = evaluation_text[json_start:json_end+1]
            result = json.loads(json_text)
            if isinstance(result, dict) and 'overall_eq' in result:
                return result
        except json.JSONDecodeError:
            pass
    
    # Fallback: try to extract simple winner format
    if 'A0493' in evaluation_text:
        winner = 'A0493'
    elif 'A0488' in evaluation_text:
        winner = 'A0488'
    else:
        return {"error": "Could not parse evaluation"}
    
    # Count confidence level
    plus_count = evaluation_text.count('+')
    confidence_str = '+' * plus_count if plus_count > 0 else ''
    
    # Return in the expected format
    return {
        "chain_of_thought_reasoning": "Fallback parsing - original response may have been malformed",
        "empathic_responsiveness": winner + confidence_str,
        "emotional_handling_and_insight": winner + confidence_str,
        "personalization_and_contextual_adaptation": winner + confidence_str,
        "conversational_fluency_and_naturalness": winner + confidence_str,
        "instruction_following_and_safety": winner + confidence_str,
        "overall_eq": winner + confidence_str
    }

async def main_async():
    parser = argparse.ArgumentParser(description='Add a new model to the evaluation system')
    parser.add_argument('--model-name', help='Name of the model to query (e.g., o3-2025-04-16). Required unless using --eval-only')
    parser.add_argument('--model-name-save', help='Name to save files as (e.g., gpto3). If not provided, uses --model-name')
    parser.add_argument('--model-type', choices=['openai', 'claude', 'gemini', 'api'], 
                       help='Type of model: openai, claude, gemini, or api (custom endpoint). Required unless using --eval-only')
    parser.add_argument('--api-key', help='API key for the model (defaults to environment variables OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY)')
    parser.add_argument('--api-url', help='API URL (required for api type)')
    parser.add_argument('--use-completions', action='store_true', 
                       help='Use completions endpoint instead of chat completions')
    parser.add_argument('--skip-completions', action='store_true',
                       help='Skip generating completions (if already exist)')
    parser.add_argument('--skip-pairwise', action='store_true',
                       help='Skip generating pairwise evaluations')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only run evaluations on existing pairwise files (requires --model-name-save)')
    parser.add_argument('--eval-mode', choices=['external', 'internal'], default='external',
                       help='Evaluation mode for prompt generation')
    parser.add_argument('--parallel-workers', type=int, default=6,
                       help='Number of parallel workers for completion generation (default: 6)')
    parser.add_argument('--batch-size', type=int, default=20,
                       help='Batch size for processing completions (default: 20)')
    parser.add_argument('--internal', action='store_true',
                       help='Mark this model as internal (will be excluded from combined_with_all_model_completions.json)')
    parser.add_argument('--external', action='store_true',
                       help='Force mark this model as external (overrides auto-detection of internal models)')
    parser.add_argument('--base-file', default='all_model_completions_regular.json',
                       help='Input base completions file (default: all_model_completions_regular.json)')
    parser.add_argument('--output-file', default='all_model_completions_filled.json',
                       help='Output file to write filled completions (default: all_model_completions_filled.json)')
    parser.add_argument('--gemini-delay-seconds', type=float, default=12.0,
                       help='Delay between Gemini requests (seconds) to avoid rate limits (default: 12.0)')

    args = parser.parse_args()

    # Validate internal/external flags
    if args.internal and args.external:
        print("Error: Cannot specify both --internal and --external")
        sys.exit(1)
    
    # Handle eval-only mode (disabled for pairwise generation in this workflow)
    if args.eval_only:
        print("Eval-only mode skipped: pairwise evaluation generation is disabled in this workflow.")
        return
    
    # Validate arguments for normal mode
    if not args.model_name:
        print("Error: --model-name is required unless using --eval-only")
        sys.exit(1)
    
    if not args.model_type:
        print("Error: --model-type is required unless using --eval-only")
        sys.exit(1)
    
    if args.model_type == 'api' and not args.api_url:
        print("Error: --api-url is required for api model type")
        sys.exit(1)
    
    # Check for API keys
    if not args.api_key:
        if args.model_type == 'openai' and 'OPENAI_API_KEY' not in os.environ:
            print("Error: --api-key required or set OPENAI_API_KEY environment variable")
            sys.exit(1)
        elif args.model_type == 'claude' and 'ANTHROPIC_API_KEY' not in os.environ:
            print("Error: --api-key required or set ANTHROPIC_API_KEY environment variable")
            sys.exit(1)
        elif args.model_type == 'api':
            print("Error: --api-key is required for custom API endpoints")
            sys.exit(1)
    
    # Use model-name-save if provided, otherwise use model-name
    save_name = args.model_name_save if args.model_name_save else args.model_name

    # Check if model should be marked as internal
    # Priority: explicit flags > auto-detection
    if args.external:
        is_model_internal = False
        print(f"ℹ️  Model '{save_name}' marked as EXTERNAL (explicit)")
    elif args.internal:
        is_model_internal = True
        print(f"ℹ️  Model '{save_name}' marked as INTERNAL (explicit)")
    else:
        # Auto-detect based on model name
        is_model_internal = is_internal_model(save_name)
        if is_model_internal:
            print(f"ℹ️  Model '{save_name}' auto-detected as INTERNAL (use --external to override)")
        else:
            print(f"ℹ️  Model '{save_name}' auto-detected as EXTERNAL (use --internal to override)")

    # Create model configuration
    model_config = {
        'name': args.model_name,  # Name to query the API
        'save_name': save_name,   # Name to save in files
        'type': args.model_type,
        'api_key': args.api_key,
        'use_completions_endpoint': args.use_completions,
        'eval_mode': args.eval_mode,
        'is_internal': is_model_internal,
        'gemini_delay_seconds': args.gemini_delay_seconds
    }

    if args.api_url:
        model_config['api_url'] = args.api_url

    # Set the model name to query (for API calls)
    model_config['model_name'] = args.model_name
    
    try:
        # Step 1: Load base data
        print(f"Loading base data from {args.base_file}...")
        base_data = load_base_data(args.base_file)
        
        # Step 2: Generate completions (unless skipped)
        if not args.skip_completions:
            updated_data = generate_model_completions(base_data, model_config, args.parallel_workers, args.batch_size)
            
            # Save filled output to the requested file
            with open(args.output_file, 'w') as f:
                json.dump(updated_data, f, indent=2)
            print(f"Saved filled completions to {args.output_file}")
        else:
            print("Skipping completion generation...")
            updated_data = base_data
        
        # Pairwise generation disabled
        print("\\n🎉 Completion fill complete!")
        print(f"📝 Files use '{save_name}' as the model identifier and were written to {args.output_file}")
        if save_name != args.model_name:
            print(f"🔗 Model API calls used '{args.model_name}'")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

def main():
    """Entry point that sets up and runs the async main."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()
