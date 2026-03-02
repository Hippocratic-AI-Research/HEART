#!/usr/bin/env python3
"""
Bradley-Terry Score Evaluation using Pairwise Evaluation for Dialogue Systems.
Calculates Bradley-Terry model strengths for dialogue models based on pairwise comparison results.
Uses maximum likelihood estimation to solve for model strengths.
"""

import argparse
import json
import logging
import os
import sys
import copy  
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set, Union
from collections import defaultdict, Counter
from datetime import datetime, timezone
import re
import math
import random
import glob

# Fallback implementations for numpy functions
def mean_fallback(values: List[float]) -> float:
    """Fallback implementation for numpy.mean"""
    return sum(values) / len(values) if values else 0.0

def std_fallback(values: List[float]) -> float:
    """Fallback implementation for numpy.std"""
    if len(values) < 2:
        return 0.0
    mean_val = mean_fallback(values)
    variance = sum((x - mean_val) ** 2 for x in values) / len(values)
    return math.sqrt(variance)

def norm_ppf_fallback(p: float) -> float:
    """Fallback implementation for scipy.stats.norm.ppf (approximation)"""
    # Simple approximation for 95% confidence interval (p=0.975)
    if abs(p - 0.975) < 0.001:
        return 1.96
    elif abs(p - 0.025) < 0.001:
        return -1.96
    else:
        # Very rough approximation - for production use, consider a proper implementation
        return 1.96 if p > 0.5 else -1.96

# Try to import numpy, with fallbacks
try:
    import numpy as np  # type: ignore
    np_mean = np.mean
    np_std = np.std
    np_exp = np.exp
    np_log = np.log
    np_array = np.array
    np_zeros = np.zeros
    np_ones = np.ones
    HAS_NUMPY = True
except ImportError:
    np_mean = mean_fallback
    np_std = std_fallback
    np_exp = math.exp
    np_log = math.log
    np_array = lambda x: x
    np_zeros = lambda n: [0.0] * n
    np_ones = lambda n: [1.0] * n
    HAS_NUMPY = False

try:
    from scipy import stats  # type: ignore
    from scipy.optimize import minimize  # type: ignore
    norm_ppf = stats.norm.ppf
    HAS_SCIPY = True
except ImportError:
    norm_ppf = norm_ppf_fallback
    HAS_SCIPY = False

# ───────────────────────────── Bradley-Terry Configuration ────────────────────────── #

# Bradley-Terry System Constants
DEFAULT_STRENGTH = 1.0  # Default strength parameter
DEFAULT_RATING = 1500.0  # Default rating for display
INITIAL_SIGMA = 200.0  # Initial uncertainty for new models
MAX_ITERATIONS = 1000  # Maximum iterations for Bradley-Terry estimation
CONVERGENCE_THRESHOLD = 1e-6  # Convergence threshold for parameter estimation
STRENGTH_REGULARIZATION = 1e-6  # Regularization to prevent infinite strengths

# Numerical stability constants
MAX_STRENGTH = 1000.0  # Maximum allowed strength
MIN_STRENGTH = 0.001   # Minimum allowed strength
MAX_RATING = 3000.0    # Maximum allowed rating
MIN_RATING = 500.0     # Minimum allowed rating

# Evaluation Categories (Updated for current data format)
EVALUATION_CATEGORIES = [
    "empathic_responsiveness",
    "emotional_handling_and_insight",
    "personalization_and_contextual_adaptation", 
    "conversational_fluency_and_naturalness",
    "instruction_following_and_safety",
    "overall_eq"
]

CATEGORY_DISPLAY_NAMES = {
    "empathic_responsiveness": "Empathic Responsiveness",
    "emotional_handling_and_insight": "Emotional Handling & Insight", 
    "personalization_and_contextual_adaptation": "Personalization & Contextual Adaptation",
    "conversational_fluency_and_naturalness": "Conversational Fluency & Naturalness",
    "instruction_following_and_safety": "Instruction Following & Safety",
    "overall_eq": "Overall EQ"
}

# ───────────────────────────── Utility Functions ───────────────────────── #

def parse_evaluation_score(score_str: str) -> Tuple[Optional[str], int]:
    """
    Parse evaluation score like "A0493+++" into (winner, strength).
    Returns: (winner, strength_score) where strength_score is 1-5
    """
    if not score_str or score_str == "Error":
        return None, 0
    
    # Extract model ID and plus signs - supports alphanumeric with dots, underscores, hyphens
    match = re.match(r'([A-Za-z0-9._-]+)(\+*)', score_str.strip())
    if not match:
        return None, 0
    
    winner = match.group(1)
    plus_signs = match.group(2)
    strength = len(plus_signs) if plus_signs else 1
    
    return winner, strength

def strength_to_score(strength: int) -> float:
    """
    Convert plus-sign strength to match score with win margin consideration.
    """
    if strength <= 0:
        return 0.5  # Draw/no clear winner
    elif strength == 1:
        return 0.55  # Slight advantage
    elif strength == 2:
        return 0.65  # Moderate advantage
    elif strength == 3:
        return 0.75  # Clear advantage
    elif strength == 4:
        return 0.85  # Strong advantage
    else:  # strength >= 5
        return 0.95  # Dominant performance

def strength_to_rating(strength: float, base_rating: float = DEFAULT_RATING) -> float:
    """
    Convert Bradley-Terry strength to rating for display.
    Uses log scale to convert strengths to ratings.
    """
    if strength <= 0:
        strength = MIN_STRENGTH
    rating = base_rating + 400 * np_log(strength) / np_log(10)
    return max(MIN_RATING, min(MAX_RATING, rating))

def calculate_confidence_interval(strength: float, n_comparisons: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for Bradley-Terry strength.
    Uses asymptotic normal approximation.
    """
    if n_comparisons <= 1:
        return strength * 0.5, strength * 2.0
    
    # Asymptotic standard error approximation
    se = math.sqrt(1.0 / max(1, n_comparisons))
    z_score = norm_ppf(1 - (1 - confidence) / 2)
    
    # Calculate on log scale then transform back
    log_strength = np_log(max(MIN_STRENGTH, strength))
    log_margin = z_score * se
    
    lower = np_exp(log_strength - log_margin)
    upper = np_exp(log_strength + log_margin)
    
    return max(MIN_STRENGTH, lower), min(MAX_STRENGTH, upper)

def extract_models_from_record(record: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract the two models being compared from an evaluation record.
    """
    # Try to get models from metadata (preferred method)
    model_a_meta = record.get("conversation_a_metadata", {})
    model_b_meta = record.get("conversation_b_metadata", {})
    
    model_a_id = model_a_meta.get("model_name")
    model_b_id = model_b_meta.get("model_name")
    
    if model_a_id and model_b_id:
        return model_a_id, model_b_id
    
    # Fallback: look for model identifiers in evaluation scores
    evaluation = record.get("evaluation", {})
    if not evaluation:
        return None, None
    
    models = set()
    for category in EVALUATION_CATEGORIES:
        if category in evaluation:
            winner, _ = parse_evaluation_score(evaluation[category])
            if winner:
                models.add(winner)
    
    # Should have exactly 2 models in pairwise comparison
    if len(models) == 2:
        return tuple(sorted(models))  # Sort for consistency
    elif len(models) == 1:
        # If only one model appears, we need to infer the other
        logging.debug(f"Only found one model in evaluation: {models}")
        return tuple(models)[0], "UNKNOWN"
    else:
        logging.debug(f"Found {len(models)} models in evaluation: {models}")
        return None, None

# ───────────────────────────── Bradley-Terry Model Classes ───────────────────────── #

class BTModel:
    """Represents a model with Bradley-Terry strength and statistics."""
    
    def __init__(self, model_id: str, initial_strength: float = DEFAULT_STRENGTH):
        self.model_id = model_id
        self.strength = initial_strength
        self.initial_strength = initial_strength
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.strength_history = [initial_strength]
        self.category_strengths = {cat: initial_strength for cat in EVALUATION_CATEGORIES}
        self.category_games = {cat: 0 for cat in EVALUATION_CATEGORIES}
        self.category_wins = {cat: 0 for cat in EVALUATION_CATEGORIES}  # Track category-specific wins
        self.category_losses = {cat: 0 for cat in EVALUATION_CATEGORIES}  # Track category-specific losses
        self.category_draws = {cat: 0 for cat in EVALUATION_CATEGORIES}  # Track category-specific draws
        self.win_margins = []  # Track win margins for statistics
        
    def update_strength(self, new_strength: float, category: str = "overall"):
        """Update model strength."""
        new_strength = max(MIN_STRENGTH, min(MAX_STRENGTH, new_strength))
        
        if category == "overall":
            self.strength = new_strength
            self.strength_history.append(new_strength)
        else:
            self.category_strengths[category] = new_strength
            
    def record_game(self, result: float, win_margin: float = 0.0, category: str = "overall"):
        """Record game result with win margin tracking. Each model counts every game they participate in."""
        if category == "overall":
            # Every game is counted for each participating model
            self.games_played += 1
            if result > 0.52:  # More sensitive threshold for wins
                self.wins += 1
                self.win_margins.append(win_margin)
            elif result < 0.48:  # More sensitive threshold for losses
                self.losses += 1
                self.win_margins.append(-win_margin)
            else:
                self.draws += 1
                self.win_margins.append(0.0)
        else:
            # Count category games and results for each model
            if category in self.category_games:
                self.category_games[category] += 1
                if result > 0.52:  # More sensitive threshold for wins
                    self.category_wins[category] += 1
                elif result < 0.48:  # More sensitive threshold for losses
                    self.category_losses[category] += 1
                else:
                    self.category_draws[category] += 1
            
    def win_rate(self) -> float:
        """Calculate overall win rate."""
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played
    
    def category_win_rate(self, category: str) -> float:
        """Calculate win rate for a specific category."""
        if category not in self.category_games or self.category_games[category] == 0:
            return 0.0
        return self.category_wins[category] / self.category_games[category]
    
    def average_win_margin(self) -> float:
        """Calculate average win margin."""
        if not self.win_margins:
            return 0.0
        return np_mean(self.win_margins)
    
    def get_rating(self) -> float:
        """Get rating equivalent of strength."""
        return strength_to_rating(self.strength)
    
    def confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for the strength."""
        return calculate_confidence_interval(self.strength, self.games_played, confidence)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        ci_low, ci_high = self.confidence_interval()
        
        return {
            "model_id": self.model_id,
            "strength": round(self.strength, 6),
            "rating": round(self.get_rating(), 2),
            "ci_low": round(ci_low, 6),
            "ci_high": round(ci_high, 6),
            "initial_strength": self.initial_strength,
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": round(self.win_rate(), 3),
            "average_win_margin": round(self.average_win_margin(), 3),
            "strength_history": [round(s, 6) for s in self.strength_history],
            "category_strengths": {k: round(v, 6) for k, v in self.category_strengths.items()},
            "category_games": self.category_games.copy(),
            "category_wins": self.category_wins.copy(),
            "category_losses": self.category_losses.copy(),
            "category_draws": self.category_draws.copy()
        }

class BTComparison:
    """Represents a pairwise comparison for Bradley-Terry model."""
    
    def __init__(self, model_a: str, model_b: str, score: float, weight: float = 1.0):
        self.model_a = model_a
        self.model_b = model_b
        self.score = score  # Score for model A (0-1)
        self.weight = weight  # Weight of this comparison
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "score": self.score,
            "weight": self.weight
        }

class BradleyTerryEstimator:
    """
    Bradley-Terry model estimator using maximum likelihood estimation.
    """
    
    def __init__(self, regularization: float = STRENGTH_REGULARIZATION):
        self.regularization = regularization
        self.models = set()
        self.comparisons = []
        
    def add_comparison(self, model_a: str, model_b: str, score: float, weight: float = 1.0):
        """Add a pairwise comparison."""
        self.models.add(model_a)
        self.models.add(model_b)
        self.comparisons.append(BTComparison(model_a, model_b, score, weight))
        
    def estimate_strengths_iterative(self, max_iterations: int = MAX_ITERATIONS) -> Dict[str, float]:
        """
        Estimate Bradley-Terry strengths using iterative algorithm.
        This is a fallback when scipy is not available.
        """
        if not self.models or not self.comparisons:
            return {}
        
        # Initialize strengths
        model_list = sorted(self.models)
        n_models = len(model_list)
        model_to_idx = {model: i for i, model in enumerate(model_list)}
        
        # Initialize with equal strengths
        strengths = [DEFAULT_STRENGTH] * n_models
        
        # Iterative update (MM algorithm)
        for iteration in range(max_iterations):
            old_strengths = strengths.copy()
            
            # Calculate wins and total comparisons for each model
            wins = [0.0] * n_models
            totals = [0.0] * n_models
            
            for comp in self.comparisons:
                i = model_to_idx[comp.model_a]
                j = model_to_idx[comp.model_b]
                
                # Model A wins
                wins[i] += comp.score * comp.weight
                wins[j] += (1.0 - comp.score) * comp.weight
                
                # Total comparisons
                denom = strengths[i] + strengths[j]
                if denom > 0:
                    totals[i] += comp.weight * strengths[j] / denom
                    totals[j] += comp.weight * strengths[i] / denom
            
            # Update strengths
            for i in range(n_models):
                if totals[i] > 0:
                    strengths[i] = wins[i] / totals[i]
                else:
                    strengths[i] = DEFAULT_STRENGTH
                    
                # Apply regularization and bounds
                strengths[i] = max(MIN_STRENGTH, min(MAX_STRENGTH, strengths[i]))
            
            # Check convergence
            max_change = max(abs(strengths[i] - old_strengths[i]) for i in range(n_models))
            if max_change < CONVERGENCE_THRESHOLD:
                logging.info(f"Converged after {iteration + 1} iterations")
                break
        
        # Normalize strengths (set geometric mean to 1)
        if strengths:
            log_mean = sum(np_log(max(MIN_STRENGTH, s)) for s in strengths) / len(strengths)
            normalization = np_exp(log_mean)
            strengths = [s / normalization for s in strengths]
        
        return {model_list[i]: strengths[i] for i in range(n_models)}
    
    def estimate_strengths_scipy(self) -> Dict[str, float]:
        """
        Estimate Bradley-Terry strengths using scipy optimization.
        """
        if not HAS_SCIPY:
            return self.estimate_strengths_iterative()
        
        if not self.models or not self.comparisons:
            return {}
        
        model_list = sorted(self.models)
        n_models = len(model_list)
        model_to_idx = {model: i for i, model in enumerate(model_list)}
        
        def negative_log_likelihood(log_strengths):
            """Negative log-likelihood for Bradley-Terry model."""
            strengths = [np_exp(ls) for ls in log_strengths]
            
            nll = 0.0
            for comp in self.comparisons:
                i = model_to_idx[comp.model_a]
                j = model_to_idx[comp.model_b]
                
                prob_a_wins = strengths[i] / (strengths[i] + strengths[j])
                prob_a_wins = max(1e-10, min(1-1e-10, prob_a_wins))  # Numerical stability
                
                # Likelihood for this comparison
                ll = comp.score * np_log(prob_a_wins) + (1 - comp.score) * np_log(1 - prob_a_wins)
                nll -= comp.weight * ll
            
            # Add regularization
            nll += self.regularization * sum(ls * ls for ls in log_strengths)
            
            return nll
        
        # Initial guess (log of equal strengths)
        initial_log_strengths = [np_log(DEFAULT_STRENGTH)] * n_models
        
        # Optimize
        try:
            result = minimize(negative_log_likelihood, initial_log_strengths, method='BFGS')
            if result.success:
                log_strengths = result.x
                strengths = [np_exp(ls) for ls in log_strengths]
                
                # Normalize (set geometric mean to 1)
                log_mean = sum(log_strengths) / len(log_strengths)
                strengths = [np_exp(ls - log_mean) for ls in log_strengths]
                
                return {model_list[i]: strengths[i] for i in range(n_models)}
            else:
                logging.warning("Scipy optimization failed, falling back to iterative method")
                return self.estimate_strengths_iterative()
        except Exception as e:
            logging.warning(f"Scipy optimization error: {e}, falling back to iterative method")
            return self.estimate_strengths_iterative()
    
    def estimate_strengths(self) -> Dict[str, float]:
        """Estimate Bradley-Terry strengths using best available method."""
        if HAS_SCIPY:
            return self.estimate_strengths_scipy()
        else:
            return self.estimate_strengths_iterative()

class BradleyTerryEvaluationSystem:
    """Bradley-Terry-based evaluation system for dialogue models."""
    
    def __init__(self):
        self.models: Dict[str, BTModel] = {}
        self.comparisons: List[Dict[str, Any]] = []
        self.processed_comparisons = set()  # Track processed comparison IDs
        
    def get_or_create_model(self, model_id: str) -> BTModel:
        """Get existing model or create new one."""
        if model_id not in self.models:
            self.models[model_id] = BTModel(model_id)
            logging.info(f"Created new Bradley-Terry model: {model_id}")
        return self.models[model_id]
    
    def calculate_win_margin(self, strength: int) -> float:
        """Convert strength to win margin for statistics."""
        if strength <= 0:
            return 0.0
        elif strength == 1:
            return 0.1
        elif strength == 2:
            return 0.3
        elif strength == 3:
            return 0.5
        elif strength == 4:
            return 0.7
        else:  # strength >= 5
            return 0.9
    
    def process_evaluation_record(self, record: Dict[str, Any]) -> bool:
        """Process a single evaluation record."""
        # Create unique ID for this comparison to avoid duplicates
        comparison_id = f"{record.get('row_index', 0)}_{hash(str(record))}"
        if comparison_id in self.processed_comparisons:
            logging.debug(f"Skipping duplicate comparison: {comparison_id}")
            return False

        evaluation = record.get("evaluation", {})
        if not evaluation:
            logging.debug(f"Skipping record with no evaluation data: row_index={record.get('row_index', 'unknown')}")
            return False

        # Get the two models being compared
        model_a_id, model_b_id = extract_models_from_record(record)
        if model_a_id is None or model_b_id is None:
            logging.debug(f"Skipping record - could not extract model IDs: row_index={record.get('row_index', 'unknown')}, evaluation_keys={list(evaluation.keys())}")
            return False

        # Skip comparisons involving human baselines (human_1, human_2, human_3, etc.)
        # These should be filtered out and not included in the Bradley-Terry model
        if model_a_id.startswith('human_') or model_b_id.startswith('human_'):
            logging.debug(f"Skipping comparison with human baseline: {model_a_id} vs {model_b_id}")
            return False
            
        # Get or create model objects
        model_a = self.get_or_create_model(model_a_id)
        model_b = self.get_or_create_model(model_b_id)
        
        # Process each category
        categories_processed = 0
        for category in EVALUATION_CATEGORIES:
            if category not in evaluation:
                continue
                
            winner, strength = parse_evaluation_score(evaluation[category])
            if winner is None:
                logging.debug(f"Skipping category {category} - could not parse score: {evaluation[category]}")
                continue
                
            # Calculate score and win margin
            if winner == model_a_id:
                score_a = strength_to_score(strength)
                win_margin = self.calculate_win_margin(strength)
            else:  # winner == model_b_id
                score_a = 1.0 - strength_to_score(strength)
                win_margin = self.calculate_win_margin(strength)
                
            # Record category-specific games for both models
            model_a.record_game(score_a, win_margin, category)
            model_b.record_game(1.0 - score_a, win_margin, category)
            categories_processed += 1
            
        # Record overall games based on overall_eq category
        overall_winner, overall_strength = parse_evaluation_score(evaluation.get("overall_eq", ""))
        if overall_winner:
            if overall_winner == model_a_id:
                overall_score = strength_to_score(overall_strength)
                overall_margin = self.calculate_win_margin(overall_strength)
            else:
                overall_score = 1.0 - strength_to_score(overall_strength)
                overall_margin = self.calculate_win_margin(overall_strength)
                
            # Record overall game for both models
            model_a.record_game(overall_score, overall_margin, "overall")
            model_b.record_game(1.0 - overall_score, overall_margin, "overall")
        else:
            logging.debug(f"Skipping overall_eq - could not parse score: {evaluation.get('overall_eq', 'missing')}")
            
        if categories_processed == 0:
            logging.debug(f"Skipping record - no valid categories processed: row_index={record.get('row_index', 'unknown')}")
            return False
            
        # Store comparison for history
        comparison_record = {
            "id": comparison_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_a": model_a_id,
            "model_b": model_b_id, 
            "evaluation": evaluation,
            "metadata": {
                "conversation_a": record.get("conversation_a_metadata", {}),
                "conversation_b": record.get("conversation_b_metadata", {})
            }
        }
        self.comparisons.append(comparison_record)
        self.processed_comparisons.add(comparison_id)
        
        return True
    
    def update_model_strengths(self, category: str = "overall"):
        """Update model strengths using Bradley-Terry estimation."""
        # Collect comparisons for this category
        estimator = BradleyTerryEstimator()
        
        for comp in self.comparisons:
            evaluation = comp.get("evaluation", {})
            
            if category == "overall":
                if "overall_eq" not in evaluation:
                    continue
                winner, strength = parse_evaluation_score(evaluation["overall_eq"])
            else:
                if category not in evaluation:
                    continue
                winner, strength = parse_evaluation_score(evaluation[category])
            
            if winner is None:
                continue
                
            model_a = comp["model_a"]
            model_b = comp["model_b"]
            
            # Calculate score for model A
            if winner == model_a:
                score_a = strength_to_score(strength)
            else:
                score_a = 1.0 - strength_to_score(strength)
                
            estimator.add_comparison(model_a, model_b, score_a)
        
        # Estimate strengths
        estimated_strengths = estimator.estimate_strengths()
        
        # Update model strengths
        for model_id, strength in estimated_strengths.items():
            if model_id in self.models:
                self.models[model_id].update_strength(strength, category)
        
        return estimated_strengths
    
    def update_all_strengths(self):
        """Update strengths for all categories."""
        logging.info("Updating Bradley-Terry strengths for all categories...")
        
        # Update overall strengths
        self.update_model_strengths("overall")
        
        # Update category strengths
        for category in EVALUATION_CATEGORIES:
            self.update_model_strengths(category)
    
    def get_leaderboard(self, category: str = "overall") -> List[Dict[str, Any]]:
        """Get ranked leaderboard for a category."""
        if category == "overall":
            strengths = {model.model_id: model.strength for model in self.models.values()}
            games = {model.model_id: model.games_played for model in self.models.values()}
            win_rates = {model.model_id: model.win_rate() for model in self.models.values()}
        else:
            strengths = {model.model_id: model.category_strengths[category] for model in self.models.values()}
            games = {model.model_id: model.category_games[category] for model in self.models.values()}
            win_rates = {model.model_id: model.category_win_rate(category) for model in self.models.values()}
        
        # Sort by strength (descending)
        sorted_models = sorted(strengths.keys(), key=lambda x: strengths[x], reverse=True)
        
        leaderboard = []
        for rank, model_id in enumerate(sorted_models, 1):
            strength = strengths[model_id]
            rating = strength_to_rating(strength)
            
            # Calculate confidence intervals
            ci_low, ci_high = calculate_confidence_interval(strength, games[model_id])
            ci_low_rating = strength_to_rating(ci_low)
            ci_high_rating = strength_to_rating(ci_high)
            
            leaderboard.append({
                "rank": rank,
                "model_id": model_id,
                "strength": round(strength, 6),
                "rating": round(rating, 2),
                "ci_low": round(ci_low, 6),
                "ci_high": round(ci_high, 6),
                "ci_low_rating": round(ci_low_rating, 2),
                "ci_high_rating": round(ci_high_rating, 2),
                "games_played": games[model_id],
                "win_rate": round(win_rates[model_id], 3)
            })
            
        return leaderboard
    
    def save_state(self, filepath: Path):
        """Save Bradley-Terry system state to JSON file."""
        # Generate leaderboards
        leaderboards = {"overall": self.get_leaderboard("overall")}
        for cat in EVALUATION_CATEGORIES:
            leaderboards[cat] = self.get_leaderboard(cat)
        
        state = {
            "metadata": {
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "total_comparisons": len(self.comparisons),
                "method": "Bradley-Terry",
                "has_numpy": HAS_NUMPY,
                "has_scipy": HAS_SCIPY
            },
            "models": {model_id: model.to_dict() for model_id, model in self.models.items()},
            "comparisons": self.comparisons,
            "leaderboards": leaderboards
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved Bradley-Terry state to {filepath}")
        
    def load_state(self, filepath: Path):
        """Load Bradley-Terry system state from JSON file."""
        if not filepath.exists():
            logging.warning(f"Bradley-Terry state file not found: {filepath}")
            return
            
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
            
        # Restore models
        for model_id, model_data in state.get("models", {}).items():
            model = BTModel(model_id, model_data.get("initial_strength", DEFAULT_STRENGTH))
            model.strength = model_data.get("strength", DEFAULT_STRENGTH)
            model.games_played = model_data.get("games_played", 0)
            model.wins = model_data.get("wins", 0)
            model.losses = model_data.get("losses", 0)
            model.draws = model_data.get("draws", 0)
            model.strength_history = model_data.get("strength_history", [DEFAULT_STRENGTH])
            model.category_strengths = model_data.get("category_strengths", {cat: DEFAULT_STRENGTH for cat in EVALUATION_CATEGORIES})
            model.category_games = model_data.get("category_games", {cat: 0 for cat in EVALUATION_CATEGORIES})
            model.win_margins = model_data.get("win_margins", [])
            self.models[model_id] = model
            
        # Restore comparisons
        self.comparisons = state.get("comparisons", [])
        self.processed_comparisons = {comp["id"] for comp in self.comparisons}
        
        logging.info(f"Loaded Bradley-Terry state from {filepath}")
        logging.info(f"Models: {len(self.models)}, Comparisons: {len(self.comparisons)}")

# ───────────────────────────── Analysis Functions ───────────────────────── #

def save_leaderboard_data(bt_system: BradleyTerryEvaluationSystem, output_file: Path):
    """Save leaderboard data in format suitable for the UI."""
    # Map category names to match the UI expectations
    category_mapping = {
        "overall_eq": "overall",
        "empathic_responsiveness": "empathy",
        "emotional_handling_and_insight": "emotional", 
        "personalization_and_contextual_adaptation": "personalization",
        "conversational_fluency_and_naturalness": "conversational",
        "instruction_following_and_safety": "safety"
    }
    
    ui_data = {}
    
    for category, ui_category in category_mapping.items():
        leaderboard = bt_system.get_leaderboard(category)
        ui_models = []
        
        for entry in leaderboard:
            model_data = {
                "name": entry["model_id"],
                "strength": entry["strength"],
                "rating": entry["rating"],
                "ci": [entry["ci_low_rating"], entry["ci_high_rating"]],
                "games": entry["games_played"],
                "winRate": entry["win_rate"] * 100 if entry["win_rate"] > 0 else 0.0
            }
            
            # Add wins/losses for overall category
            if category == "overall":
                model = bt_system.models[entry["model_id"]]
                model_data["wins"] = model.wins
                model_data["losses"] = model.losses
                model_data["draws"] = model.draws
            
            ui_models.append(model_data)
        
        ui_data[ui_category] = ui_models
    
    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ui_data, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Saved leaderboard data to {output_file}")

def print_detailed_analysis(bt_system: BradleyTerryEvaluationSystem):
    """Print detailed analysis of Bradley-Terry model results."""
    print("="*80)
    print("BRADLEY-TERRY MODEL EVALUATION ANALYSIS")
    print("="*80)
    
    # Overall leaderboard
    print("\nOVERALL LEADERBOARD")
    print("-" * 70)
    overall_board = bt_system.get_leaderboard("overall")
    for entry in overall_board:
        print(f"{entry['rank']}. {entry['model_id']:>6} | "
              f"Strength: {entry['strength']:>8.4f} | "
              f"Rating: {entry['rating']:>7.1f} | "
              f"CI: [{entry['ci_low_rating']:>6.1f}, {entry['ci_high_rating']:>6.1f}] | "
              f"Games: {entry['games_played']:>3} | "
              f"Win Rate: {entry['win_rate']:.1%}")
    
    # Category breakdowns
    for category in EVALUATION_CATEGORIES:
        print(f"\n{CATEGORY_DISPLAY_NAMES.get(category, category).upper()}")
        print("-" * 70)
        cat_board = bt_system.get_leaderboard(category)
        for entry in cat_board:
            print(f"{entry['rank']}. {entry['model_id']:>6} | "
                  f"Strength: {entry['strength']:>8.4f} | "
                  f"Rating: {entry['rating']:>7.1f} | "
                  f"CI: [{entry['ci_low_rating']:>6.1f}, {entry['ci_high_rating']:>6.1f}] | "
                  f"Games: {entry['games_played']:>3} | "
                  f"Accuracy: {entry['win_rate']:.1%}")
                  
    # Model details
    print(f"\nMODEL DETAILS")
    print("-" * 70)
    for model_id, model in bt_system.models.items():
        print(f"\n{model_id}:")
        print(f"  Overall Strength: {model.strength:.4f}")
        print(f"  Overall Rating: {model.get_rating():.1f}")
        ci_low, ci_high = model.confidence_interval()
        ci_low_rating = strength_to_rating(ci_low)
        ci_high_rating = strength_to_rating(ci_high)
        print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}] ([{ci_low_rating:.1f}, {ci_high_rating:.1f}])")
        print(f"  Games Played: {model.games_played}")
        print(f"  Record: {model.wins}W-{model.losses}L-{model.draws}D")
        print(f"  Win Rate: {model.win_rate():.1%}")
        print(f"  Avg Win Margin: {model.average_win_margin():.3f}")
        print(f"  Strength Change: {model.strength / model.initial_strength:.3f}x")
        
        print("  Category Strengths:")
        for cat, strength in model.category_strengths.items():
            games = model.category_games[cat]
            rating = strength_to_rating(strength)
            accuracy = model.category_win_rate(cat)
            display_name = CATEGORY_DISPLAY_NAMES.get(cat, cat)
            print(f"    {display_name:<35}: {strength:>8.4f} ({rating:>6.1f}, {games} games, {accuracy:.1%} accuracy)")

def analyze_by_metadata(bt_system: BradleyTerryEvaluationSystem):
    """Analyze performance by conversation metadata."""
    print(f"\nMETADATA ANALYSIS")
    print("-" * 50)
    
    # Group by emotion type and problem type
    emotion_performance = defaultdict(lambda: defaultdict(list))
    problem_performance = defaultdict(lambda: defaultdict(list))
    
    for comp in bt_system.comparisons:
        metadata_a = comp.get("metadata", {}).get("conversation_a", {})
        emotion_type = metadata_a.get("emotion_type", "unknown")
        problem_type = metadata_a.get("problem_type", "unknown")
        
        evaluation = comp.get("evaluation", {})
        for category in EVALUATION_CATEGORIES:
            if category in evaluation:
                winner, strength = parse_evaluation_score(evaluation[category])
                if winner:
                    emotion_performance[emotion_type][category].append((winner, strength))
                    problem_performance[problem_type][category].append((winner, strength))
    
    # Print emotion type analysis
    print(f"\nPerformance by Emotion Type:")
    for emotion_type, categories in emotion_performance.items():
        print(f"\n{emotion_type.title()}:")
        for category, results in categories.items():
            if results:
                # Count wins for each model
                model_wins = defaultdict(int)
                for winner, _ in results:
                    model_wins[winner] += 1
                
                total = len(results)
                print(f"  {CATEGORY_DISPLAY_NAMES.get(category, category):<35}: ", end="")
                win_strs = [f"{model} {wins}/{total} ({wins/total:.1%})" for model, wins in model_wins.items()]
                print(", ".join(win_strs))
    
    # Print problem type analysis  
    print(f"\nPerformance by Problem Type:")
    for problem_type, categories in problem_performance.items():
        print(f"\n{problem_type.title()}:")
        for category, results in categories.items():
            if results:
                # Count wins for each model
                model_wins = defaultdict(int)
                for winner, _ in results:
                    model_wins[winner] += 1
                
                total = len(results)
                print(f"  {CATEGORY_DISPLAY_NAMES.get(category, category):<35}: ", end="")
                win_strs = [f"{model} {wins}/{total} ({wins/total:.1%})" for model, wins in model_wins.items()]
                print(", ".join(win_strs))

def find_evaluation_files(folder_path: Path, pattern: str = "**/*.jsonl") -> List[Path]:
    """Find all evaluation files in the given folder matching the pattern."""
    if not folder_path.exists():
        logging.error(f"Folder not found: {folder_path}")
        return []
        
    if not folder_path.is_dir():
        logging.error(f"Not a directory: {folder_path}")
        return []
    
    # Find all matching files recursively
    files = list(folder_path.glob(pattern))
    
    if not files:
        logging.warning(f"No files matching pattern '{pattern}' found in {folder_path}")
        return []
        
    logging.info(f"Found {len(files)} evaluation files in {folder_path}")
    for file in files:
        logging.debug(f"Found evaluation file: {file}")
        
    return sorted(files)  # Sort for consistent ordering before randomization

def load_and_randomize_evaluations(evaluation_files: List[Path]) -> List[Dict[str, Any]]:
    """Load evaluations from multiple files, concatenate, and randomize."""
    all_evaluations = []
    error_count = 0
    
    for eval_file in evaluation_files:
        if not eval_file.exists():
            logging.error(f"Evaluation file not found: {eval_file}")
            continue
            
        logging.info(f"Loading evaluations from {eval_file}")
        with open(eval_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                    
                try:
                    record = json.loads(line)
                    # Add source file info to record metadata
                    if "metadata" not in record:
                        record["metadata"] = {}
                    record["metadata"]["source_file"] = str(eval_file)
                    all_evaluations.append(record)
                    
                except json.JSONDecodeError as e:
                    logging.warning(f"Failed to parse line {line_num} in {eval_file}: {e}")
                    error_count += 1
                except Exception as e:
                    logging.error(f"Error processing line {line_num} in {eval_file}: {e}")
                    error_count += 1
    
    # Randomize the order
    random.shuffle(all_evaluations)
    
    logging.info(f"Loaded {len(all_evaluations)} total evaluations from {len(evaluation_files)} files")
    if error_count > 0:
        logging.warning(f"Encountered {error_count} errors during loading")
        
    return all_evaluations

def run_bradley_terry_evaluation(
    evaluation_source: Union[Path, List[Path]],
    bt_state_file: Path,
    load_existing: bool = True,
    save_results: bool = True,
    show_analysis: bool = True,
    show_metadata: bool = False,
    leaderboard_output: Optional[Path] = None,
    random_seed: Optional[int] = None,
    file_pattern: str = "**/*.jsonl"
):
    """Run Bradley-Terry evaluation on pairwise comparison results."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
        logging.info(f"Set random seed to {random_seed}")
    
    # Get list of evaluation files
    if isinstance(evaluation_source, list):
        evaluation_files = evaluation_source
    else:
        evaluation_files = find_evaluation_files(evaluation_source, file_pattern)
        if not evaluation_files:
            return None
    
    # Initialize Bradley-Terry system
    bt_system = BradleyTerryEvaluationSystem()
    
    # Load existing state if requested
    if load_existing and bt_state_file.exists():
        bt_system.load_state(bt_state_file)
    
    # Load and randomize evaluations from all files
    all_evaluations = load_and_randomize_evaluations(evaluation_files)
    
    # Process all evaluations
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for record_num, record in enumerate(all_evaluations, 1):
        try:
            if bt_system.process_evaluation_record(record):
                processed_count += 1
            else:
                skipped_count += 1
                
            if processed_count % 100 == 0:
                logging.info(f"Processed {processed_count} evaluations, skipped {skipped_count}...")
                
        except Exception as e:
            logging.error(f"Error processing record {record_num}: {e}")
            logging.debug(f"Record data: {record}")
            error_count += 1
            continue
    
    logging.info(f"Processed {processed_count} total evaluations, skipped {skipped_count} evaluations")
    if skipped_count > 0:
        logging.info(f"Use --verbose flag to see detailed skip reasons")
    if error_count > 0:
        logging.warning(f"Encountered {error_count} errors during processing")
    
    # Update model strengths using Bradley-Terry estimation
    bt_system.update_all_strengths()
    
    # Save results
    if save_results:
        bt_system.save_state(bt_state_file)
        
    # Save leaderboard data for UI
    if leaderboard_output:
        save_leaderboard_data(bt_system, leaderboard_output)
        
    # Show analysis
    if show_analysis:
        print_detailed_analysis(bt_system)
        
    if show_metadata:
        analyze_by_metadata(bt_system)
        
    return bt_system

def parse_args():
    parser = argparse.ArgumentParser(
        description="Bradley-Terry Model Evaluation for Dialogue Systems using Pairwise Comparisons"
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--files", 
        type=Path,
        nargs='+',
        help="One or more JSONL files with pairwise evaluation results"
    )
    input_group.add_argument(
        "--folder",
        type=Path,
        help="Folder containing JSONL files with pairwise evaluation results"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/*.json*",
        help="File pattern to match when searching folder (default: **/*.json*)"
    )
    parser.add_argument(
        "--bt-state", 
        type=Path,
        default=Path("bradley_terry_state.json"),
        help="Bradley-Terry state file to load/save (default: bradley_terry_state.json)"
    )
    parser.add_argument(
        "--no-load",
        action="store_true",
        help="Don't load existing Bradley-Terry state"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset Bradley-Terry state (ignore existing state file)"
    )
    parser.add_argument(
        "--no-save", 
        action="store_true",
        help="Don't save Bradley-Terry state after processing"
    )
    parser.add_argument(
        "--leaderboard-output", 
        type=Path,
        help="Path to save leaderboard JSON data for UI (optional)"
    )
    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="Don't show detailed analysis"
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Show metadata analysis (emotion/problem type)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (shows detailed skip reasons)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        help="Random seed for reproducible evaluation order"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Setup logging level
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Show available libraries
    logging.info(f"NumPy available: {HAS_NUMPY}")
    logging.info(f"SciPy available: {HAS_SCIPY}")
    
    # Run Bradley-Terry evaluation
    try:
        bt_system = run_bradley_terry_evaluation(
            evaluation_source=args.files if args.files else args.folder,
            bt_state_file=args.bt_state,
            load_existing=not args.no_load and not args.reset,
            save_results=not args.no_save,
            show_analysis=not args.no_analysis,
            show_metadata=args.metadata,
            leaderboard_output=args.leaderboard_output,
            random_seed=args.random_seed,
            file_pattern=args.pattern if args.folder else "**/*.jsonl"
        )
        
        if bt_system is None:
            sys.exit(1)
            
        logging.info("Bradley-Terry evaluation completed successfully")
        
    except KeyboardInterrupt:
        logging.info("Bradley-Terry evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Bradley-Terry evaluation failed: {e}")
        sys.exit(1) 