from __future__ import annotations

import dataclasses
import random
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


Signature = Tuple[float, ...]
ScoresPerTest = Mapping[str, float]


def _reduce_score(scores_per_test: ScoresPerTest) -> float:
    values = [scores_per_test[k] for k in scores_per_test.keys()]
    return float(sum(values) / len(values)) if values else float('-inf')


def _get_signature(scores_per_test: ScoresPerTest) -> Signature:
    return tuple(scores_per_test[k] for k in sorted(scores_per_test.keys()))


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    if not np.all(np.isfinite(logits)):
        logits = np.array([0.0])
    if not np.issubdtype(logits.dtype, np.floating):
        logits = np.array(logits, dtype=np.float32)
    if logits.size == 0:
        logits = np.array([0.0], dtype=np.float32)
    probs = np.exp(logits / max(temperature, 1e-6))
    probs_sum = probs.sum()
    if probs_sum <= 0:
        probs = np.ones_like(probs)
        probs_sum = probs.sum()
    probs = probs / probs_sum
    # ensure numerical stability
    idx = int(np.argmax(probs))
    probs[idx] = 1.0 - (probs.sum() - probs[idx])
    return probs


@dataclasses.dataclass(frozen=True)
class Prompt:
    memory_success: List[Dict[str, Any]]
    memory_failure: List[Dict[str, Any]]
    island_id: int


class Cluster:
    def __init__(self, score: float, candidate: Dict[str, Any]):
        self._score = float(score)
        self._candidates: List[Dict[str, Any]] = [candidate]

    @property
    def score(self) -> float:
        return self._score

    def register(self, candidate: Dict[str, Any]) -> None:
        self._candidates.append(candidate)

    def sample(self) -> Dict[str, Any]:
        return random.choice(self._candidates)


class Island:
    def __init__(self, functions_per_prompt: int, temp_init: float, temp_period: int, memory_max_items: int = 50, memory_top_k_success: int = 10, memory_bottom_k_failure: int = 10):
        self._clusters: Dict[Signature, Cluster] = {}
        self._num_items: int = 0
        self._functions_per_prompt = functions_per_prompt
        self._temp_init = temp_init
        self._temp_period = max(1, temp_period)
        self._memory_max_items = memory_max_items
        self._memory_top_k_success = memory_top_k_success
        self._memory_bottom_k_failure = memory_bottom_k_failure
        
        # Island-specific memory buffers for deduplication
        self._unique_candidates: Dict[str, Dict[str, Any]] = {}  # formula -> candidate
        self._iteration_history: List[Dict[str, Any]] = []  # Track candidates by iteration
        self._refresh_interval = 50  # Refresh every 15 iterations
        self._last_refresh_iteration = 0

    def register_candidate(self, candidate: Dict[str, Any], scores_per_test: ScoresPerTest, iteration: int = 0) -> None:
        # Track unique candidates for deduplication
        formula = candidate.get('formula', '')
        compound = candidate.get('compound', '')
        candidate_key = formula or compound or str(hash(str(candidate)))
        
        # Store in unique candidates dict (keeps best version if duplicate)
        if candidate_key not in self._unique_candidates:
            self._unique_candidates[candidate_key] = candidate
        else:
            # Keep the better scoring candidate
            current_score = self._unique_candidates[candidate_key].get('score', float('-inf'))
            new_score = candidate.get('score', float('-inf'))
            if new_score > current_score:
                self._unique_candidates[candidate_key] = candidate
        
        # Track iteration history for periodic refresh
        candidate_with_iteration = candidate.copy()
        candidate_with_iteration['_iteration'] = iteration
        self._iteration_history.append(candidate_with_iteration)
        
        # Register in clusters as before
        sig = _get_signature(scores_per_test)
        if sig not in self._clusters:
            self._clusters[sig] = Cluster(_reduce_score(scores_per_test), candidate)
        else:
            self._clusters[sig].register(candidate)
        self._num_items += 1

    def get_prompt_memories(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        # Split into successes (positive scores) and failures (negative scores)
        all_items: List[Tuple[float, Dict[str, Any]]] = []
        for cl in self._clusters.values():
            # Use individual candidate scores, not cluster scores
            for c in cl._candidates:
                candidate_score = float(c.get('score', 0.0))
                all_items.append((candidate_score, c))
        
        if not all_items:
            return [], []
        
        # Sort by individual candidate scores (high to low)
        all_items.sort(key=lambda x: x[0], reverse=True)
        
        # Proper success/failure partitioning: successes have positive scores, failures have negative scores
        successes = [c for s, c in all_items if s > 0.0]  # Only positive scores are successes
        failures = [c for s, c in all_items if s <= 0.0]  # Zero and negative scores are failures
        
        # Limit to top K successes and bottom K failures
        successes = successes[:min(self._memory_top_k_success, len(successes))]
        failures = failures[:min(self._memory_bottom_k_failure, len(failures))]
        
        return successes, failures

    def refresh_memory_if_needed(self, current_iteration: int) -> None:
        """Refresh memory by removing candidates from previous 15 iterations"""
        if current_iteration - self._last_refresh_iteration >= self._refresh_interval:
            print(f"[Agent] Refreshing island memory at iteration {current_iteration}")
            
            # Remove candidates from previous 15 iterations
            cutoff_iteration = current_iteration - self._refresh_interval
            self._iteration_history = [
                c for c in self._iteration_history 
                if c.get('_iteration', 0) > cutoff_iteration
            ]
            
            # Rebuild unique candidates from remaining history
            self._unique_candidates = {}
            for candidate in self._iteration_history:
                formula = candidate.get('formula', '')
                compound = candidate.get('compound', '')
                candidate_key = formula or compound or str(hash(str(candidate)))
                
                if candidate_key not in self._unique_candidates:
                    self._unique_candidates[candidate_key] = candidate
                else:
                    current_score = self._unique_candidates[candidate_key].get('score', float('-inf'))
                    new_score = candidate.get('score', float('-inf'))
                    if new_score > current_score:
                        self._unique_candidates[candidate_key] = candidate
            
            self._last_refresh_iteration = current_iteration
            print(f"[Agent] Island memory refreshed: {len(self._unique_candidates)} unique candidates remaining")

    def get_unique_examples(self, max_examples: int = 20) -> Dict[str, List[Dict[str, Any]]]:
        """Get unique examples for evolution, split by success/failure"""
        # Get all unique candidates
        all_candidates = list(self._unique_candidates.values())
        
        if not all_candidates:
            return {'success': [], 'failure': []}
        
        # Sort by score
        all_candidates.sort(key=lambda c: c.get('score', 0), reverse=True)
        
        # Split into success/failure
        successes = [c for c in all_candidates if c.get('score', 0) > 0]
        failures = [c for c in all_candidates if c.get('score', 0) <= 0]
        
        # Limit to max_examples
        successes = successes[:max_examples]
        failures = failures[:max_examples]
        
        return {'success': successes, 'failure': failures}

    def is_duplicate(self, candidate: Dict[str, Any]) -> bool:
        """Check if candidate is a duplicate of existing ones"""
        formula = candidate.get('formula', '')
        compound = candidate.get('compound', '')
        candidate_key = formula or compound or str(hash(str(candidate)))
        return candidate_key in self._unique_candidates

    def sample_clusters_indices(self) -> List[int]:
        signatures = list(self._clusters.keys())
        if not signatures:
            return []
        scores = np.array([self._clusters[s].score for s in signatures], dtype=np.float32)
        # Use constant temperature of 0.8 for balanced exploration-exploitation
        temperature = 0.8
        probs = _softmax(scores, max(temperature, 1e-6))
        k = min(len(signatures), self._functions_per_prompt)
        idx = np.random.choice(len(signatures), size=k, p=probs, replace=False)
        return list(idx)


class ExperienceBuffer:
    def __init__(self, num_islands: int, functions_per_prompt: int = 4, temp_init: float = 1.0, temp_period: int = 10, reset_period_seconds: int = 120, max_items_per_island: int = 50, top_k_success: int = 10, bottom_k_failure: int = 10):
        self._islands: List[Island] = [Island(functions_per_prompt, temp_init, temp_period, max_items_per_island, top_k_success, bottom_k_failure) for _ in range(max(1, num_islands))]
        self._best_score_per_island: List[float] = [-float('inf')] * len(self._islands)
        self._best_candidate_per_island: List[Dict[str, Any] | None] = [None] * len(self._islands)
        self._best_scores_per_test_per_island: List[Dict[str, float] | None] = [None] * len(self._islands)
        self._last_reset_time: float = time.time()
        self._reset_period_seconds = reset_period_seconds
        self._max_items_per_island = max_items_per_island
        self._top_k_success = top_k_success
        self._bottom_k_failure = bottom_k_failure

    def register(self, island_id: int, candidate: Dict[str, Any], scores_per_test: ScoresPerTest, iteration: int = 0) -> None:
        self._islands[island_id].register_candidate(candidate, scores_per_test, iteration)
        score = _reduce_score(scores_per_test)
        if score > self._best_score_per_island[island_id]:
            self._best_candidate_per_island[island_id] = candidate
            self._best_scores_per_test_per_island[island_id] = scores_per_test
            self._best_score_per_island[island_id] = score
        self._maybe_reset()

    def seed_all(self, candidates: List[Dict[str, Any]], score_key: str = 'score') -> None:
        if not candidates:
            return
        print(f"[Agent] SEEDING {len(candidates)} candidates to ALL {len(self._islands)} islands:")
        # Seed the SAME candidates to ALL islands
        for c in candidates:
            score = float(c.get(score_key, 0.0))
            formula = c.get('formula', 'Unknown')
            # Create a proper scores mapping for the candidate
            scores_per_test = {'total': score}
            # Add individual property scores if available
            if 'predictions' in c:
                for prop, value in c['predictions'].items():
                    if isinstance(value, (int, float)):
                        scores_per_test[prop] = float(value)
            
            # Register to ALL islands
            for island_id in range(len(self._islands)):
                self.register(island_id, c, scores_per_test)
            print(f"  - Seeded {formula} (score: {score:.3f}) to ALL islands")

    def get_prompt(self, iteration: int = 0) -> Prompt:
        island_id = int(np.random.randint(len(self._islands)))
        # Refresh memory if needed
        self._islands[island_id].refresh_memory_if_needed(iteration)
        # Prune memory periodically per island size
        self._prune_island(island_id)
        successes, failures = self._islands[island_id].get_prompt_memories()
        return Prompt(successes, failures, island_id)

    def get_unique_examples_for_evolution(self, island_id: int, max_examples: int = 20) -> Dict[str, List[Dict[str, Any]]]:
        """Get unique examples from specific island for evolution"""
        return self._islands[island_id].get_unique_examples(max_examples)

    def is_duplicate_candidate(self, island_id: int, candidate: Dict[str, Any]) -> bool:
        """Check if candidate is duplicate in specific island"""
        return self._islands[island_id].is_duplicate(candidate)

    def _prune_island(self, island_id: int) -> None:
        island = self._islands[island_id]
        # Flatten items with cluster scores
        items: List[Tuple[float, Dict[str, Any]]] = []
        for cl in island._clusters.values():
            items.extend([(cl.score, c) for c in cl._candidates])
        if len(items) <= self._max_items_per_island:
            return
        items.sort(key=lambda x: x[0], reverse=True)
        keep: List[Dict[str, Any]] = [c for _, c in items[: self._top_k_success]] + [c for _, c in items[-self._bottom_k_failure :]]
        # Rebuild clusters from kept items
        island._clusters = {}
        island._num_items = 0
        for c in keep:
            score = float(c.get('score', 0.0))
            island.register_candidate(c, {'total': score})

    def _maybe_reset(self) -> None:
        if time.time() - self._last_reset_time < self._reset_period_seconds:
            return
        self._last_reset_time = time.time()
        scores = np.array(self._best_score_per_island, dtype=np.float32)
        order = np.argsort(scores)
        half = len(order) // 2
        to_reset = order[:half]
        keep = order[half:]
        if keep.size == 0:
            return
        founder = int(np.random.choice(keep))
        
        print(f"[Agent] ISLAND RESET triggered!")
        print(f"  - Islands to reset: {to_reset.tolist()}")
        print(f"  - Islands to keep: {keep.tolist()}")
        print(f"  - Founder island: {founder} (score: {self._best_score_per_island[founder]:.3f})")
        
        # Reset by reinitializing islands list entries and seeding with founder
        for idx in to_reset:
            old_score = self._best_score_per_island[idx]
            self._islands[idx] = Island(
                self._islands[founder]._functions_per_prompt, 
                self._islands[founder]._temp_init, 
                self._islands[founder]._temp_period,
                self._max_items_per_island,
                self._top_k_success,
                self._bottom_k_failure
            )
            self._best_score_per_island[idx] = -float('inf')
            self._best_candidate_per_island[idx] = None
            self._best_scores_per_test_per_island[idx] = None
            
            # Seed reset island with founder's best candidate
            if self._best_candidate_per_island[founder] is not None:
                founder_candidate = self._best_candidate_per_island[founder]
                founder_scores = self._best_scores_per_test_per_island[founder]
                self.register(idx, founder_candidate, founder_scores)
                print(f"  - Reset island {idx} (old best score: {old_score:.3f}) and seeded with founder {founder}'s best candidate")
            else:
                print(f"  - Reset island {idx} (old best score: {old_score:.3f}) but no founder candidate available")

