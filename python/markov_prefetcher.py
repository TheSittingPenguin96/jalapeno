# python/markov_prefetcher.py
"""
Python interface for Markov Prefetcher
"""

import ctypes
import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from enum import IntEnum

@dataclass
class ContextFeatures:
    """Features for context-aware prefetching"""
    sequence_length: float = 0.0
    attention_pattern_entropy: float = 0.0
    layer_diversity: float = 0.0
    temporal_locality: float = 0.0
    model_size: float = 0.0
    model_type: str = ""
    memory_pressure: float = 0.0
    gpu_utilization: float = 0.0
    
    def to_vector(self) -> List[float]:
        """Convert to feature vector"""
        return [
            self.sequence_length,
            self.attention_pattern_entropy,
            self.layer_diversity,
            self.temporal_locality,
            self.model_size,
            self.memory_pressure,
            self.gpu_utilization,
        ]

class MarkovPrefetcher:
    """
    Python wrapper for Markov Chain Prefetcher.
    
    Learns access patterns and predicts future layer accesses.
    """
    
    def __init__(self, use_neural: bool = False, use_hierarchical: bool = False):
        """
        Initialize prefetcher.
        
        Args:
            use_neural: Use neural network for predictions
            use_hierarchical: Use hierarchical Markov chains
        """
        self.use_neural = use_neural
        self.use_hierarchical = use_hierarchical
        
        # Load native library
        self._load_native_lib()
        
        # Create native prefetcher
        self._native_handle = self._create_native_prefetcher()
        
        # Statistics
        self.stats = {
            'total_transitions': 0,
            'prediction_hits': 0,
            'prediction_misses': 0,
            'average_confidence': 0.0,
            'states_learned': 0,
        }
        
        # State tracking
        self.current_state = None
        self.state_history = []
        self.transition_history = []
        
    def _load_native_lib(self):
        """Load the native C++ library"""
        lib_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "build", "libjalapeno.so"
        )
        self._native = ctypes.CDLL(lib_path)
        
        # Define function signatures
        self._native.jalapeno_create_markov_prefetcher.argtypes = [
            ctypes.c_bool,  # use_neural
            ctypes.c_bool,  # use_hierarchical
        ]
        self._native.jalapeno_create_markov_prefetcher.restype = ctypes.c_void_p
        
        self._native.jalapeno_prefetcher_record.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.c_char_p,  # from_state
            ctypes.c_char_p,  # to_state
            ctypes.c_float,   # latency_ms
            ctypes.c_void_p,  # context features
        ]
        
        self._native.jalapeno_prefetcher_predict.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.c_char_p,  # current_state
            ctypes.c_void_p,  # context features
            ctypes.c_size_t,  # k predictions
            ctypes.POINTER(ctypes.c_char_p),  # predictions output
            ctypes.POINTER(ctypes.c_float),   # confidences output
        ]
        self._native.jalapeno_prefetcher_predict.restype = ctypes.c_size_t
        
        self._native.jalapeno_prefetcher_get_stats.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.c_void_p,  # stats struct
        ]
    
    def _create_native_prefetcher(self):
        """Create native prefetcher instance"""
        handle = self._native.jalapeno_create_markov_prefetcher(
            self.use_neural, self.use_hierarchical
        )
        if not handle:
            raise RuntimeError("Failed to create Markov prefetcher")
        return handle
    
    def record_transition(
        self,
        from_state: str,
        to_state: str,
        latency_ms: float = 0.0,
        context: Optional[ContextFeatures] = None
    ):
        """
        Record a transition between states.
        
        Args:
            from_state: Current state/segment
            to_state: Next state/segment accessed
            latency_ms: Transition latency (for learning)
            context: Context features for this transition
        """
        # Record in native
        context_ptr = self._context_to_native(context) if context else None
        
        self._native.jalapeno_prefetcher_record(
            self._native_handle,
            ctypes.c_char_p(from_state.encode()),
            ctypes.c_char_p(to_state.encode()),
            latency_ms,
            context_ptr
        )
        
        # Update Python state
        self.current_state = to_state
        self.state_history.append(to_state)
        self.transition_history.append((from_state, to_state, latency_ms))
        
        # Update statistics
        self.stats['total_transitions'] += 1
        self.stats['states_learned'] = len(set(self.state_history))
        
        # Cleanup if needed
        if context_ptr:
            # Free native context memory
            pass
    
    def predict(
        self,
        current_state: str,
        context: Optional[ContextFeatures] = None,
        k: int = 3
    ) -> Tuple[List[str], List[float]]:
        """
        Predict next states.
        
        Args:
            current_state: Current state/segment
            context: Current context features
            k: Number of predictions to return
            
        Returns:
            Tuple of (predicted_states, confidences)
        """
        # Prepare output buffers
        predictions = (ctypes.c_char_p * k)()
        confidences = (ctypes.c_float * k)()
        
        # Convert context
        context_ptr = self._context_to_native(context) if context else None
        
        # Call native prediction
        num_pred = self._native.jalapeno_prefetcher_predict(
            self._native_handle,
            ctypes.c_char_p(current_state.encode()),
            context_ptr,
            k,
            predictions,
            confidences
        )
        
        # Convert results
        predicted_states = []
        confidence_scores = []
        
        for i in range(num_pred):
            if predictions[i]:
                predicted_states.append(predictions[i].decode())
                confidence_scores.append(confidences[i])
        
        # Update prediction statistics
        # (accuracy would be updated when we know actual next state)
        
        return predicted_states, confidence_scores
    
    def predict_sequence(
        self,
        start_state: str,
        length: int,
        context: Optional[ContextFeatures] = None
    ) -> List[str]:
        """
        Predict a sequence of states.
        
        Args:
            start_state: Starting state
            length: Length of sequence to predict
            context: Context features
            
        Returns:
            Predicted state sequence
        """
        sequence = []
        current = start_state
        
        for _ in range(length):
            predictions, confidences = self.predict(current, context, 1)
            if predictions:
                next_state = predictions[0]
                sequence.append(next_state)
                current = next_state
            else:
                break
        
        return sequence
    
    def update_prediction_accuracy(
        self,
        predicted_state: str,
        actual_state: str
    ):
        """
        Update prediction accuracy statistics.
        
        Args:
            predicted_state: State that was predicted
            actual_state: State that actually occurred
        """
        if predicted_state == actual_state:
            self.stats['prediction_hits'] += 1
        else:
            self.stats['prediction_misses'] += 1
        
        # Update average confidence
        total = self.stats['prediction_hits'] + self.stats['prediction_misses']
        self.stats['prediction_accuracy'] = (
            self.stats['prediction_hits'] / total if total > 0 else 0.0
        )
    
    def get_most_likely_path(
        self,
        start_state: str,
        end_state: str,
        max_steps: int = 10
    ) -> List[str]:
        """
        Get most likely path between two states.
        
        Args:
            start_state: Starting state
            end_state: Target state
            max_steps: Maximum steps to consider
            
        Returns:
            Most likely path as list of states
        """
        # This would call native function
        # For now, simulate with simple BFS
        return self._find_path_bfs(start_state, end_state, max_steps)
    
    def _find_path_bfs(
        self,
        start: str,
        end: str,
        max_depth: int
    ) -> List[str]:
        """Simple BFS path finding (placeholder)"""
        # In production, would use native Markov chain
        if start == end:
            return [start]
        
        # Get predictions from current state
        predictions, _ = self.predict(start, None, 5)
        
        for next_state in predictions:
            if next_state == end:
                return [start, end]
            
            if max_depth > 1:
                subpath = self._find_path_bfs(next_state, end, max_depth - 1)
                if subpath:
                    return [start] + subpath
        
        return []
    
    def save_model(self, path: str):
        """Save learned model to file"""
        # Convert path
        self._native.jalapeno_prefetcher_save_model(
            self._native_handle,
            ctypes.c_char_p(path.encode())
        )
    
    def load_model(self, path: str):
        """Load model from file"""
        self._native.jalapeno_prefetcher_load_model(
            self._native_handle,
            ctypes.c_char_p(path.encode())
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get prefetcher statistics"""
        # Get native statistics
        class NativeStats(ctypes.Structure):
            _fields_ = [
                ("total_states", ctypes.c_size_t),
                ("total_transitions", ctypes.c_size_t),
                ("average_branching_factor", ctypes.c_float),
                ("average_transition_confidence", ctypes.c_float),
                ("prediction_accuracy", ctypes.c_float),
            ]
        
        stats = NativeStats()
        self._native.jalapeno_prefetcher_get_stats(
            self._native_handle, ctypes.byref(stats)
        )
        
        # Combine with Python stats
        result = {
            'total_states': stats.total_states,
            'total_transitions': stats.total_transitions,
            'average_branching_factor': stats.average_branching_factor,
            'average_transition_confidence': stats.average_transition_confidence,
            'prediction_accuracy': stats.prediction_accuracy,
            
            # Python-tracked stats
            'prediction_hits': self.stats['prediction_hits'],
            'prediction_misses': self.stats['prediction_misses'],
            'states_learned': self.stats['states_learned'],
            'current_state': self.current_state,
            'history_length': len(self.state_history),
        }
        
        return result
    
    def _context_to_native(self, context: ContextFeatures) -> ctypes.c_void_p:
        """Convert ContextFeatures to native structure (placeholder)"""
        # In production, would create native struct
        return ctypes.c_void_p()
    
    def train_neural_network(
        self,
        training_data: List[Tuple[str, str, ContextFeatures]],
        epochs: int = 10
    ):
        """
        Train the neural network component.
        
        Args:
            training_data: List of (from_state, to_state, context) tuples
            epochs: Number of training epochs
        """
        if not self.use_neural:
            print("Neural network not enabled")
            return
        
        # Convert training data to native format
        # This would train the neural Markov prefetcher
        pass
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, '_native_handle') and self._native_handle:
            # Destroy native prefetcher
            # self._native.jalapeno_destroy_prefetcher(self._native_handle)
            pass

class AdaptivePrefetcher:
    """
    Adaptive prefetcher that combines multiple strategies.
    
    Dynamically switches between Markov, neural, and other prefetching
    strategies based on workload characteristics.
    """
    
    def __init__(self):
        self.markov_prefetcher = MarkovPrefetcher(use_neural=True)
        self.strategy_weights = {
            'markov': 0.7,
            'neural': 0.3,
            'linear': 0.1,
        }
        
        self.strategy_performance = {
            'markov': 0.0,
            'neural': 0.0,
            'linear': 0.0,
        }
        
        self.current_strategy = 'markov'
        self.adaptation_interval = 100  # steps
        
    def predict(
        self,
        current_state: str,
        context: ContextFeatures,
        k: int = 3
    ) -> List[str]:
        """Predict using adaptive strategy"""
        # Get predictions from all strategies
        all_predictions = {}
        
        # Markov predictions
        markov_preds, markov_confs = self.markov_prefetcher.predict(
            current_state, context, k
        )
        for pred, conf in zip(markov_preds, markov_confs):
            all_predictions[pred] = all_predictions.get(pred, 0.0) + \
                                  conf * self.strategy_weights['markov']
        
        # Neural predictions (already included in markov if use_neural=True)
        # Could add other strategies here
        
        # Sort by combined score
        sorted_predictions = sorted(
            all_predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top k
        return [pred for pred, _ in sorted_predictions[:k]]
    
    def record_transition(
        self,
        from_state: str,
        to_state: str,
        context: ContextFeatures,
        latency_ms: float = 0.0
    ):
        """Record transition and adapt strategy"""
        # Record in underlying prefetcher
        self.markov_prefetcher.record_transition(
            from_state, to_state, latency_ms, context
        )
        
        # Update strategy performance
        self._update_strategy_performance(from_state, to_state)
        
        # Adapt strategy weights periodically
        if self.markov_prefetcher.stats['total_transitions'] % \
           self.adaptation_interval == 0:
            self._adapt_strategy_weights()
    
    def _update_strategy_performance(self, from_state: str, to_state: str):
        """Update performance tracking for each strategy"""
        # Get last predictions
        # Compare with actual transition
        # Update accuracy scores
        
        # Simplified: just track Markov performance
        stats = self.markov_prefetcher.get_statistics()
        self.strategy_performance['markov'] = stats['prediction_accuracy']
        
        # Neural performance would be tracked separately
    
    def _adapt_strategy_weights(self):
        """Adapt strategy weights based on performance"""
        # Increase weight of better performing strategies
        total_perf = sum(self.strategy_performance.values())
        if total_perf > 0:
            for strategy in self.strategy_weights:
                self.strategy_weights[strategy] = \
                    self.strategy_performance[strategy] / total_perf
        
        # Normalize weights
        total_weight = sum(self.strategy_weights.values())
        if total_weight > 0:
            for strategy in self.strategy_weights:
                self.strategy_weights[strategy] /= total_weight
        
        # Choose best strategy
        best_strategy = max(
            self.strategy_performance.items(),
            key=lambda x: x[1]
        )[0]
        self.current_strategy = best_strategy
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about current strategy"""
        return {
            'current_strategy': self.current_strategy,
            'strategy_weights': self.strategy_weights,
            'strategy_performance': self.strategy_performance,
            'markov_stats': self.markov_prefetcher.get_statistics(),
        }
