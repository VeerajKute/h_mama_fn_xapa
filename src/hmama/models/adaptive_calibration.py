"""
Adaptive calibration system for improving model accuracy dynamically
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime
from collections import defaultdict, deque
import logging

class AdaptiveCalibration:
    """
    Dynamic calibration system that learns from user feedback and improves predictions
    """
    
    def __init__(self, calibration_file: str = "checkpoints/calibration.json"):
        self.calibration_file = calibration_file
        self.feedback_history = deque(maxlen=1000)  # Keep last 1000 feedback items
        self.confidence_adjustments = {}
        self.word_weights = defaultdict(float)
        self.context_patterns = defaultdict(list)
        self.load_calibration_data()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_calibration_data(self):
        """Load existing calibration data"""
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, 'r') as f:
                    data = json.load(f)
                    self.feedback_history = deque(data.get('feedback_history', []), maxlen=1000)
                    self.confidence_adjustments = data.get('confidence_adjustments', {})
                    self.word_weights = defaultdict(float, data.get('word_weights', {}))
                    self.context_patterns = defaultdict(list, data.get('context_patterns', {}))
                self.logger.info(f"Loaded calibration data from {self.calibration_file}")
            except Exception as e:
                self.logger.error(f"Error loading calibration data: {e}")
    
    def save_calibration_data(self):
        """Save calibration data"""
        try:
            data = {
                'feedback_history': list(self.feedback_history),
                'confidence_adjustments': dict(self.confidence_adjustments),
                'word_weights': dict(self.word_weights),
                'context_patterns': dict(self.context_patterns),
                'last_updated': datetime.now().isoformat()
            }
            
            os.makedirs(os.path.dirname(self.calibration_file), exist_ok=True)
            with open(self.calibration_file, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.info(f"Saved calibration data to {self.calibration_file}")
        except Exception as e:
            self.logger.error(f"Error saving calibration data: {e}")
    
    def add_feedback(self, text: str, predicted_label: str, predicted_confidence: float, 
                    actual_label: str, user_id: Optional[str] = None):
        """Add user feedback to improve the model"""
        feedback = {
            'text': text,
            'predicted_label': predicted_label,
            'predicted_confidence': predicted_confidence,
            'actual_label': actual_label,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat()
        }
        
        self.feedback_history.append(feedback)
        
        # Update word weights based on feedback
        self._update_word_weights(text, predicted_label, actual_label)
        
        # Update confidence adjustments
        self._update_confidence_adjustments(predicted_confidence, predicted_label, actual_label)
        
        # Save updated calibration data
        self.save_calibration_data()
        
        self.logger.info(f"Added feedback: {predicted_label} -> {actual_label}")
    
    def _update_word_weights(self, text: str, predicted: str, actual: str):
        """Update word weights based on feedback"""
        words = text.lower().split()
        
        for word in words:
            # Clean word (remove punctuation)
            clean_word = ''.join(c for c in word if c.isalnum())
            if len(clean_word) < 3:  # Skip short words
                continue
                
            # Adjust weight based on correctness
            if predicted == actual:
                # Correct prediction - reinforce current weights
                self.word_weights[clean_word] *= 1.01
            else:
                # Incorrect prediction - adjust weights
                if actual == 'REAL' and predicted == 'FAKE':
                    # Model was too aggressive - reduce weight
                    self.word_weights[clean_word] *= 0.99
                elif actual == 'FAKE' and predicted == 'REAL':
                    # Model was too lenient - increase weight
                    self.word_weights[clean_word] *= 1.02
    
    def _update_confidence_adjustments(self, confidence: float, predicted: str, actual: str):
        """Update confidence adjustments based on feedback"""
        if predicted != actual:
            # Wrong prediction - adjust confidence threshold
            key = f"{predicted}_threshold"
            if key not in self.confidence_adjustments:
                self.confidence_adjustments[key] = 0.5
            
            if actual == 'REAL' and predicted == 'FAKE':
                # Model was too aggressive - increase threshold
                self.confidence_adjustments[key] += 0.01
            elif actual == 'FAKE' and predicted == 'REAL':
                # Model was too lenient - decrease threshold
                self.confidence_adjustments[key] -= 0.01
    
    def get_adjusted_confidence(self, raw_confidence: float, predicted_label: str) -> float:
        """Get confidence adjusted based on historical performance"""
        threshold_key = f"{predicted_label}_threshold"
        if threshold_key in self.confidence_adjustments:
            threshold = self.confidence_adjustments[threshold_key]
            # Adjust confidence based on threshold
            if raw_confidence > threshold:
                return min(1.0, raw_confidence * 1.1)  # Boost high confidence
            else:
                return max(0.0, raw_confidence * 0.9)  # Reduce low confidence
        return raw_confidence
    
    def get_word_adjustments(self, text: str) -> Dict[str, float]:
        """Get word-level adjustments for the text"""
        words = text.lower().split()
        adjustments = {}
        
        for word in words:
            clean_word = ''.join(c for c in word if c.isalnum())
            if clean_word in self.word_weights:
                adjustments[clean_word] = self.word_weights[clean_word]
        
        return adjustments
    
    def get_confidence_interval(self, confidence: float) -> Tuple[float, float]:
        """Get confidence interval based on historical performance"""
        # Calculate confidence interval based on recent feedback
        recent_feedback = list(self.feedback_history)[-100:]  # Last 100 feedback items
        
        if not recent_feedback:
            return confidence - 0.1, confidence + 0.1
        
        # Calculate error rate for similar confidence levels
        similar_confidence = [f for f in recent_feedback 
                            if abs(f['predicted_confidence'] - confidence) < 0.1]
        
        if similar_confidence:
            error_rate = sum(1 for f in similar_confidence 
                           if f['predicted_label'] != f['actual_label']) / len(similar_confidence)
            uncertainty = error_rate * 0.2  # Convert error rate to uncertainty
        else:
            uncertainty = 0.1  # Default uncertainty
        
        return max(0.0, confidence - uncertainty), min(1.0, confidence + uncertainty)
    
    def get_model_health_score(self) -> float:
        """Get overall model health score based on recent performance"""
        recent_feedback = list(self.feedback_history)[-50:]  # Last 50 feedback items
        
        if not recent_feedback:
            return 0.5  # Neutral score if no feedback
        
        # Calculate accuracy
        correct = sum(1 for f in recent_feedback 
                     if f['predicted_label'] == f['actual_label'])
        accuracy = correct / len(recent_feedback)
        
        # Calculate confidence calibration
        confidence_scores = [f['predicted_confidence'] for f in recent_feedback]
        avg_confidence = np.mean(confidence_scores)
        
        # Health score combines accuracy and confidence calibration
        health_score = (accuracy + (1 - abs(avg_confidence - accuracy))) / 2
        
        return health_score
    
    def get_recommendations(self) -> List[str]:
        """Get recommendations for improving model performance"""
        recommendations = []
        
        health_score = self.get_model_health_score()
        
        if health_score < 0.6:
            recommendations.append("Model accuracy is low. Consider retraining with more diverse data.")
        
        if health_score < 0.7:
            recommendations.append("Model confidence calibration needs improvement.")
        
        # Check for specific patterns
        recent_feedback = list(self.feedback_history)[-100:]
        if recent_feedback:
            fake_errors = [f for f in recent_feedback 
                          if f['predicted_label'] == 'FAKE' and f['actual_label'] == 'REAL']
            real_errors = [f for f in recent_feedback 
                          if f['predicted_label'] == 'REAL' and f['actual_label'] == 'FAKE']
            
            if len(fake_errors) > len(real_errors) * 2:
                recommendations.append("Model is too aggressive in predicting FAKE. Consider adjusting thresholds.")
            elif len(real_errors) > len(fake_errors) * 2:
                recommendations.append("Model is too lenient in predicting FAKE. Consider lowering thresholds.")
        
        return recommendations
    
    def get_uncertainty_metrics(self) -> Dict[str, float]:
        """Get uncertainty metrics for the model"""
        recent_feedback = list(self.feedback_history)[-100:]
        
        if not recent_feedback:
            return {"uncertainty": 0.5, "reliability": 0.5}
        
        # Calculate various uncertainty metrics
        accuracies = []
        for i in range(0, len(recent_feedback), 10):
            batch = recent_feedback[i:i+10]
            if batch:
                correct = sum(1 for f in batch if f['predicted_label'] == f['actual_label'])
                accuracies.append(correct / len(batch))
        
        if accuracies:
            uncertainty = 1 - np.mean(accuracies)
            reliability = 1 - np.std(accuracies)
        else:
            uncertainty = 0.5
            reliability = 0.5
        
        return {
            "uncertainty": uncertainty,
            "reliability": reliability,
            "health_score": self.get_model_health_score()
        }
