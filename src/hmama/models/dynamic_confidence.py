"""
Dynamic confidence adjustment system that adapts to content patterns
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
from collections import defaultdict
import logging

class DynamicConfidenceAdjuster:
    """
    Dynamically adjusts model confidence based on content patterns and context
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Pattern-based confidence adjustments
        self.pattern_weights = {
            # News source indicators (higher confidence for known sources)
            'reliable_sources': {
                'patterns': [r'\b(Reuters|AP|BBC|CNN|NPR|PBS|Wall Street Journal|New York Times)\b'],
                'weight': 1.2
            },
            # Sensational language (lower confidence)
            'sensational': {
                'patterns': [r'\b(SHOCKING|BREAKING|URGENT|EXCLUSIVE|REVEALED|EXPOSED)\b'],
                'weight': 0.8
            },
            # Factual language (higher confidence)
            'factual': {
                'patterns': [r'\b(according to|reported by|confirmed|verified|official)\b'],
                'weight': 1.1
            },
            # Emotional language (lower confidence)
            'emotional': {
                'patterns': [r'\b(outrageous|disgusting|terrifying|amazing|incredible)\b'],
                'weight': 0.9
            },
            # Specific claims (lower confidence without evidence)
            'specific_claims': {
                'patterns': [r'\b(exactly|precisely|definitely|certainly|without doubt)\b'],
                'weight': 0.85
            },
            # Uncertainty indicators (higher confidence for honest reporting)
            'uncertainty': {
                'patterns': [r'\b(allegedly|reportedly|sources say|unconfirmed|may have)\b'],
                'weight': 1.05
            }
        }
        
        # Context-based adjustments
        self.context_adjustments = {
            'length_based': {
                'short_text': {'threshold': 50, 'weight': 0.9},  # Short texts are harder to classify
                'medium_text': {'threshold': 200, 'weight': 1.0},
                'long_text': {'threshold': 500, 'weight': 1.1}
            },
            'punctuation_based': {
                'excessive_exclamation': {'pattern': r'!{2,}', 'weight': 0.8},
                'excessive_caps': {'pattern': r'[A-Z]{5,}', 'weight': 0.85},
                'balanced_punctuation': {'pattern': r'[.!?]', 'weight': 1.05}
            }
        }
    
    def adjust_confidence(self, text: str, raw_confidence: float, 
                         predicted_label: str, image_available: bool = False) -> Tuple[float, Dict[str, float]]:
        """
        Adjust confidence based on content patterns and context
        
        Returns:
            adjusted_confidence: The adjusted confidence score
            adjustment_factors: Dictionary explaining the adjustments made
        """
        adjustment_factors = {}
        adjusted_confidence = raw_confidence
        
        # Apply pattern-based adjustments
        pattern_adjustment = self._apply_pattern_adjustments(text)
        adjustment_factors['pattern_adjustment'] = pattern_adjustment
        adjusted_confidence *= pattern_adjustment
        
        # Apply context-based adjustments
        context_adjustment = self._apply_context_adjustments(text)
        adjustment_factors['context_adjustment'] = context_adjustment
        adjusted_confidence *= context_adjustment
        
        # Apply image availability adjustment
        image_adjustment = 1.1 if image_available else 0.95
        adjustment_factors['image_adjustment'] = image_adjustment
        adjusted_confidence *= image_adjustment
        
        # Apply label-specific adjustments
        label_adjustment = self._apply_label_adjustments(text, predicted_label)
        adjustment_factors['label_adjustment'] = label_adjustment
        adjusted_confidence *= label_adjustment
        
        # Apply temporal context (if available)
        temporal_adjustment = self._apply_temporal_adjustments(text)
        adjustment_factors['temporal_adjustment'] = temporal_adjustment
        adjusted_confidence *= temporal_adjustment
        
        # Ensure confidence stays within bounds
        adjusted_confidence = max(0.1, min(0.99, adjusted_confidence))
        
        # Calculate overall adjustment factor
        adjustment_factors['overall_adjustment'] = adjusted_confidence / raw_confidence if raw_confidence > 0 else 1.0
        
        return adjusted_confidence, adjustment_factors
    
    def _apply_pattern_adjustments(self, text: str) -> float:
        """Apply pattern-based confidence adjustments"""
        total_adjustment = 1.0
        
        for category, config in self.pattern_weights.items():
            for pattern in config['patterns']:
                if re.search(pattern, text, re.IGNORECASE):
                    total_adjustment *= config['weight']
                    self.logger.debug(f"Applied {category} adjustment: {config['weight']}")
        
        return total_adjustment
    
    def _apply_context_adjustments(self, text: str) -> float:
        """Apply context-based confidence adjustments"""
        total_adjustment = 1.0
        
        # Length-based adjustment
        text_length = len(text)
        if text_length < self.context_adjustments['length_based']['short_text']['threshold']:
            total_adjustment *= self.context_adjustments['length_based']['short_text']['weight']
        elif text_length > self.context_adjustments['length_based']['long_text']['threshold']:
            total_adjustment *= self.context_adjustments['length_based']['long_text']['weight']
        else:
            total_adjustment *= self.context_adjustments['length_based']['medium_text']['weight']
        
        # Punctuation-based adjustment
        if re.search(self.context_adjustments['punctuation_based']['excessive_exclamation']['pattern'], text):
            total_adjustment *= self.context_adjustments['punctuation_based']['excessive_exclamation']['weight']
        elif re.search(self.context_adjustments['punctuation_based']['excessive_caps']['pattern'], text):
            total_adjustment *= self.context_adjustments['punctuation_based']['excessive_caps']['weight']
        elif re.search(self.context_adjustments['punctuation_based']['balanced_punctuation']['pattern'], text):
            total_adjustment *= self.context_adjustments['punctuation_based']['balanced_punctuation']['weight']
        
        return total_adjustment
    
    def _apply_label_adjustments(self, text: str, predicted_label: str) -> float:
        """Apply label-specific confidence adjustments"""
        if predicted_label == 'FAKE':
            # For FAKE predictions, look for indicators that might suggest it's actually real
            real_indicators = [
                r'\b(official|confirmed|verified|reported by|according to)\b',
                r'\b(Reuters|AP|BBC|CNN|NPR)\b',
                r'\b(government|authorities|police|hospital|court)\b'
            ]
            
            real_indicator_count = sum(1 for pattern in real_indicators 
                                     if re.search(pattern, text, re.IGNORECASE))
            
            if real_indicator_count > 0:
                return 0.9  # Reduce confidence for FAKE prediction
            else:
                return 1.1  # Increase confidence for FAKE prediction
                
        else:  # REAL prediction
            # For REAL predictions, look for indicators that might suggest it's actually fake
            fake_indicators = [
                r'\b(SHOCKING|BREAKING|URGENT|EXCLUSIVE|REVEALED)\b',
                r'\b(you won\'t believe|this will shock you|doctors hate this)\b',
                r'\b(click here|read more|share this)\b'
            ]
            
            fake_indicator_count = sum(1 for pattern in fake_indicators 
                                     if re.search(pattern, text, re.IGNORECASE))
            
            if fake_indicator_count > 0:
                return 0.9  # Reduce confidence for REAL prediction
            else:
                return 1.05  # Slightly increase confidence for REAL prediction
    
    def _apply_temporal_adjustments(self, text: str) -> float:
        """Apply temporal context adjustments"""
        # Look for time indicators
        time_patterns = [
            r'\b(today|yesterday|this morning|this afternoon|tonight)\b',
            r'\b(just now|recently|earlier today)\b',
            r'\b(breaking|developing|live)\b'
        ]
        
        time_indicator_count = sum(1 for pattern in time_patterns 
                                 if re.search(pattern, text, re.IGNORECASE))
        
        if time_indicator_count > 0:
            return 1.05  # Slightly increase confidence for recent news
        else:
            return 1.0  # No temporal adjustment
    
    def get_confidence_explanation(self, text: str, raw_confidence: float, 
                                 adjusted_confidence: float, adjustment_factors: Dict[str, float]) -> str:
        """Generate human-readable explanation of confidence adjustments"""
        explanations = []
        
        if abs(adjusted_confidence - raw_confidence) > 0.05:
            explanations.append(f"Confidence adjusted from {raw_confidence:.1%} to {adjusted_confidence:.1%}")
        
        # Explain specific adjustments
        if adjustment_factors.get('pattern_adjustment', 1.0) != 1.0:
            explanations.append("Pattern-based adjustment applied based on content style")
        
        if adjustment_factors.get('context_adjustment', 1.0) != 1.0:
            explanations.append("Context-based adjustment applied based on text characteristics")
        
        if adjustment_factors.get('image_adjustment', 1.0) != 1.0:
            if adjustment_factors['image_adjustment'] > 1.0:
                explanations.append("Confidence increased due to image analysis")
            else:
                explanations.append("Confidence slightly reduced due to text-only analysis")
        
        if adjustment_factors.get('label_adjustment', 1.0) != 1.0:
            explanations.append("Label-specific adjustment applied based on content indicators")
        
        return "; ".join(explanations) if explanations else "No significant adjustments applied"
    
    def get_uncertainty_indicators(self, text: str) -> List[str]:
        """Get indicators of uncertainty in the text"""
        indicators = []
        
        # Check for uncertainty patterns
        uncertainty_patterns = [
            (r'\b(allegedly|reportedly|sources say|unconfirmed)\b', 'Uncertainty indicators present'),
            (r'\b(may have|could be|might be|possibly)\b', 'Hedging language detected'),
            (r'\b(according to|reported by|sources claim)\b', 'Attribution present'),
            (r'\b(breaking|developing|live updates)\b', 'Time-sensitive content')
        ]
        
        for pattern, description in uncertainty_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                indicators.append(description)
        
        return indicators
