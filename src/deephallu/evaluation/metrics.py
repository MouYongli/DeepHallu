"""
Hallucination evaluation metrics for Vision Language Models
"""

import re
import json
from typing import List, Dict, Set, Tuple, Any
from collections import defaultdict
import numpy as np
from abc import ABC, abstractmethod


class BaseMetric(ABC):
    """Base class for hallucination metrics"""
    
    @abstractmethod
    def compute(self, predictions: List[str], references: List[str], **kwargs) -> Dict[str, float]:
        """Compute the metric score"""
        pass


class CHAIRScore(BaseMetric):
    """
    CHAIR (Caption Hallucination Assessment with Image Relevance) Score
    Measures hallucination in image captioning by identifying objects 
    mentioned in captions that are not present in images
    """
    
    def __init__(self, coco_objects_path: str = None):
        """
        Args:
            coco_objects_path: Path to COCO objects vocabulary file
        """
        self.coco_objects = self._load_coco_objects(coco_objects_path) if coco_objects_path else None
        
    def _load_coco_objects(self, path: str) -> Set[str]:
        """Load COCO object vocabulary"""
        try:
            with open(path, 'r') as f:
                objects = json.load(f)
            return set(objects)
        except:
            # Fallback to common COCO objects
            return {
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
                'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove'
            }
    
    def _extract_objects(self, caption: str) -> Set[str]:
        """Extract object mentions from caption"""
        # Simple noun extraction - in practice, you'd use proper NLP tools
        words = re.findall(r'\b\w+\b', caption.lower())
        if self.coco_objects:
            return set(word for word in words if word in self.coco_objects)
        return set(words)  # Fallback to all words
    
    def compute_chair_s(self, predictions: List[str], ground_truth_objects: List[Set[str]]) -> float:
        """
        Compute CHAIR_S (sentence-level)
        Returns proportion of sentences with at least one hallucinated object
        """
        hallucinated_sentences = 0
        
        for pred, gt_objs in zip(predictions, ground_truth_objects):
            pred_objs = self._extract_objects(pred)
            hallucinated_objs = pred_objs - gt_objs
            
            if len(hallucinated_objs) > 0:
                hallucinated_sentences += 1
                
        return hallucinated_sentences / len(predictions) if predictions else 0.0
    
    def compute_chair_i(self, predictions: List[str], ground_truth_objects: List[Set[str]]) -> float:
        """
        Compute CHAIR_I (instance-level)  
        Returns proportion of mentioned objects that are hallucinated
        """
        total_mentioned = 0
        total_hallucinated = 0
        
        for pred, gt_objs in zip(predictions, ground_truth_objects):
            pred_objs = self._extract_objects(pred)
            hallucinated_objs = pred_objs - gt_objs
            
            total_mentioned += len(pred_objs)
            total_hallucinated += len(hallucinated_objs)
            
        return total_hallucinated / total_mentioned if total_mentioned > 0 else 0.0
    
    def compute(self, predictions: List[str], references: List[str], **kwargs) -> Dict[str, float]:
        """
        Compute CHAIR scores
        
        Args:
            predictions: List of generated captions
            references: List of ground truth object sets or image paths
            **kwargs: Additional arguments including 'ground_truth_objects'
        """
        ground_truth_objects = kwargs.get('ground_truth_objects', [])
        
        if not ground_truth_objects:
            # If no ground truth objects provided, try to extract from references
            ground_truth_objects = [self._extract_objects(ref) for ref in references]
        
        chair_s = self.compute_chair_s(predictions, ground_truth_objects)
        chair_i = self.compute_chair_i(predictions, ground_truth_objects)
        
        return {
            'CHAIR_S': chair_s,
            'CHAIR_I': chair_i,
            'CHAIR_avg': (chair_s + chair_i) / 2
        }


class POPEEvaluator(BaseMetric):
    """
    POPE (Polling-based Object Probing Evaluation) Evaluator
    Evaluates object hallucination using yes/no questions
    """
    
    def __init__(self):
        self.positive_keywords = {'yes', 'true', 'correct', 'right', 'sure', 'definitely'}
        self.negative_keywords = {'no', 'false', 'incorrect', 'wrong', 'not', 'never'}
    
    def _parse_response(self, response: str) -> bool:
        """Parse model response to yes/no"""
        response = response.lower().strip()
        
        # Check for explicit yes/no
        if any(word in response for word in self.positive_keywords):
            return True
        elif any(word in response for word in self.negative_keywords):
            return False
        else:
            # Default to positive if uncertain
            return True
    
    def compute(self, predictions: List[str], references: List[str], **kwargs) -> Dict[str, float]:
        """
        Compute POPE evaluation metrics
        
        Args:
            predictions: List of model responses (yes/no)
            references: List of ground truth labels (yes/no or True/False)
            **kwargs: Additional arguments including 'question_types'
        """
        question_types = kwargs.get('question_types', ['all'] * len(predictions))
        
        # Parse predictions and references
        pred_binary = [self._parse_response(pred) for pred in predictions]
        ref_binary = []
        
        for ref in references:
            if isinstance(ref, bool):
                ref_binary.append(ref)
            elif isinstance(ref, str):
                ref_binary.append(ref.lower() in {'yes', 'true', '1'})
            else:
                ref_binary.append(bool(ref))
        
        # Compute metrics
        tp = sum(1 for p, r in zip(pred_binary, ref_binary) if p and r)
        fp = sum(1 for p, r in zip(pred_binary, ref_binary) if p and not r)
        tn = sum(1 for p, r in zip(pred_binary, ref_binary) if not p and not r)
        fn = sum(1 for p, r in zip(pred_binary, ref_binary) if not p and r)
        
        accuracy = (tp + tn) / len(predictions) if predictions else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Hallucination rate (false positive rate)
        hallucination_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision, 
            'recall': recall,
            'f1_score': f1,
            'hallucination_rate': hallucination_rate,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }


class HallucinationRate(BaseMetric):
    """
    General hallucination rate metric
    Computes the rate of hallucinated content across different categories
    """
    
    def __init__(self, categories: List[str] = None):
        """
        Args:
            categories: List of hallucination categories to track
        """
        self.categories = categories or ['object', 'attribute', 'relation', 'factual']
    
    def compute(self, predictions: List[str], references: List[str], **kwargs) -> Dict[str, float]:
        """
        Compute hallucination rates by category
        
        Args:
            predictions: List of model outputs
            references: List of ground truth
            **kwargs: Additional arguments including 'annotations'
        """
        annotations = kwargs.get('annotations', [])
        
        if not annotations:
            # Simple binary hallucination detection
            hallucinations = 0
            for pred, ref in zip(predictions, references):
                # Simple heuristic: check if prediction contains content not in reference
                if len(pred.split()) > len(ref.split()) * 1.5:  # 50% longer suggests hallucination
                    hallucinations += 1
            
            return {
                'overall_hallucination_rate': hallucinations / len(predictions) if predictions else 0.0
            }
        
        # Category-wise hallucination analysis
        category_counts = defaultdict(int)
        category_totals = defaultdict(int)
        
        for annotation in annotations:
            for category in self.categories:
                if category in annotation:
                    category_totals[category] += 1
                    if annotation[category].get('hallucinated', False):
                        category_counts[category] += 1
        
        results = {}
        for category in self.categories:
            if category_totals[category] > 0:
                results[f'{category}_hallucination_rate'] = category_counts[category] / category_totals[category]
            else:
                results[f'{category}_hallucination_rate'] = 0.0
        
        # Overall rate
        total_hallucinations = sum(category_counts.values())
        total_instances = sum(category_totals.values())
        results['overall_hallucination_rate'] = total_hallucinations / total_instances if total_instances > 0 else 0.0
        
        return results


class FaithfulnessScore(BaseMetric):
    """
    Measures semantic faithfulness between generated text and image content
    """
    
    def __init__(self, similarity_threshold: float = 0.8):
        """
        Args:
            similarity_threshold: Threshold for considering content faithful
        """
        self.similarity_threshold = similarity_threshold
    
    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts
        In practice, this would use sentence embeddings like BERT, SentenceTransformers, etc.
        """
        # Simplified implementation - in practice use proper embeddings
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def compute(self, predictions: List[str], references: List[str], **kwargs) -> Dict[str, float]:
        """
        Compute faithfulness scores
        
        Args:
            predictions: List of generated texts
            references: List of reference texts or image descriptions
            **kwargs: Additional arguments
        """
        similarities = []
        faithful_count = 0
        
        for pred, ref in zip(predictions, references):
            similarity = self._compute_semantic_similarity(pred, ref)
            similarities.append(similarity)
            
            if similarity >= self.similarity_threshold:
                faithful_count += 1
        
        return {
            'avg_faithfulness': np.mean(similarities) if similarities else 0.0,
            'faithfulness_rate': faithful_count / len(predictions) if predictions else 0.0,
            'min_faithfulness': np.min(similarities) if similarities else 0.0,
            'max_faithfulness': np.max(similarities) if similarities else 0.0
        } 