#!/usr/bin/env python3
"""
Attention Modification Experiment for VLM Hallucination Analysis

This script demonstrates:
1. Vision-text embedding fusion strategies
2. Attention mechanism modifications (sparse vision, causal text)
3. Hallucination detection and analysis
4. Comparative evaluation of different attention patterns
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from deephallu.models.attention_hooks import AttentionModifier, EmbeddingFuser, visualize_attention_patterns
from deephallu.evaluation.metrics import CHAIRScore, POPEEvaluator, HallucinationRate

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('attention_experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VLMAttentionExperiment:
    """Main experiment class for VLM attention modification"""
    
    def __init__(self, 
                 model_name: str = "deepseek-vl2",
                 device: str = "cuda",
                 output_dir: str = "./outputs"):
        """
        Initialize experiment
        
        Args:
            model_name: Name of the VLM to experiment with
            device: Device to run experiments on
            output_dir: Directory to save results
        """
        self.model_name = model_name
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        self.model = None
        self.attention_modifier = None
        self.embedding_fuser = None
        self.evaluators = {}
        
        # Experiment results
        self.results = {}
        self.attention_analyses = {}
        
    def setup_model(self) -> None:
        """Setup the VLM model"""
        logger.info(f"Setting up model: {self.model_name}")
        
        try:
            if self.model_name == "deepseek-vl2":
                from transformers import AutoModel, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-vl2-small")
                self.model = AutoModel.from_pretrained("deepseek-ai/deepseek-vl2-small")
            elif self.model_name == "qwen2.5-vl":
                from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
                self.model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
            else:
                # Fallback: create a mock model for demonstration
                self.model = self._create_mock_vlm()
                self.tokenizer = None
                logger.warning("Using mock model for demonstration")
                
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.warning(f"Failed to load {self.model_name}: {e}")
            logger.info("Creating mock model for demonstration")
            self.model = self._create_mock_vlm()
            self.tokenizer = None
            
    def _create_mock_vlm(self) -> nn.Module:
        """Create a mock VLM for demonstration purposes"""
        
        class MockVLM(nn.Module):
            def __init__(self, hidden_dim=768, num_layers=12, num_heads=12):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.num_layers = num_layers
                self.num_heads = num_heads
                
                # Vision encoder
                self.vision_encoder = nn.Sequential(
                    nn.Linear(2048, hidden_dim),  # Assume ResNet features
                    nn.LayerNorm(hidden_dim)
                )
                
                # Text encoder
                self.text_embeddings = nn.Embedding(50000, hidden_dim)
                
                # Transformer layers with attention
                self.layers = nn.ModuleList([
                    MockTransformerLayer(hidden_dim, num_heads) 
                    for _ in range(num_layers)
                ])
                
                # Output head
                self.output_projection = nn.Linear(hidden_dim, 50000)
                
            def forward(self, vision_features, input_ids, vision_seq_len=49, text_seq_len=128):
                batch_size = vision_features.size(0)
                
                # Process vision features
                vision_embeddings = self.vision_encoder(vision_features)  # [B, V, H]
                
                # Process text
                text_embeddings = self.text_embeddings(input_ids)  # [B, T, H]
                
                # Concatenate
                combined_embeddings = torch.cat([vision_embeddings, text_embeddings], dim=1)
                
                # Pass through transformer layers
                hidden_states = combined_embeddings
                attention_weights_list = []
                
                for layer in self.layers:
                    hidden_states, attention_weights = layer(hidden_states)
                    attention_weights_list.append(attention_weights)
                
                # Output projection
                logits = self.output_projection(hidden_states)
                
                return {
                    'logits': logits,
                    'hidden_states': hidden_states,
                    'attention_weights': attention_weights_list[-1],  # Last layer attention
                    'all_attention_weights': attention_weights_list
                }
        
        class MockTransformerLayer(nn.Module):
            def __init__(self, hidden_dim, num_heads):
                super().__init__()
                self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
                self.layer_norm1 = nn.LayerNorm(hidden_dim)
                self.feed_forward = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim)
                )
                self.layer_norm2 = nn.LayerNorm(hidden_dim)
                
            def forward(self, hidden_states):
                # Self-attention
                attn_output, attention_weights = self.attention(
                    hidden_states, hidden_states, hidden_states
                )
                hidden_states = self.layer_norm1(hidden_states + attn_output)
                
                # Feed forward
                ff_output = self.feed_forward(hidden_states)
                hidden_states = self.layer_norm2(hidden_states + ff_output)
                
                return hidden_states, attention_weights
        
        return MockVLM().to(self.device)
    
    def setup_attention_modifier(self, 
                               vision_sparsity_ratio: float = 0.1,
                               vision_attention_type: str = "symmetric",
                               text_attention_type: str = "causal",
                               cross_attention_type: str = "bidirectional") -> None:
        """Setup attention modifier with specified parameters"""
        logger.info("Setting up attention modifier")
        
        self.attention_modifier = AttentionModifier(
            vision_sparsity_ratio=vision_sparsity_ratio,
            vision_attention_type=vision_attention_type,
            text_attention_type=text_attention_type,
            cross_attention_type=cross_attention_type
        )
        
    def setup_embedding_fuser(self, 
                            fusion_method: str = "attention",
                            hidden_dim: int = 768) -> None:
        """Setup embedding fusion strategy"""
        logger.info(f"Setting up embedding fuser with method: {fusion_method}")
        
        self.embedding_fuser = EmbeddingFuser(
            fusion_method=fusion_method,
            hidden_dim=hidden_dim
        ).to(self.device)
        
    def setup_evaluators(self) -> None:
        """Setup hallucination evaluators"""
        logger.info("Setting up evaluators")
        
        self.evaluators = {
            'chair': CHAIRScore(),
            'pope': POPEEvaluator(),
            'hallucination_rate': HallucinationRate()
        }
        
    def generate_sample_data(self, 
                           batch_size: int = 4,
                           vision_seq_len: int = 49,
                           text_seq_len: int = 128) -> Dict[str, torch.Tensor]:
        """Generate sample data for testing"""
        logger.info(f"Generating sample data: batch_size={batch_size}")
        
        # Generate mock vision features (ResNet-like)
        vision_features = torch.randn(batch_size, vision_seq_len, 2048).to(self.device)
        
        # Generate mock text input IDs
        input_ids = torch.randint(0, 50000, (batch_size, text_seq_len)).to(self.device)
        
        # Generate mock labels for evaluation
        labels = torch.randint(0, 2, (batch_size,)).to(self.device)  # Binary labels for POPE
        
        return {
            'vision_features': vision_features,
            'input_ids': input_ids,
            'labels': labels,
            'vision_seq_len': vision_seq_len,
            'text_seq_len': text_seq_len
        }
        
    def run_baseline_experiment(self, data: Dict[str, torch.Tensor]) -> Dict[str, any]:
        """Run baseline experiment without attention modifications"""
        logger.info("Running baseline experiment")
        
        with torch.no_grad():
            outputs = self.model(
                data['vision_features'], 
                data['input_ids'],
                data['vision_seq_len'],
                data['text_seq_len']
            )
        
        # Extract attention weights
        attention_weights = outputs['attention_weights']
        
        # Generate text responses (mock)
        generated_texts = [f"Generated text {i} without modification" for i in range(len(data['input_ids']))]
        reference_texts = [f"Reference text {i}" for i in range(len(data['input_ids']))]
        
        # Evaluate hallucinations
        results = {}
        for name, evaluator in self.evaluators.items():
            if name == 'pope':
                # Convert to binary predictions for POPE
                binary_predictions = ['yes' if i % 2 == 0 else 'no' for i in range(len(generated_texts))]
                binary_references = ['yes' if label.item() == 1 else 'no' for label in data['labels']]
                results[name] = evaluator.compute(binary_predictions, binary_references)
            else:
                results[name] = evaluator.compute(generated_texts, reference_texts)
        
        return {
            'outputs': outputs,
            'attention_weights': attention_weights,
            'generated_texts': generated_texts,
            'evaluation_results': results
        }
        
    def run_modified_experiment(self, data: Dict[str, torch.Tensor]) -> Dict[str, any]:
        """Run experiment with attention modifications"""
        logger.info("Running modified attention experiment")
        
        # Register attention hooks
        self.attention_modifier.register_hooks(
            self.model,
            data['vision_seq_len'],
            data['text_seq_len'],
            target_layers=['layers.11']  # Hook last layer
        )
        
        try:
            with torch.no_grad():
                outputs = self.model(
                    data['vision_features'], 
                    data['input_ids'],
                    data['vision_seq_len'],
                    data['text_seq_len']
                )
            
            # Extract modified attention weights
            attention_weights = outputs['attention_weights']
            
            # Generate text responses (mock with modification)
            generated_texts = [f"Generated text {i} with modified attention" for i in range(len(data['input_ids']))]
            reference_texts = [f"Reference text {i}" for i in range(len(data['input_ids']))]
            
            # Evaluate hallucinations
            results = {}
            for name, evaluator in self.evaluators.items():
                if name == 'pope':
                    # Convert to binary predictions for POPE
                    binary_predictions = ['no' if i % 2 == 0 else 'yes' for i in range(len(generated_texts))]  # Different pattern
                    binary_references = ['yes' if label.item() == 1 else 'no' for label in data['labels']]
                    results[name] = evaluator.compute(binary_predictions, binary_references)
                else:
                    results[name] = evaluator.compute(generated_texts, reference_texts)
            
            # Get attention analysis
            attention_analysis = self.attention_modifier.get_attention_analysis()
            
        finally:
            # Remove hooks
            self.attention_modifier.remove_hooks()
        
        return {
            'outputs': outputs,
            'attention_weights': attention_weights,
            'generated_texts': generated_texts,
            'evaluation_results': results,
            'attention_analysis': attention_analysis
        }
        
    def compare_attention_patterns(self, 
                                 baseline_attention: torch.Tensor,
                                 modified_attention: torch.Tensor,
                                 vision_seq_len: int,
                                 text_seq_len: int) -> None:
        """Compare attention patterns before and after modification"""
        logger.info("Comparing attention patterns")
        
        # Create comparison visualizations
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # Average attention across batch and heads
        baseline_avg = baseline_attention.mean(dim=(0, 1)).cpu().numpy()
        modified_avg = modified_attention.mean(dim=(0, 1)).cpu().numpy()
        
        # Vision-to-vision attention
        baseline_v2v = baseline_avg[:vision_seq_len, :vision_seq_len]
        modified_v2v = modified_avg[:vision_seq_len, :vision_seq_len]
        
        sns.heatmap(baseline_v2v, ax=axes[0, 0], cmap='Blues', cbar=False)
        axes[0, 0].set_title('Baseline Vision-to-Vision')
        
        sns.heatmap(modified_v2v, ax=axes[1, 0], cmap='Blues', cbar=False)
        axes[1, 0].set_title('Modified Vision-to-Vision')
        
        # Text-to-text attention
        baseline_t2t = baseline_avg[vision_seq_len:, vision_seq_len:]
        modified_t2t = modified_avg[vision_seq_len:, vision_seq_len:]
        
        sns.heatmap(baseline_t2t, ax=axes[0, 1], cmap='Reds', cbar=False)
        axes[0, 1].set_title('Baseline Text-to-Text')
        
        sns.heatmap(modified_t2t, ax=axes[1, 1], cmap='Reds', cbar=False)
        axes[1, 1].set_title('Modified Text-to-Text')
        
        # Vision-to-text attention
        baseline_v2t = baseline_avg[:vision_seq_len, vision_seq_len:]
        modified_v2t = modified_avg[:vision_seq_len, vision_seq_len:]
        
        sns.heatmap(baseline_v2t, ax=axes[0, 2], cmap='Greens', cbar=False)
        axes[0, 2].set_title('Baseline Vision-to-Text')
        
        sns.heatmap(modified_v2t, ax=axes[1, 2], cmap='Greens', cbar=False)
        axes[1, 2].set_title('Modified Vision-to-Text')
        
        # Text-to-vision attention
        baseline_t2v = baseline_avg[vision_seq_len:, :vision_seq_len]
        modified_t2v = modified_avg[vision_seq_len:, :vision_seq_len]
        
        sns.heatmap(baseline_t2v, ax=axes[0, 3], cmap='Purples', cbar=False)
        axes[0, 3].set_title('Baseline Text-to-Vision')
        
        sns.heatmap(modified_t2v, ax=axes[1, 3], cmap='Purples', cbar=False)
        axes[1, 3].set_title('Modified Text-to-Vision')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'attention_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Attention comparison saved to {self.output_dir / 'attention_comparison.png'}")
        
    def analyze_hallucination_reduction(self, 
                                      baseline_results: Dict[str, Dict],
                                      modified_results: Dict[str, Dict]) -> Dict[str, float]:
        """Analyze hallucination reduction between baseline and modified models"""
        logger.info("Analyzing hallucination reduction")
        
        analysis = {}
        
        for evaluator_name in baseline_results:
            baseline_metrics = baseline_results[evaluator_name]
            modified_metrics = modified_results[evaluator_name]
            
            if evaluator_name == 'chair':
                # Lower CHAIR scores are better
                chair_reduction = baseline_metrics.get('CHAIR_avg', 0) - modified_metrics.get('CHAIR_avg', 0)
                analysis[f'{evaluator_name}_reduction'] = chair_reduction
                
            elif evaluator_name == 'pope':
                # Higher accuracy is better, lower hallucination rate is better
                accuracy_improvement = modified_metrics.get('accuracy', 0) - baseline_metrics.get('accuracy', 0)
                hallucination_reduction = baseline_metrics.get('hallucination_rate', 0) - modified_metrics.get('hallucination_rate', 0)
                analysis[f'{evaluator_name}_accuracy_improvement'] = accuracy_improvement
                analysis[f'{evaluator_name}_hallucination_reduction'] = hallucination_reduction
                
            elif evaluator_name == 'hallucination_rate':
                # Lower overall hallucination rate is better
                overall_reduction = baseline_metrics.get('overall_hallucination_rate', 0) - modified_metrics.get('overall_hallucination_rate', 0)
                analysis[f'{evaluator_name}_overall_reduction'] = overall_reduction
        
        return analysis
        
    def save_results(self, results: Dict) -> None:
        """Save experiment results"""
        logger.info("Saving experiment results")
        
        # Save to JSON
        results_path = self.output_dir / 'experiment_results.json'
        
        # Convert tensors to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                serializable_results[key] = value.cpu().tolist()
            elif isinstance(value, dict):
                serializable_results[key] = {k: v.cpu().tolist() if isinstance(v, torch.Tensor) else v for k, v in value.items()}
            else:
                serializable_results[key] = value
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        logger.info(f"Results saved to {results_path}")
        
    def run_full_experiment(self, 
                          batch_size: int = 4,
                          vision_seq_len: int = 49,
                          text_seq_len: int = 128) -> Dict[str, any]:
        """Run the complete attention modification experiment"""
        logger.info("Starting full attention modification experiment")
        
        # Setup all components
        self.setup_model()
        self.setup_attention_modifier(
            vision_sparsity_ratio=0.1,
            vision_attention_type="symmetric",
            text_attention_type="causal"
        )
        self.setup_embedding_fuser(fusion_method="attention")
        self.setup_evaluators()
        
        # Generate sample data
        data = self.generate_sample_data(batch_size, vision_seq_len, text_seq_len)
        
        # Run baseline experiment
        logger.info("Running baseline experiment...")
        baseline_results = self.run_baseline_experiment(data)
        
        # Run modified experiment
        logger.info("Running modified experiment...")
        modified_results = self.run_modified_experiment(data)
        
        # Compare attention patterns
        self.compare_attention_patterns(
            baseline_results['attention_weights'],
            modified_results['attention_weights'],
            vision_seq_len,
            text_seq_len
        )
        
        # Analyze hallucination reduction
        hallucination_analysis = self.analyze_hallucination_reduction(
            baseline_results['evaluation_results'],
            modified_results['evaluation_results']
        )
        
        # Compile final results
        final_results = {
            'baseline_evaluation': baseline_results['evaluation_results'],
            'modified_evaluation': modified_results['evaluation_results'],
            'hallucination_analysis': hallucination_analysis,
            'attention_analysis': modified_results.get('attention_analysis', {}),
            'experiment_config': {
                'model_name': self.model_name,
                'batch_size': batch_size,
                'vision_seq_len': vision_seq_len,
                'text_seq_len': text_seq_len,
                'vision_sparsity_ratio': self.attention_modifier.vision_sparsity_ratio,
                'vision_attention_type': self.attention_modifier.vision_attention_type,
                'text_attention_type': self.attention_modifier.text_attention_type,
                'cross_attention_type': self.attention_modifier.cross_attention_type
            }
        }
        
        # Save results
        self.save_results(final_results)
        
        # Print summary
        self.print_experiment_summary(final_results)
        
        return final_results
        
    def print_experiment_summary(self, results: Dict) -> None:
        """Print a summary of experiment results"""
        logger.info("=== EXPERIMENT SUMMARY ===")
        
        print(f"\nModel: {self.model_name}")
        print(f"Configuration: {results['experiment_config']}")
        
        print("\n--- Baseline Results ---")
        for evaluator, metrics in results['baseline_evaluation'].items():
            print(f"{evaluator.upper()}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        print("\n--- Modified Results ---")
        for evaluator, metrics in results['modified_evaluation'].items():
            print(f"{evaluator.upper()}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        print("\n--- Hallucination Analysis ---")
        for analysis, value in results['hallucination_analysis'].items():
            direction = "↓" if "reduction" in analysis else "↑"
            print(f"  {analysis}: {value:.4f} {direction}")
        
        print("\n--- Attention Analysis ---")
        for layer, metrics in results['attention_analysis'].items():
            print(f"{layer}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")


def main():
    """Main function to run the experiment"""
    parser = argparse.ArgumentParser(description="VLM Attention Modification Experiment")
    parser.add_argument("--model", type=str, default="mock", 
                       choices=["deepseek-vl2", "qwen2.5-vl", "mock"],
                       help="Model to experiment with")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size for experiments")
    parser.add_argument("--vision-seq-len", type=int, default=49,
                       help="Vision sequence length")
    parser.add_argument("--text-seq-len", type=int, default=128,
                       help="Text sequence length")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run on")
    parser.add_argument("--vision-sparsity", type=float, default=0.1,
                       help="Vision attention sparsity ratio")
    
    args = parser.parse_args()
    
    # Create experiment instance
    experiment = VLMAttentionExperiment(
        model_name=args.model,
        device=args.device,
        output_dir=args.output_dir
    )
    
    # Run experiment
    results = experiment.run_full_experiment(
        batch_size=args.batch_size,
        vision_seq_len=args.vision_seq_len,
        text_seq_len=args.text_seq_len
    )
    
    logger.info("Experiment completed successfully!")
    return results


if __name__ == "__main__":
    main() 