#!/usr/bin/env python3
"""
Simple demonstration of attention modification hooks for VLMs

This script shows how to:
1. Apply attention hooks to modify vision and text attention patterns
2. Analyze the changes in attention mechanisms  
3. Evaluate the impact on hallucination rates
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from deephallu.models.attention_hooks import AttentionModifier, visualize_attention_patterns
from deephallu.evaluation.metrics import CHAIRScore, POPEEvaluator


def create_simple_vlm():
    """Create a simple VLM for demonstration"""
    
    class SimpleVLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.MultiheadAttention(
                embed_dim=768, 
                num_heads=12, 
                batch_first=True
            )
            self.layer_norm = nn.LayerNorm(768)
            
        def forward(self, embeddings):
            # Simple self-attention
            attn_output, attention_weights = self.attention(
                embeddings, embeddings, embeddings
            )
            output = self.layer_norm(embeddings + attn_output)
            return output, attention_weights
    
    return SimpleVLM()


def demo_attention_modification():
    """Demonstrate attention modification process"""
    print("🔧 Attention Modification Demo")
    print("=" * 50)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_simple_vlm().to(device)
    
    # Create sample data: [batch_size, seq_len, hidden_dim]
    # First 49 tokens are vision, next 128 are text
    batch_size = 2
    vision_seq_len = 49
    text_seq_len = 128
    total_seq_len = vision_seq_len + text_seq_len
    hidden_dim = 768
    
    sample_embeddings = torch.randn(batch_size, total_seq_len, hidden_dim).to(device)
    
    print(f"📊 Input shape: {sample_embeddings.shape}")
    print(f"   Vision tokens: {vision_seq_len}")
    print(f"   Text tokens: {text_seq_len}")
    
    # 1. Baseline forward pass
    print("\n🔍 Running baseline model...")
    model.eval()
    with torch.no_grad():
        baseline_output, baseline_attention = model(sample_embeddings)
    
    print(f"   Baseline attention shape: {baseline_attention.shape}")
    
    # 2. Setup attention modifier
    print("\n⚙️ Setting up attention modifier...")
    attention_modifier = AttentionModifier(
        vision_sparsity_ratio=0.3,  # Keep only 30% of vision attention weights
        vision_attention_type="symmetric",  # Bidirectional for vision
        text_attention_type="causal",  # Causal for text
        cross_attention_type="bidirectional"  # Allow cross-modal attention
    )
    
    # 3. Register hooks
    print("🪝 Registering attention hooks...")
    attention_modifier.register_hooks(
        model=model,
        vision_seq_len=vision_seq_len,
        text_seq_len=text_seq_len
    )
    
    # 4. Modified forward pass
    print("\n🔄 Running modified model...")
    with torch.no_grad():
        modified_output, modified_attention = model(sample_embeddings)
    
    # 5. Analyze changes
    print("\n📈 Analyzing attention changes...")
    analysis = attention_modifier.get_attention_analysis()
    
    for layer_name, metrics in analysis.items():
        print(f"\n   Layer: {layer_name}")
        for metric_name, value in metrics.items():
            print(f"     {metric_name}: {value:.4f}")
    
    # 6. Compare attention patterns
    print("\n🎨 Generating attention visualizations...")
    
    # Calculate attention differences
    attention_diff = torch.abs(baseline_attention - modified_attention).mean()
    sparsity_baseline = (baseline_attention < 0.01).float().mean()
    sparsity_modified = (modified_attention < 0.01).float().mean()
    
    print(f"   Attention difference (L1): {attention_diff:.4f}")
    print(f"   Baseline sparsity: {sparsity_baseline:.4f}")
    print(f"   Modified sparsity: {sparsity_modified:.4f}")
    
    # 7. Simulate hallucination evaluation
    print("\n🧠 Simulating hallucination evaluation...")
    
    # Mock generated texts for evaluation
    baseline_texts = [
        "A red car is driving on the road with a blue elephant in the sky",
        "The person is holding a green phone while sitting on a purple cloud"
    ]
    
    modified_texts = [
        "A red car is driving on the road",
        "The person is holding a phone while sitting in a chair"
    ]
    
    reference_texts = [
        "A car is driving on the road",
        "A person is holding a phone while sitting"
    ]
    
    # Evaluate with CHAIR score
    chair_evaluator = CHAIRScore()
    
    baseline_chair = chair_evaluator.compute(baseline_texts, reference_texts)
    modified_chair = chair_evaluator.compute(modified_texts, reference_texts)
    
    print(f"   Baseline CHAIR Score: {baseline_chair}")
    print(f"   Modified CHAIR Score: {modified_chair}")
    
    # Calculate improvement
    chair_improvement = baseline_chair['CHAIR_avg'] - modified_chair['CHAIR_avg']
    print(f"   CHAIR Improvement: {chair_improvement:.4f}")
    
    # 8. Cleanup
    attention_modifier.remove_hooks()
    
    print("\n✅ Demo completed successfully!")
    
    return {
        'baseline_attention': baseline_attention,
        'modified_attention': modified_attention,
        'attention_analysis': analysis,
        'chair_improvement': chair_improvement
    }


def demo_embedding_fusion():
    """Demonstrate different embedding fusion strategies"""
    print("\n🔀 Embedding Fusion Demo") 
    print("=" * 50)
    
    from deephallu.models.attention_hooks import EmbeddingFuser
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Sample embeddings
    batch_size = 2
    vision_seq_len = 49
    text_seq_len = 128
    hidden_dim = 768
    
    vision_embeddings = torch.randn(batch_size, vision_seq_len, hidden_dim).to(device)
    text_embeddings = torch.randn(batch_size, text_seq_len, hidden_dim).to(device)
    
    print(f"📊 Vision embeddings: {vision_embeddings.shape}")
    print(f"📊 Text embeddings: {text_embeddings.shape}")
    
    # Test different fusion methods
    fusion_methods = ["concat", "gated", "attention"]
    
    for method in fusion_methods:
        print(f"\n🔧 Testing {method} fusion...")
        
        fuser = EmbeddingFuser(
            fusion_method=method,
            hidden_dim=hidden_dim
        ).to(device)
        
        fused = fuser.fuse_embeddings(vision_embeddings, text_embeddings)
        print(f"   Fused shape: {fused.shape}")
        print(f"   Fusion ratio: {fused.shape[1] / (vision_seq_len + text_seq_len):.2f}")


def demo_sparse_attention():
    """Demonstrate sparse attention effects"""
    print("\n🕳️ Sparse Attention Demo")
    print("=" * 50)
    
    # Create sample attention weights
    batch_size, num_heads, seq_len = 1, 12, 100
    attention_weights = torch.rand(batch_size, num_heads, seq_len, seq_len)
    attention_weights = torch.softmax(attention_weights, dim=-1)
    
    modifier = AttentionModifier()
    
    # Test different sparsity ratios
    sparsity_ratios = [0.1, 0.3, 0.5, 0.7]
    
    print("Sparsity Ratio | Original Entropy | Sparse Entropy | Sparsity")
    print("-" * 60)
    
    for ratio in sparsity_ratios:
        # Create sparse mask
        sparse_mask = modifier.create_sparse_attention_mask(
            attention_weights, ratio, method="topk"
        )
        
        # Apply mask
        sparse_attention = attention_weights * sparse_mask
        sparse_attention = torch.softmax(sparse_attention, dim=-1)
        
        # Calculate metrics
        original_entropy = modifier._compute_attention_entropy(attention_weights)
        sparse_entropy = modifier._compute_attention_entropy(sparse_attention)
        sparsity = modifier._compute_attention_sparsity(sparse_attention)
        
        print(f"{ratio:>12.1f} | {original_entropy:>15.4f} | {sparse_entropy:>13.4f} | {sparsity:>8.4f}")


if __name__ == "__main__":
    print("🚀 DeepHallu Attention Hooks Demonstration")
    print("=" * 60)
    
    # Run all demos
    try:
        results = demo_attention_modification()
        demo_embedding_fusion()
        demo_sparse_attention()
        
        print("\n🎉 All demonstrations completed successfully!")
        print(f"📝 Check the attention analysis results above for insights.")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc() 