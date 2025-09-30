"""
LlavaNext Image Processor Analysis
从原始图像 (1100, 808) 到 pixel_values [1, 5, 3, 336, 336] 的全过程分析

这个文件详细展示了LlavaNextImageProcessor如何处理图像的完整流程
"""

import os
os.environ["HF_HOME"] = "/DATA2/HuggingFace"

import numpy as np
import torch
from PIL import Image
from transformers import LlavaNextProcessor
import matplotlib.pyplot as plt
from typing import List, Tuple


class ImageProcessorAnalysis:
    """分析LlavaNext图像处理器的完整流程"""
    
    def __init__(self):
        self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        self.image_processor = self.processor.image_processor
        
    def analyze_full_pipeline(self, image_path_or_pil: str | Image.Image):
        """完整分析从原始图像到最终tensor的处理流程"""
        
        # Step 1: 加载原始图像
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil)
        else:
            image = image_path_or_pil
            
        print("=" * 80)
        print("LlavaNext Image Processing Pipeline Analysis")
        print("=" * 80)
        
        # Step 2: 原始图像信息
        print(f"\n📸 Step 1: Original Image Info")
        print(f"   Size: {image.size} (width x height)")
        print(f"   Mode: {image.mode}")
        print(f"   Format: {getattr(image, 'format', 'N/A')}")
        
        # Step 3: 分析image processor配置
        self._analyze_processor_config()
        
        # Step 4: 最佳分辨率选择
        best_resolution = self._analyze_resolution_selection(image.size)
        
        # Step 5: 多尺度处理
        processed_results = self._analyze_multiscale_processing(image)
        
        # Step 6: 最终输出分析
        self._analyze_final_output(processed_results)
        
        return processed_results
    
    def _analyze_processor_config(self):
        """分析图像处理器配置"""
        print(f"\n⚙️  Step 2: Image Processor Configuration")
        print(f"   Do resize: {getattr(self.image_processor, 'do_resize', 'N/A')}")
        print(f"   Size: {getattr(self.image_processor, 'size', 'N/A')}")
        print(f"   Image grid pinpoints: {getattr(self.image_processor, 'image_grid_pinpoints', 'N/A')}")
        print(f"   Resample: {getattr(self.image_processor, 'resample', 'N/A')}")
        print(f"   Do center crop: {getattr(self.image_processor, 'do_center_crop', 'N/A')}")
        print(f"   Crop size: {getattr(self.image_processor, 'crop_size', 'N/A')}")
        print(f"   Do rescale: {getattr(self.image_processor, 'do_rescale', 'N/A')}")
        print(f"   Rescale factor: {getattr(self.image_processor, 'rescale_factor', 'N/A')}")
        print(f"   Do normalize: {getattr(self.image_processor, 'do_normalize', 'N/A')}")
        print(f"   Image mean: {getattr(self.image_processor, 'image_mean', 'N/A')}")
        print(f"   Image std: {getattr(self.image_processor, 'image_std', 'N/A')}")
        print(f"   Do pad: {getattr(self.image_processor, 'do_pad', 'N/A')}")
        print(f"   Do convert RGB: {getattr(self.image_processor, 'do_convert_rgb', 'N/A')}")
        
        if hasattr(self.image_processor, 'image_grid_pinpoints'):
            print(f"   Grid pinpoints: {len(self.image_processor.image_grid_pinpoints)} resolutions")
            print(f"   Grid pinpoints: {self.image_processor.image_grid_pinpoints[:5]}...")
    
    def _analyze_resolution_selection(self, original_size: Tuple[int, int]) -> Tuple[int, int]:
        """分析最佳分辨率选择过程"""
        print(f"\n🎯 Step 3: Best Resolution Selection")
        
        # 获取image_grid_pinpoints
        if hasattr(self.image_processor, 'image_grid_pinpoints'):
            grid_pinpoints = self.image_processor.image_grid_pinpoints
            print(f"   Available resolutions: {len(grid_pinpoints)}")
            
            # 模拟select_best_resolution逻辑
            width, height = original_size
            original_area = width * height
            best_resolution = None
            min_wasted_resolution = float('inf')
            
            print(f"   Original size: {width} x {height} (area: {original_area})")
            
            for i, (pin_width, pin_height) in enumerate(grid_pinpoints):
                # 计算如何缩放原始图像以适应这个分辨率
                scale = min(pin_width / width, pin_height / height)
                scaled_width = int(width * scale)
                scaled_height = int(height * scale)
                wasted_resolution = (pin_width * pin_height) - (scaled_width * scaled_height)
                
                if wasted_resolution < min_wasted_resolution:
                    min_wasted_resolution = wasted_resolution
                    best_resolution = (pin_height, pin_width)  # 注意：返回格式是(height, width)
                    best_scale = scale
                
                if i < 3:  # 显示前几个计算过程
                    print(f"   Candidate {pin_width}x{pin_height}: scale={scale:.3f}, "
                          f"scaled={scaled_width}x{scaled_height}, wasted={wasted_resolution}")
            
            print(f"   ✅ Best resolution: {best_resolution} (scale: {best_scale:.3f})")
            print(f"   ✅ Minimum wasted pixels: {min_wasted_resolution}")
            
            return best_resolution
        else:
            # 如果没有grid_pinpoints，使用默认的336x336
            print(f"   Using default size: 336x336")
            return (336, 336)
    
    def _analyze_multiscale_processing(self, image: Image.Image) -> dict:
        """分析多尺度处理过程"""
        print(f"\n🔄 Step 4: Multi-scale Processing")
        
        # 直接调用image_processor获取结果
        processed_inputs = self.image_processor(image, return_tensors="pt")
        pixel_values = processed_inputs["pixel_values"]
        
        print(f"   Final tensor shape: {pixel_values.shape}")
        
        if len(pixel_values.shape) == 5:
            batch, num_scales, channels, height, width = pixel_values.shape
            print(f"   Batch size: {batch}")
            print(f"   Number of scales: {num_scales}")
            print(f"   Channels: {channels}")
            print(f"   Height: {height}")
            print(f"   Width: {width}")
            
            print(f"\n   💡 Multi-scale explanation:")
            print(f"   - LlavaNext uses {num_scales} different scales for better feature extraction")
            print(f"   - Each scale captures different levels of detail")
            print(f"   - Base resolution: {height}x{width}")
            
            # 分析每个尺度的统计信息
            for scale_idx in range(num_scales):
                scale_tensor = pixel_values[0, scale_idx]  # [C, H, W]
                mean_val = scale_tensor.mean().item()
                std_val = scale_tensor.std().item()
                min_val = scale_tensor.min().item()
                max_val = scale_tensor.max().item()
                
                print(f"   Scale {scale_idx}: mean={mean_val:.3f}, std={std_val:.3f}, "
                      f"range=[{min_val:.3f}, {max_val:.3f}]")
        
        # 返回处理结果和额外信息
        return {
            "pixel_values": pixel_values,
            "image_sizes": processed_inputs.get("image_sizes"),
            "processed_inputs": processed_inputs
        }
    
    def _analyze_final_output(self, results: dict):
        """分析最终输出"""
        print(f"\n📊 Step 5: Final Output Analysis")
        
        pixel_values = results["pixel_values"]
        image_sizes = results["image_sizes"]
        
        print(f"   Final pixel_values shape: {pixel_values.shape}")
        print(f"   Data type: {pixel_values.dtype}")
        print(f"   Image sizes tensor: {image_sizes}")
        print(f"   Image sizes shape: {image_sizes.shape if image_sizes is not None else 'None'}")
        
        # 内存使用分析
        num_elements = pixel_values.numel()
        memory_mb = num_elements * 4 / (1024 * 1024)  # 假设float32
        print(f"   Total elements: {num_elements:,}")
        print(f"   Approximate memory: {memory_mb:.2f} MB")
        
        # 数据分布分析
        print(f"\n   📈 Data Distribution:")
        print(f"   Global mean: {pixel_values.mean().item():.6f}")
        print(f"   Global std: {pixel_values.std().item():.6f}")
        print(f"   Global min: {pixel_values.min().item():.6f}")
        print(f"   Global max: {pixel_values.max().item():.6f}")
    
    def visualize_processing_steps(self, image: Image.Image, save_path: str = None):
        """可视化处理步骤"""
        print(f"\n🎨 Step 6: Visualization")
        
        # 获取处理结果
        processed_inputs = self.image_processor(image, return_tensors="pt")
        pixel_values = processed_inputs["pixel_values"]
        
        if len(pixel_values.shape) == 5:
            batch, num_scales, channels, height, width = pixel_values.shape
            
            # 创建子图
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'LlavaNext Multi-scale Processing\nOriginal: {image.size} → Processed: {num_scales} scales of {height}×{width}', 
                        fontsize=14)
            
            # 显示原始图像
            axes[0, 0].imshow(image)
            axes[0, 0].set_title(f'Original Image\n{image.size}')
            axes[0, 0].axis('off')
            
            # 显示前5个尺度
            for scale_idx in range(min(5, num_scales)):
                row = (scale_idx + 1) // 3
                col = (scale_idx + 1) % 3
                
                # 转换tensor为可显示格式
                scale_tensor = pixel_values[0, scale_idx]  # [C, H, W]
                
                # 反归一化（假设使用ImageNet标准化）
                mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
                std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
                
                unnormalized = scale_tensor * std + mean
                unnormalized = torch.clamp(unnormalized, 0, 1)
                
                # 转换为HWC格式用于显示
                img_to_show = unnormalized.permute(1, 2, 0).numpy()
                
                axes[row, col].imshow(img_to_show)
                axes[row, col].set_title(f'Scale {scale_idx}\n{height}×{width}')
                axes[row, col].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"   Visualization saved to: {save_path}")
            else:
                plt.show()
        
        print(f"   ✅ Visualization completed")
    
    def compare_with_simple_resize(self, image: Image.Image):
        """对比简单resize和LlavaNext处理的差异"""
        print(f"\n🔍 Step 7: Comparison with Simple Resize")
        
        # LlavaNext处理
        llava_result = self.image_processor(image, return_tensors="pt")
        llava_pixel_values = llava_result["pixel_values"]
        
        # 简单resize处理
        simple_resized = image.resize((336, 336))
        simple_array = np.array(simple_resized).transpose(2, 0, 1)  # HWC -> CHW
        simple_tensor = torch.tensor(simple_array, dtype=torch.float32) / 255.0
        
        # 应用归一化
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        simple_normalized = (simple_tensor - mean) / std
        simple_final = simple_normalized.unsqueeze(0).unsqueeze(0)  # 添加batch和scale维度
        
        print(f"   LlavaNext shape: {llava_pixel_values.shape}")
        print(f"   Simple resize shape: {simple_final.shape}")
        
        print(f"\n   📊 Statistical Comparison:")
        print(f"   LlavaNext - mean: {llava_pixel_values.mean():.6f}, std: {llava_pixel_values.std():.6f}")
        print(f"   Simple resize - mean: {simple_normalized.mean():.6f}, std: {simple_normalized.std():.6f}")
        
        print(f"\n   💡 Key Differences:")
        print(f"   - LlavaNext: Multi-scale processing ({llava_pixel_values.shape[1]} scales)")
        print(f"   - Simple: Single scale processing")
        print(f"   - LlavaNext: Adaptive resolution selection")
        print(f"   - Simple: Fixed 336×336 resize")


def main():
    """主函数：运行完整分析"""
    # 创建分析器
    analyzer = ImageProcessorAnalysis()
    
    # 创建一个示例图像 (1100, 808)
    # 你可以替换为实际的图像路径
    example_image = Image.new('RGB', (1100, 808), color='red')
    # 或者使用实际图像：
    # example_image = Image.open("path/to/your/image.jpg")
    
    print("🚀 Starting LlavaNext Image Processor Analysis")
    print(f"📝 This analysis shows how an image of size (1100, 808) becomes pixel_values [1, 5, 3, 336, 336]")
    
    # 运行完整分析
    results = analyzer.analyze_full_pipeline(example_image)
    
    # 可视化处理步骤
    # analyzer.visualize_processing_steps(example_image)
    
    # 对比分析
    analyzer.compare_with_simple_resize(example_image)
    
    print("\n" + "=" * 80)
    print("🎉 Analysis completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()