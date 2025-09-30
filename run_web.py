#!/usr/bin/env python3
"""
DeepHallu Web Interface Launcher
启动DeepHallu的Web界面，用于数据集浏览和模型对比
"""

import os
import sys
import uvicorn
from pathlib import Path

# 添加src目录到Python路径
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def main():
    """启动Web服务器"""
    print("🚀 Starting DeepHallu Web Interface...")
    print("📊 Dataset Browser and Model Comparison Tool")
    print("-" * 50)
    
    # 检查数据集路径
    mme_path = project_root / "data" / "mme" / "MME_Benchmark_release_version" / "MME_Benchmark"
    if not mme_path.exists():
        print(f"⚠️  Warning: MME dataset not found at {mme_path}")
        print("   Please ensure the dataset is properly configured.")
    else:
        print(f"✅ MME dataset found at {mme_path}")
    
    # 检查HuggingFace缓存路径
    hf_home = os.environ.get("HF_HOME", "/DATA2/HuggingFace")
    if not Path(hf_home).exists():
        print(f"⚠️  Warning: HuggingFace cache directory not found at {hf_home}")
        print("   Models will be downloaded to default location.")
    else:
        print(f"✅ HuggingFace cache found at {hf_home}")
    
    print("-" * 50)
    print("🌐 Starting server at: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("❤️  Health Check: http://localhost:8000/health")
    print("-" * 50)
    
    try:
        uvicorn.run(
            "deephallu.web.app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down DeepHallu Web Interface...")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()