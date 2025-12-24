#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置检查脚本

运行此脚本以验证实验配置是否正确
"""

import os
import sys
from pathlib import Path

def check_model_path(config_path, model_key="model.path"):
    """检查模型路径是否存在"""
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model_path = config['model']['path']
        expanded_path = os.path.expanduser(model_path)
        
        if os.path.exists(expanded_path):
            print(f"✓ 模型路径存在: {model_path}")
            return True
        else:
            print(f"✗ 模型路径不存在: {model_path}")
            print(f"  展开后的路径: {expanded_path}")
            return False
    except Exception as e:
        print(f"✗ 检查模型路径时出错: {e}")
        return False


def check_gpu_availability():
    """检查GPU是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✓ 检测到 {gpu_count} 个GPU")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"  GPU {i}: {gpu_name}")
            return gpu_count
        else:
            print("✗ 未检测到GPU")
            return 0
    except ImportError:
        print("⚠ PyTorch未安装，无法检查GPU")
        return None


def check_dependencies():
    """检查必要的依赖是否已安装"""
    required_packages = [
        'pandas',
        'numpy',
        'datasets',
        'hydra',
        'ray',
        'verl'
    ]
    
    all_installed = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} 已安装")
        except ImportError:
            print(f"✗ {package} 未安装")
            all_installed = False
    
    return all_installed


def check_config_consistency():
    """检查两个阶段的配置是否一致"""
    try:
        import yaml
        
        with open('configs/stage1_config.yaml', 'r') as f:
            config1 = yaml.safe_load(f)
        
        with open('configs/stage3_config.yaml', 'r') as f:
            config3 = yaml.safe_load(f)
        
        # 检查关键配置是否一致
        checks = [
            ('model.path', config1['model']['path'], config3['model']['path']),
            ('temperature', config1['rollout']['temperature'], config3['rollout']['temperature']),
            ('top_p', config1['rollout']['top_p'], config3['rollout']['top_p']),
        ]
        
        all_consistent = True
        for key, val1, val3 in checks:
            if val1 == val3:
                print(f"✓ {key} 一致: {val1}")
            else:
                print(f"✗ {key} 不一致: stage1={val1}, stage3={val3}")
                all_consistent = False
        
        return all_consistent
    except Exception as e:
        print(f"✗ 检查配置一致性时出错: {e}")
        return False


def check_disk_space():
    """检查磁盘空间是否足够"""
    try:
        import shutil
        stat = shutil.disk_usage('outputs')
        free_gb = stat.free / (1024**3)
        
        # 估算需要的空间：约50GB
        required_gb = 50
        
        if free_gb >= required_gb:
            print(f"✓ 磁盘空间充足: {free_gb:.1f} GB 可用 (需要约 {required_gb} GB)")
            return True
        else:
            print(f"⚠ 磁盘空间可能不足: {free_gb:.1f} GB 可用 (建议至少 {required_gb} GB)")
            return False
    except Exception as e:
        print(f"⚠ 检查磁盘空间时出错: {e}")
        return None


def main():
    """主函数"""
    print("=" * 80)
    print("Token 2048 最佳思考实验 - 配置检查")
    print("=" * 80)
    print()
    
    # 切换到脚本所在目录
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)
    print(f"工作目录: {script_dir}\n")
    
    all_ok = True
    
    # 检查1：依赖
    print("检查1：Python依赖")
    print("-" * 80)
    if not check_dependencies():
        all_ok = False
        print("\n请安装缺失的依赖:")
        print("  pip install -r requirements.txt")
    print()
    
    # 检查2：GPU
    print("检查2：GPU可用性")
    print("-" * 80)
    gpu_count = check_gpu_availability()
    if gpu_count == 0:
        print("\n⚠ 警告：未检测到GPU，实验将无法运行")
        all_ok = False
    print()
    
    # 检查3：配置文件
    print("检查3：配置文件")
    print("-" * 80)
    
    if not os.path.exists('configs/stage1_config.yaml'):
        print("✗ configs/stage1_config.yaml 不存在")
        all_ok = False
    else:
        print("✓ configs/stage1_config.yaml 存在")
        if not check_model_path('configs/stage1_config.yaml'):
            all_ok = False
            print("  请修改 configs/stage1_config.yaml 中的 model.path")
    
    if not os.path.exists('configs/stage3_config.yaml'):
        print("✗ configs/stage3_config.yaml 不存在")
        all_ok = False
    else:
        print("✓ configs/stage3_config.yaml 存在")
    print()
    
    # 检查4：配置一致性
    print("检查4：配置一致性")
    print("-" * 80)
    if not check_config_consistency():
        print("\n⚠ 警告：两个阶段的配置不一致，可能导致实验结果不可比")
    print()
    
    # 检查5：磁盘空间
    print("检查5：磁盘空间")
    print("-" * 80)
    check_disk_space()
    print()
    
    # 总结
    print("=" * 80)
    if all_ok:
        print("✓ 所有检查通过！可以开始实验")
        print("\n运行实验:")
        print("  bash run_experiment.sh")
    else:
        print("✗ 部分检查未通过，请修复上述问题后再运行实验")
        sys.exit(1)
    print("=" * 80)


if __name__ == "__main__":
    main()

