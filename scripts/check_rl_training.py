#!/usr/bin/env python3
"""
检查RL训练是否真实在训练模型
通过分析检查点文件中的训练统计信息来判断
"""

import json
import argparse
from pathlib import Path
import sys

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

def check_checkpoint_training(checkpoint_dir: str):
    """检查检查点文件，判断训练是否真实进行"""
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        print(f"❌ 检查点目录不存在: {checkpoint_dir}")
        return False
    
    # 检查training_stats.json
    stats_file = checkpoint_path / "training_stats.json"
    if not stats_file.exists():
        print(f"❌ 训练统计文件不存在: {stats_file}")
        return False
    
    try:
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)
    except Exception as e:
        print(f"❌ 无法读取训练统计文件: {e}")
        return False
    
    print("=" * 80)
    print("📊 RL训练检查报告")
    print("=" * 80)
    
    # 1. 检查基本统计信息
    step = stats.get("step", 0)
    print(f"\n✅ 训练步数: {step}")
    
    # 2. 检查loss历史
    policy_losses = stats.get("policy_losses", [])
    value_losses = stats.get("value_losses", [])
    
    print(f"\n📈 Loss统计:")
    print(f"   Policy losses记录数: {len(policy_losses)}")
    print(f"   Value losses记录数: {len(value_losses)}")
    
    if len(policy_losses) > 0:
        non_zero_policy = [l for l in policy_losses if abs(l) > 1e-10]
        print(f"   非零policy loss数量: {len(non_zero_policy)}/{len(policy_losses)}")
        if len(non_zero_policy) > 0:
            print(f"   Policy loss范围: [{min(policy_losses):.6f}, {max(policy_losses):.6f}]")
            print(f"   平均policy loss: {sum(policy_losses)/len(policy_losses):.6f}")
        else:
            print(f"   ⚠️  所有policy loss都为0！")
    
    if len(value_losses) > 0:
        non_zero_value = [l for l in value_losses if abs(l) > 1e-10]
        print(f"   非零value loss数量: {len(non_zero_value)}/{len(value_losses)}")
        if len(non_zero_value) > 0:
            print(f"   Value loss范围: [{min(value_losses):.6f}, {max(value_losses):.6f}]")
            print(f"   平均value loss: {sum(value_losses)/len(value_losses):.6f}")
        else:
            print(f"   ⚠️  所有value loss都为0！")
    
    # 3. 检查KL散度
    kl_divergences = stats.get("kl_divergences", [])
    print(f"\n📊 KL散度统计:")
    print(f"   KL散度记录数: {len(kl_divergences)}")
    if len(kl_divergences) > 0:
        non_zero_kl = [k for k in kl_divergences if k is not None and abs(k) > 1e-10]
        print(f"   非零KL散度数量: {len(non_zero_kl)}/{len(kl_divergences)}")
        if len(non_zero_kl) > 0:
            print(f"   KL散度范围: [{min(kl_divergences):.6f}, {max(kl_divergences):.6f}]")
            print(f"   平均KL散度: {sum(kl_divergences)/len(kl_divergences):.6f}")
        else:
            print(f"   ⚠️  所有KL散度都为0或接近0！")
    
    # 4. 检查奖励
    total_rewards = stats.get("total_rewards", [])
    print(f"\n🎁 奖励统计:")
    print(f"   奖励记录数: {len(total_rewards)}")
    if len(total_rewards) > 0:
        print(f"   奖励范围: [{min(total_rewards):.4f}, {max(total_rewards):.4f}]")
        print(f"   平均奖励: {sum(total_rewards)/len(total_rewards):.4f}")
        print(f"   标准差: {(sum((r - sum(total_rewards)/len(total_rewards))**2 for r in total_rewards) / len(total_rewards))**0.5:.4f}")
    
    # 5. 综合判断
    print("\n" + "=" * 80)
    print("🔍 训练状态诊断:")
    print("=" * 80)
    
    is_training = True
    issues = []
    
    # 检查1: Loss是否都为0
    if len(policy_losses) > 0:
        all_policy_zero = all(abs(l) < 1e-10 for l in policy_losses)
        if all_policy_zero:
            is_training = False
            issues.append("⚠️  所有policy loss都为0，模型可能没有计算损失")
    
    # 检查2: KL散度是否为0
    if len(kl_divergences) > 0:
        all_kl_zero = all(k is None or abs(k) < 1e-10 for k in kl_divergences)
        if all_kl_zero:
            issues.append("⚠️  所有KL散度都为0，策略可能没有更新（policy和ref_model可能相同）")
    
    # 检查3: Loss是否有变化
    if len(policy_losses) > 10:
        recent_losses = policy_losses[-10:]
        if all(abs(l - recent_losses[0]) < 1e-10 for l in recent_losses):
            issues.append("⚠️  最近10步的policy loss完全相同，可能模型没有在训练")
    
    # 检查4: 奖励是否有变化
    if len(total_rewards) > 10:
        recent_rewards = total_rewards[-10:]
        reward_std = (sum((r - sum(recent_rewards)/len(recent_rewards))**2 for r in recent_rewards) / len(recent_rewards))**0.5
        if reward_std < 0.01:
            issues.append("⚠️  奖励变化很小，可能奖励计算或归一化有问题")
    
    # 输出诊断结果
    if is_training and len(issues) == 0:
        print("✅ 训练状态正常：")
        print("   - Loss值非零")
        print("   - KL散度非零")
        print("   - Loss有变化")
        print("   - 奖励有变化")
        return True
    else:
        print("⚠️  发现潜在问题：")
        for issue in issues:
            print(f"   {issue}")
        
        if not is_training:
            print("\n❌ 结论：模型可能没有真实训练")
        else:
            print("\n⚠️  结论：训练可能在运行，但存在异常")
        return False

def main():
    parser = argparse.ArgumentParser(description="检查RL训练是否真实在训练模型")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/rl_model/checkpoint-1000",
        help="检查点目录路径"
    )
    
    args = parser.parse_args()
    
    result = check_checkpoint_training(args.checkpoint_dir)
    
    sys.exit(0 if result else 1)

if __name__ == "__main__":
    main()

