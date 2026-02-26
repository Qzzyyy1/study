"""
超参数搜索脚本 - 寻找最优的伪标签参数
支持网格搜索和随机搜索两种模式
"""

import os
import json
import subprocess
import argparse
from datetime import datetime
from itertools import product
import numpy as np

class HyperparameterSearch:
    def __init__(self, base_args, search_mode='grid'):
        self.base_args = base_args
        self.search_mode = search_mode
        self.results = []
        self.log_file = f'hyperparam_search_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
    def grid_search(self, param_grid):
        """网格搜索：遍历所有参数组合"""
        print("=" * 60)
        print("开始网格搜索 (Grid Search)")
        print("=" * 60)
        
        # 生成所有参数组合
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(product(*values))
        
        total = len(combinations)
        print(f"总共需要测试 {total} 组参数\n")
        
        for idx, combo in enumerate(combinations, 1):
            params = dict(zip(keys, combo))
            print(f"\n[{idx}/{total}] 测试参数组合:")
            for k, v in params.items():
                print(f"  {k}: {v}")
            
            result = self.run_experiment(params, idx)
            self.results.append(result)
            self.save_results()
            
        return self.get_best_result()
    
    def random_search(self, param_ranges, n_trials=20):
        """随机搜索：随机采样参数组合"""
        print("=" * 60)
        print(f"开始随机搜索 (Random Search) - {n_trials} 次试验")
        print("=" * 60)
        
        for idx in range(1, n_trials + 1):
            # 随机采样参数
            params = {}
            for key, (low, high) in param_ranges.items():
                if isinstance(low, float):
                    params[key] = np.random.uniform(low, high)
                else:
                    params[key] = np.random.randint(low, high + 1)
            
            print(f"\n[{idx}/{n_trials}] 测试参数组合:")
            for k, v in params.items():
                print(f"  {k}: {v}")
            
            result = self.run_experiment(params, idx)
            self.results.append(result)
            self.save_results()
            
        return self.get_best_result()
    
    def run_experiment(self, params, trial_id):
        """运行单次实验"""
        # 构建命令
        cmd = ['python', 'main.py']
        
        # 添加基础参数
        for key, value in self.base_args.items():
            cmd.extend([f'--{key}', str(value)])
        
        # 添加搜索参数
        for key, value in params.items():
            cmd.extend([f'--{key}', str(value)])
        
        print(f"\n执行命令: {' '.join(cmd)}")
        print("-" * 60)
        
        try:
            # 运行实验
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1小时超时
            )
            
            # 解析结果
            metrics = self.parse_output(result.stdout)
            
            experiment_result = {
                'trial_id': trial_id,
                'params': params,
                'metrics': metrics,
                'status': 'success' if result.returncode == 0 else 'failed',
                'timestamp': datetime.now().isoformat()
            }
            
            print("\n结果:")
            if metrics:
                for k, v in metrics.items():
                    print(f"  {k}: {v:.4f}")
            else:
                print("  未能解析到结果指标")
            
            return experiment_result
            
        except subprocess.TimeoutExpired:
            print("实验超时！")
            return {
                'trial_id': trial_id,
                'params': params,
                'metrics': {},
                'status': 'timeout',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"实验失败: {str(e)}")
            return {
                'trial_id': trial_id,
                'params': params,
                'metrics': {},
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def parse_output(self, output):
        """从输出中解析性能指标 - 增强版"""
        import re
        metrics = {}
        
        # 保存完整输出到文件，方便调试
        debug_file = f'output_debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(output)
        
        lines = output.split('\n')
        
        # 模式1: 查找 "指标名: 数值" 格式
        for line in lines:
            # OS*, OS, H-score等常见指标
            patterns = [
                r'OS\*[:\s]+([0-9.]+)',           # OS*: 0.8523 或 OS* 0.8523
                r'OS[:\s]+([0-9.]+)',             # OS: 0.8523
                r'H-score[:\s]+([0-9.]+)',        # H-score: 0.8523
                r'h-score[:\s]+([0-9.]+)',        # h-score: 0.8523
                r'accuracy[:\s]+([0-9.]+)',       # accuracy: 0.9012
                r'Accuracy[:\s]+([0-9.]+)',       # Accuracy: 0.9012
                r'known[_\s]acc[:\s]+([0-9.]+)',  # known_acc: 0.9012
                r'unknown[_\s]acc[:\s]+([0-9.]+)',# unknown_acc: 0.8012
                r'target[_\s]acc[:\s]+([0-9.]+)', # target_acc: 0.8523
            ]
            
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    value = float(match.group(1))
                    # 提取指标名称
                    metric_name = re.search(r'([a-zA-Z*_-]+)', pattern.replace('[:\\s]+', '').replace('([0-9.]+)', '')).group(1)
                    metrics[metric_name] = value
        
        # 模式2: 查找表格格式的输出
        # 例如: | OS* | 0.8523 |
        for line in lines:
            if '|' in line:
                parts = [p.strip() for p in line.split('|') if p.strip()]
                if len(parts) >= 2:
                    try:
                        metric_name = parts[0]
                        metric_value = float(parts[1])
                        metrics[metric_name] = metric_value
                    except:
                        pass
        
        # 模式3: 查找最后几行的数字（通常是最终结果）
        last_lines = lines[-20:]  # 检查最后20行
        for line in last_lines:
            # 查找所有浮点数
            numbers = re.findall(r'\b([0-9]+\.[0-9]+)\b', line)
            if numbers and any(keyword in line.lower() for keyword in ['test', 'final', 'result', 'performance']):
                # 如果这行包含关键词，尝试提取指标
                for num in numbers:
                    value = float(num)
                    if 0 < value <= 1:  # 假设指标在0-1之间
                        if 'final_score' not in metrics:
                            metrics['final_score'] = value
        
        return metrics
    
    def save_results(self):
        """保存结果到JSON文件"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump({
                'search_mode': self.search_mode,
                'base_args': self.base_args,
                'results': self.results,
                'best_result': self.get_best_result() if self.results else None
            }, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {self.log_file}")
    
    def get_best_result(self):
        """获取最佳结果"""
        if not self.results:
            return None
        
        # 过滤成功的实验
        successful = [r for r in self.results if r['status'] == 'success' and r['metrics']]
        
        if not successful:
            return None
        
        # 根据主要指标排序（优先使用OS*，其次OS，最后accuracy）
        def get_score(result):
            metrics = result['metrics']
            if 'OS*' in metrics:
                return metrics['OS*']
            elif 'OS' in metrics:
                return metrics['OS']
            elif 'accuracy' in metrics:
                return metrics['accuracy']
            elif 'h-score' in metrics:
                return metrics['h-score']
            else:
                return 0.0
        
        best = max(successful, key=get_score)
        return best
    
    def print_summary(self):
        """打印搜索总结"""
        print("\n" + "=" * 60)
        print("超参数搜索完成！")
        print("=" * 60)
        
        if not self.results:
            print("没有完成任何实验")
            return
        
        successful = [r for r in self.results if r['status'] == 'success' and r['metrics']]
        print(f"\n总实验次数: {len(self.results)}")
        print(f"成功次数: {len(successful)}")
        print(f"失败次数: {len(self.results) - len(successful)}")
        
        best = self.get_best_result()
        if best:
            print("\n" + "🏆 最佳参数组合:")
            print("-" * 60)
            for k, v in best['params'].items():
                print(f"  {k}: {v}")
            print("\n性能指标:")
            for k, v in best['metrics'].items():
                print(f"  {k}: {v:.4f}")
        else:
            print("\n未找到有效的最佳结果")
        
        print(f"\n详细结果已保存到: {self.log_file}")


def main():
    parser = argparse.ArgumentParser(description='WGDT超参数搜索')
    
    # 搜索模式
    parser.add_argument('--search_mode', type=str, choices=['grid', 'random'], default='grid',
                        help='搜索模式: grid(网格搜索) 或 random(随机搜索)')
    parser.add_argument('--n_trials', type=int, default=20,
                        help='随机搜索的试验次数')
    
    # 基础参数（不参与搜索）
    parser.add_argument('--source_dataset', type=str, default='PaviaU_7gt')
    parser.add_argument('--target_dataset', type=str, default='PaviaC_OS')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--seed', type=int, default=0)
    
    args = parser.parse_args()
    
    # 基础参数
    base_args = {
        'source_dataset': args.source_dataset,
        'target_dataset': args.target_dataset,
        'device': args.device,
        'epochs': args.epochs,
        'seed': args.seed,
    }
    
    # 创建搜索器
    searcher = HyperparameterSearch(base_args, search_mode=args.search_mode)
    
    if args.search_mode == 'grid':
        # 网格搜索的参数空间
        param_grid = {
            'pseudo_label_weight': [0.3, 0.5, 0.7, 1.0],
            'pseudo_label_threshold': [0.1, 0.2, 0.3, 0.4, 0.5]
        }
        
        print("网格搜索参数空间:")
        for k, v in param_grid.items():
            print(f"  {k}: {v}")
        
        best = searcher.grid_search(param_grid)
        
    else:  # random search
        # 随机搜索的参数范围
        param_ranges = {
            'pseudo_label_weight': (0.1, 1.0),      # [0.1, 1.0]
            'pseudo_label_threshold': (0.1, 0.6)    # [0.1, 0.6]
        }
        
        print("随机搜索参数范围:")
        for k, v in param_ranges.items():
            print(f"  {k}: {v}")
        
        best = searcher.random_search(param_ranges, n_trials=args.n_trials)
    
    # 打印总结
    searcher.print_summary()


if __name__ == '__main__':
    main()

