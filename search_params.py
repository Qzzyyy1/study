import json
import itertools
import subprocess
import re
import csv
import os

# 配置路径
PARAMS_FILE = 'datasets/dataset_params.json'
RESULTS_CSV = 'search_results.csv'
TARGET_DATASET = 'Houston18_OS'

# 1. 定义要搜索的超参数空间 (您可以根据机器算力和时间自行增删)
# 这里的组合数为 2 * 3 * 3 * 3 = 54 种组合
search_space = {
    "lr_domain": [0.0001, 0.0005],           # 域适应学习率
    "domain_loss_weight": [0.1, 0.5, 1.0],   # 对齐权重 (建议偏小)
    "radius_loss_weight": [0.5, 1.0, 1.5],   # 紧凑度惩罚 (建议偏大)
    "radius_margin": [0.01, 0.05, 0.1],      # 半径边界 (建议偏小)
    "alpha": [0.01]                          # 拒绝阈值系数
}

def load_params():
    with open(PARAMS_FILE, 'r') as f:
        return json.load(f)

def save_params(params_dict):
    with open(PARAMS_FILE, 'w') as f:
        json.dump(params_dict, f, indent=4)

def parse_result_from_output(output):
    # 用正则表达式从控制台输出中提取字典内容
    # 匹配最后一个形如 { "oa": ..., "hos": ... } 的结构
    match = re.search(r'\{[^{}]*"oa":.*?"hos":.*\}', output, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            return None
    return None

def main():
    keys = list(search_space.keys())
    values = list(search_space.values())
    combinations = list(itertools.product(*values))
    
    print(f"总计需要运行 {len(combinations)} 组参数组合...")
    
    best_hos = -1
    best_params = None
    all_results = []
    
    # 写入 CSV 表头
    with open(RESULTS_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(keys + ['oa', 'aa', 'oa_known', 'aa_known', 'unknown', 'hos'])
    
    for i, combo in enumerate(combinations):
        current_setup = dict(zip(keys, combo))
        print(f"\n[{i+1}/{len(combinations)}] 正在测试参数: {current_setup}")
        
        # 读取并更新 JSON 文件
        full_params = load_params()
        for k, v in current_setup.items():
            full_params[TARGET_DATASET][k] = v
        save_params(full_params)
        
        # 运行 main.py
        cmd = ['python', 'main.py', '--source_dataset', 'Houston13_7gt', '--target_dataset', 'Houston18_OS']
        try:
            # 捕获标准输出和标准错误
            process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            output = process.stdout
            
            # 解析输出结果
            metrics = parse_result_from_output(output)
            
            if metrics:
                print(f"--> 结果: OA={metrics.get('oa',0):.4f}, Unknown={metrics.get('unknown',0):.4f}, HOS={metrics.get('hos',0):.4f}")
                
                # 记录到 CSV
                with open(RESULTS_CSV, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(list(combo) + [
                        metrics.get('oa', 0), metrics.get('aa', 0), 
                        metrics.get('oa_known', 0), metrics.get('aa_known', 0), 
                        metrics.get('unknown', 0), metrics.get('hos', 0)
                    ])
                
                # 更新最优记录
                if metrics.get('hos', 0) > best_hos:
                    best_hos = metrics['hos']
                    best_params = current_setup
            else:
                print("--> 错误: 未能在输出中找到有效的结果 JSON 格式，请检查报错。")
                
        except Exception as e:
            print(f"--> 运行出错: {e}")

    print("\n" + "="*50)
    print(f"搜索完成！最佳 HOS 值为: {best_hos:.4f}")
    print(f"最佳参数组合为: {best_params}")
    print(f"详细的所有结果已保存至 {RESULTS_CSV}")

if __name__ == '__main__':
    main()