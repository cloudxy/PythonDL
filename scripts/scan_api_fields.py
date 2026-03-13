"""
扫描 API 中使用的字段
分析所有 API 路由文件中对 model 字段的引用
"""
import os
import re
from pathlib import Path
from collections import defaultdict

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
API_DIR = PROJECT_ROOT / "app" / "api" / "v1"
MODELS_DIR = PROJECT_ROOT / "app" / "models"

# 存储结果
api_field_usage = defaultdict(lambda: defaultdict(list))
model_fields = defaultdict(list)

def scan_api_files():
    """扫描所有 API 文件"""
    print(f"扫描 API 目录：{API_DIR}")
    
    for py_file in API_DIR.glob("**/*.py"):
        if py_file.name.startswith("__"):
            continue
            
        print(f"\n分析文件：{py_file.relative_to(PROJECT_ROOT)}")
        
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            
        # 查找 model 导入
        model_imports = re.findall(r'from app\.models\.\w+ import (\w+)', content)
        model_imports += re.findall(r'from app\.models import (\w+)', content)
        
        # 查找字段使用
        for line_num, line in enumerate(lines, 1):
            # 跳过注释
            if line.strip().startswith('#'):
                continue
                
            # 查找 model 字段访问
            field_patterns = [
                r'\.(\w+)\s*=',  # 赋值：.field =
                r'\.(\w+)(?:\)|,|\]|\s)',  # 访问：.field)
                r'select\(\w+\.(\w+)',  # select 语句
                r'where\(\w+\.(\w+)',  # where 语句
                r'order_by\(\w+\.(\w+)',  # order_by 语句
            ]
            
            for pattern in field_patterns:
                matches = re.findall(pattern, line)
                for match in matches:
                    if match not in ['py', 'json', 'http']:  # 排除常见非字段名
                        api_field_usage[py_file.name][match].append(line_num)

def scan_model_files():
    """扫描所有 Model 文件"""
    print(f"\n\n扫描 Model 目录：{MODELS_DIR}")
    
    for py_file in MODELS_DIR.glob("**/*.py"):
        if py_file.name.startswith("__"):
            continue
            
        print(f"\n分析 Model: {py_file.relative_to(PROJECT_ROOT)}")
        
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 查找 Column 定义
        column_patterns = [
            r'(\w+)\s*=\s*Column\(',  # 标准 Column 定义
            r'(\w+)\s*=\s*relationship\(',  # relationship 定义
        ]
        
        model_name = py_file.stem
        for pattern in column_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if match not in ['Column', 'relationship', 'func']:
                    model_fields[model_name].append(match)

def generate_report():
    """生成报告"""
    report = []
    report.append("# PythonDL API 字段使用报告\n")
    report.append("**生成时间**: 2026-03-13\n\n")
    
    # Model 字段统计
    report.append("## 📊 Model 字段统计\n\n")
    report.append("| Model | 字段数 | 字段列表 |\n")
    report.append("|-------|--------|----------|\n")
    
    for model_name in sorted(model_fields.keys()):
        fields = model_fields[model_name]
        field_list = ", ".join(fields[:10])  # 只显示前 10 个
        if len(fields) > 10:
            field_list += f"... 等共{len(fields)}个"
        report.append(f"| {model_name} | {len(fields)} | {field_list} |\n")
    
    # API 字段使用统计
    report.append("\n## 🔍 API 字段使用情况\n\n")
    
    for api_file in sorted(api_field_usage.keys()):
        report.append(f"### {api_file}\n\n")
        report.append("| 字段 | 使用次数 | 行号 |\n")
        report.append("|------|----------|------|\n")
        
        field_counts = sorted(
            api_field_usage[api_file].items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        for field, lines in field_counts[:20]:  # 只显示前 20 个
            line_str = ", ".join(map(str, lines[:5]))
            if len(lines) > 5:
                line_str += f"... 等共{len(lines)}次"
            report.append(f"| {field} | {len(lines)} | {line_str} |\n")
        
        report.append("\n")
    
    # 未使用的字段
    report.append("\n## ⚠️ 可能未使用的字段\n\n")
    report.append("以下字段在 Model 中定义，但在 API 中未发现使用：\n\n")
    
    for model_name in sorted(model_fields.keys()):
        fields = set(model_fields[model_name])
        used_fields = set(api_field_usage.keys())
        # 简单对比，实际需要更复杂的分析
        unused = fields - used_fields
        if unused:
            report.append(f"- **{model_name}**: {', '.join(list(unused)[:10])}\n")
    
    return "".join(report)

if __name__ == "__main__":
    print("=" * 60)
    print("开始扫描 API 和 Model 字段...")
    print("=" * 60)
    
    scan_model_files()
    scan_api_files()
    
    print("\n" + "=" * 60)
    print("生成报告...")
    print("=" * 60)
    
    report = generate_report()
    
    # 保存报告
    report_file = PROJECT_ROOT / "docs" / "API_FIELD_USAGE_REPORT.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✅ 报告已保存到：{report_file}")
    print(f"📊 分析了 {len(model_fields)} 个 Model")
    print(f"📊 分析了 {len(api_field_usage)} 个 API 文件")
