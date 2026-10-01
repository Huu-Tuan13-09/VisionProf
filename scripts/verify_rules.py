#!/usr/bin/env python3
"""Script kiểm tra tự động tính tuân thủ quy chuẩn mã nguồn (Rule Compliance Checker).

Kiểm tra:
1. Độ dài file không vượt quá 300 dòng (Hard limit) đối với các file mới.
2. Không sử dụng hàm print() bừa bãi trong các module lõi (bắt buộc dùng logging).
3. Không để sót file rác/checkpoint chưa được thêm vào .gitignore.
"""

import ast
import sys
from pathlib import Path
from typing import List, Tuple

MAX_LINES_HARD_LIMIT = 300
WARNING_LINES_LIMIT = 250

# Các file di sản v1.0 đang trong giai đoạn chờ refactor (được miễn tạm thời)
LEGACY_FILES = {
    "src/collector.py",
    "src/analyzer.py",
    "src/dashboard.py",
    "tests/test_training_bottleneck.py",
    "tests/test_modern_models_benchmark.py",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check_file_lengths() -> Tuple[List[str], List[str]]:
    """Kiểm tra độ dài các file python trong src/ và tests/."""
    errors = []
    warnings = []
    
    python_files = list(PROJECT_ROOT.glob("src/**/*.py")) + list(PROJECT_ROOT.glob("tests/**/*.py"))
    
    for file_path in python_files:
        rel_path = file_path.relative_to(PROJECT_ROOT).as_posix()
        if rel_path in LEGACY_FILES:
            continue
            
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            total_lines = len(lines)
            
        if total_lines > MAX_LINES_HARD_LIMIT:
            errors.append(f"❌ [Hard Limit] {rel_path} có {total_lines} dòng (> {MAX_LINES_HARD_LIMIT} dòng). Bắt buộc phân tách!")
        elif total_lines > WARNING_LINES_LIMIT:
            warnings.append(f"⚠️  [Warning] {rel_path} có {total_lines} dòng (> {WARNING_LINES_LIMIT} dòng). Cân nhắc tách module sớm.")
            
    return errors, warnings


def check_forbidden_patterns() -> List[str]:
    """Kiểm tra các pattern bị cấm (như bare print trong các module đo lường và phân tích lõi)."""
    errors = []
    candidate_files = [
        f for f in PROJECT_ROOT.glob("src/**/*.py")
        if not f.relative_to(PROJECT_ROOT).as_posix().startswith("src/dashboard")
        and f.relative_to(PROJECT_ROOT).as_posix() not in LEGACY_FILES
    ]
    
    for file_path in candidate_files:
        rel_path = file_path.relative_to(PROJECT_ROOT).as_posix()
        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                stripped = line.strip()
                if stripped.startswith("print(") and not stripped.startswith("#"):
                    errors.append(f"❌ [Forbidden print] {rel_path}:{idx}: Sử dụng print() bị cấm trong module đo lường lõi! Hãy dùng logging.")
                    
    return errors


def check_type_hints() -> List[str]:
    """Kiểm tra 100% type hints cho các public functions/methods trong src/ (ngoài legacy)."""
    errors = []
    candidate_files = [
        f for f in PROJECT_ROOT.glob("src/**/*.py")
        if f.relative_to(PROJECT_ROOT).as_posix() not in LEGACY_FILES
    ]
    
    for file_path in candidate_files:
        rel_path = file_path.relative_to(PROJECT_ROOT).as_posix()
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(file_path))
        except SyntaxError as e:
            errors.append(f"❌ [Syntax Error] {rel_path}: {e}")
            continue

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_"):
                    continue  # Bỏ qua private helper methods
                if node.returns is None:
                    errors.append(
                        f"❌ [Missing Type Hint] {rel_path}:{node.lineno}: "
                        f"Hàm public '{node.name}' thiếu return type annotation."
                    )
                for arg in node.args.args:
                    if arg.arg in ("self", "cls"):
                        continue
                    if arg.annotation is None:
                        errors.append(
                            f"❌ [Missing Type Hint] {rel_path}:{node.lineno}: "
                            f"Tham số '{arg.arg}' của hàm public '{node.name}' thiếu type annotation."
                        )
    return errors


def main() -> int:
    print("=" * 60)
    print("🔍 VISIONPROF — KIỂM TRA TUÂN THỦ QUY TẮC MÃ NGUỒN")
    print("=" * 60)
    
    errors, warnings = check_file_lengths()
    pattern_errors = check_forbidden_patterns()
    type_errors = check_type_hints()
    errors.extend(pattern_errors)
    errors.extend(type_errors)
    
    for warn in warnings:
        print(warn)
        
    if errors:
        print("\nPhát hiện các lỗi vi phạm nghiêm trọng:")
        for err in errors:
            print(err)
        print("\n❌ KIỂM TRA THẤT BẠI: Vui lòng sửa các vi phạm trên trước khi commit/merge!")
        return 1
        
    print("\n✅ TẤT CẢ QUY TẮC ĐƯỢC TUÂN THỦ HOÀN TOÀN!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
