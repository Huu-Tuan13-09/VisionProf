"""Test tự động kiểm tra tính tuân thủ quy tắc kiến trúc và code standards."""

import subprocess
import sys
from pathlib import Path

def test_verify_rules_script():
    """Kiểm tra toàn bộ codebase không có file mới vi phạm hard limit > 300 dòng hoặc bare print."""
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "scripts" / "verify_rules.py"
    
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        cwd=str(repo_root)
    )
    
    assert result.returncode == 0, f"Quy tắc mã nguồn bị vi phạm:\n{result.stdout}\n{result.stderr}"
