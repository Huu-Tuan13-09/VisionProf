#!/usr/bin/env python3
"""Run all unit tests in the tests/ directory."""

import importlib
import inspect
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Legacy test files that depend on v1 monolithic layout
SKIP_FILES = {
    "test_modern_models_benchmark.py",
    "test_training_bottleneck.py",
}


def discover_and_run_tests() -> Tuple[int, int, List[str]]:
    """Discover all test functions in tests/ and execute them.

    Returns:
        Tuple of (passed_count, failed_count, failed_test_names)
    """
    tests_dir = PROJECT_ROOT / "tests"
    test_files = sorted(tests_dir.glob("test_*.py"))

    passed = 0
    failed = 0
    failures: List[str] = []

    print("=" * 65)
    print("🧪 VISIONPROF — CHẠY KIỂM THỬ TỰ ĐỘNG (UNIT TESTS)")
    print("=" * 65)

    for test_file in test_files:
        if test_file.name in SKIP_FILES:
            print(f"⏩ Bỏ qua test di sản v1.0: {test_file.name}")
            continue

        module_name = f"tests.{test_file.stem}"
        try:
            mod = importlib.import_module(module_name)
        except Exception as exc:
            print(f"❌ Không thể nạp module {module_name}: {exc}")
            failed += 1
            failures.append(f"{module_name}: import error ({exc})")
            continue

        test_functions = [
            (name, func)
            for name, func in inspect.getmembers(mod, inspect.isfunction)
            if name.startswith("test_")
        ]

        print(f"\n📂 {test_file.name} ({len(test_functions)} tests):")
        for func_name, func in test_functions:
            try:
                func()
                print(f"  ✅ {func_name} — PASS")
                passed += 1
            except AssertionError as ae:
                print(f"  ❌ {func_name} — ASSERTION FAILED: {ae}")
                failed += 1
                failures.append(f"{module_name}.{func_name}: {ae}")
            except Exception as e:
                print(f"  ❌ {func_name} — ERROR: {e}")
                failed += 1
                failures.append(f"{module_name}.{func_name}: {e}")

    print("\n" + "=" * 65)
    print(f"📊 KẾT QUẢ: {passed} PASSED | {failed} FAILED")
    print("=" * 65)

    return passed, failed, failures


def main() -> None:
    """CLI entrypoint for running unit tests."""
    _, failed, _ = discover_and_run_tests()
    if failed > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
