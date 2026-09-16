#!/usr/bin/env python3
"""列出 test/ 下"纯 CPU 测试类"的类名，输出一条 pytest 的 -k 排除表达式。

由 .ci/ppu/cuda_only_filter.sh 调用（PPU 门禁的「只跑 CUDA」第 2 层过滤），
不 import torch，也不需要装好 torch 就能跑，可单独执行自查：

    python .ci/ppu/list_cpu_only_test_classes.py test

背景与设计取舍见 .ci/ppu/cuda_only_filter.sh 的头注释；这里只说清单是怎么算出来的：

来源 1（主力）copy_tests(<Template>, <Cls>, "cpu") 的目标类名。
    上游造"整份 CPU 拷贝"的标准手法：一个 CommonTemplate 拷成 CpuTests + GPUTests
    两份，CPU 那份就是门禁里最费时间、又与 PPU 无关的部分。
来源 2（约定）类名形如 *CpuTest* / *CPUTest* / *TestCpu / *TestCPU 的类。
    兜住不经 copy_tests、直接继承 Template 或自己写一份的 CPU 套件，两种词序都要收：
    test_fused_attention.py 的 SDPAPatternRewriterCpuTests（Cpu 在 Test 前）与
    test_c10d_functional_native.py 的 CompileTestCPU（CPU 在 Test 后，真的跑 device="cpu"）。
来源 3（补充）EXTRA_CPU_ONLY_CLASSES：既不走 copy_tests、类名也不含 CpuTest 的纯
    CPU 套件，只能手工列。列进来的名字会校验"当前仍存在于 test/ 下"，过期即硬失败。

输出前会做一次子串归约：pytest 的 -k 是子串匹配，清单里若 A 是 B 的子串则 B 是多余的
（例如 `not CpuTests` 一项就盖住 DynamicShapesCpuTests / FreezingCpuTests /
EfficientConvBNEvalCpuTests…），只保留最短的那批，表达式更短、也天然覆盖上游后续新增
的同名系列。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


# 来源 3：手工补充。只列"确实会被本仓库某条门禁跑到"的类 —— 已在 full_ut_test.sh 的
# 文件级 --exclude 里剔掉的文件不必列（如 test_cpu_repro.py 的 CPUReproTests、
# test_pallas.py 的 PallasTestsCPU、test_triton_cpu_backend.py 的 CpuTritonTests）。
EXTRA_CPU_ONLY_CLASSES = (
    # inductor/test_fused_attention.py：`if HAS_CPU:` 下的 CPU 版 SDPA pattern rewriter。
    # 静态那份叫 SDPAPatternRewriterCpuTests（来源 2 收得到），动态这份名字里没有 CpuTest
    "SDPAPatternRewriterCpuDynamicTests",
    # test_transformers.py：整个类都是 CPU 侧 SDPA（fused attention 的 CPU 实现）
    "TestSDPACpuOnly",
    # test_autocast.py：CPU autocast，GPU 侧另有 TestAutocastGPU
    "TestAutocastCPU",
)

# copy_tests(<Template>, <Cls>, "cpu"[, ...])；re.S 兼容上游把参数折行的写法
COPY_TESTS_RE = re.compile(
    r"""copy_tests\(\s*[\w.]+\s*,\s*(\w+)\s*,\s*['"]cpu['"]""", re.S
)
# 命名约定：*CpuTest* / *CPUTest*（Cpu 在 Test 前）与 *TestCpu* / *TestCPU*（在 Test 后）。
# 两种词序上游都在用，只写一种会漏（漏过 CompileTestCPU 那类真跑 device="cpu" 的套件）。
CONVENTION_RE = re.compile(
    r"^\s*class\s+(\w*(?:(?:Cpu|CPU)Test|Test(?:Cpu|CPU))\w*)\s*[(:]", re.M
)
# 用于校验来源 3 的名字是否还在
ANY_CLASS_RE = re.compile(r"^\s*class\s+(\w+)\s*[(:]", re.M)


def collect(test_dir: Path) -> tuple[set[str], set[str]]:
    """返回 (CPU 专属类名集合, test/ 下所有类名集合)。"""
    cpu_only: set[str] = set()
    declared: set[str] = set()
    for path in sorted(test_dir.rglob("*.py")):
        # errors="ignore"：test/ 下有少量刻意构造的非 UTF-8 / 二进制样例文件
        src = path.read_text(encoding="utf-8", errors="ignore")
        cpu_only.update(COPY_TESTS_RE.findall(src))
        cpu_only.update(CONVENTION_RE.findall(src))
        declared.update(ANY_CLASS_RE.findall(src))
    return cpu_only, declared


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"用法: {argv[0]} <test 目录>", file=sys.stderr)
        return 2

    test_dir = Path(argv[1])
    if not test_dir.is_dir():
        print(f"[cuda-only] 目录不存在: {test_dir}", file=sys.stderr)
        return 1

    cpu_only, declared = collect(test_dir)

    stale = sorted(name for name in EXTRA_CPU_ONLY_CLASSES if name not in declared)
    if stale:
        print(
            "[cuda-only] EXTRA_CPU_ONLY_CLASSES 已过期，请修正 "
            ".ci/ppu/list_cpu_only_test_classes.py：\n"
            "  以下类名在 test/ 下已不存在（被上游改名或删除了）: " + " ".join(stale),
            file=sys.stderr,
        )
        return 1
    cpu_only.update(EXTRA_CPU_ONLY_CLASSES)

    # 上游若把 copy_tests 与命名约定一起重构掉，这里会算出空集：那意味着本过滤器已经
    # 失效，必须硬失败，而不是让 CPU 用例悄悄跑回门禁里。
    if not cpu_only:
        print(
            '[cuda-only] 没有从 test/ 扫到任何 CPU 专属测试类：copy_tests(..., "cpu") '
            "与 *CpuTest* 命名约定可能都被上游改掉了，\n"
            "  请修正 .ci/ppu/list_cpu_only_test_classes.py",
            file=sys.stderr,
        )
        return 1

    minimal = sorted(
        name
        for name in cpu_only
        if not any(other != name and other in name for other in cpu_only)
    )
    print(" and ".join(f"not {name}" for name in minimal))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
