#!/usr/bin/env bash
# =============================================================================
# PPU pod 内：把测试跑批收敛到「只跑 CUDA」的公共过滤器。必须被 source（不是执行）。
# smoke / accuracy / perf / full_ut 四条门禁共用这一份，改行为只改这里。
#
# 为什么需要它：run_test.py 的 --include / --exclude 只能选**文件**，而 test/ 下大量
# 测试文件内部同时定义了 CPU 与 GPU 两套用例 —— inductor 尤其明显：一个 CommonTemplate
# 会被 copy_tests() 拷成 CpuTests + GPUTests 两份，于是"挑一个 CUDA 文件"实际上会连带
# 跑掉一半 CPU 用例。PPU 门禁只关心 CUDA 侧行为，那一半既拖时长、又与被测硬件无关。
#
# -----------------------------------------------------------------------------
# 两层过滤，缺一不可
# -----------------------------------------------------------------------------
# 第 1 层 PYTORCH_TESTING_DEVICE_ONLY_FOR=cuda（官方机制，对齐 .ci/pytorch/test.sh 对
#   cuda BUILD_ENVIRONMENT 的处理）。它经 common_device_type.py 的
#   get_desired_device_type_test_bases() 把测试基类过滤成只剩 CUDA，于是：
#     - instantiate_device_type_tests 只实例化 CUDA 变体（TestFoo -> 只有 TestFooCUDA）；
#     - torch/testing/_internal/inductor_utils.py 的 RUN_CPU 变成 False，`if RUN_CPU:`
#       守着的整份 CPU 套件（test_torchinductor.py 的 CpuTests / SweepInputsCpuTest 等）
#       直接不再定义 —— 是"不生成"而不是"跑了再 skip"，省的是真实时间。
# 第 2 层 -k 按**类名**排除（本文件的 PPU_K_CPU_ONLY）。第 1 层管不到两种写法：
#     - `if HAS_CPU:` 守着的 CPU 套件：HAS_CPU 只探测有没有 C++ 编译器，与该环境变量无关
#       （如 test_torchinductor_dynamic_shapes.py 的 DynamicShapesCpuTests）；
#     - 直接 `class XxxCpuTests(Template)` / copy_tests(..., "cpu") 手工拷出来的套件，
#       它们根本不走 device type 那套实例化机制。
#
# -----------------------------------------------------------------------------
# 第 2 层的类名清单为什么"算"出来而不是"写"死
# -----------------------------------------------------------------------------
# 本仓库要长期 rebase 上游，写死的清单会悄悄过期：-k 里写一个已被改名的类名，pytest
# 不报错、只是静默失效，于是 CPU 用例又跑回来、门禁时长翻倍还没人发现。所以每次跑批前
# 由 list_cpu_only_test_classes.py 扫一遍 test/ 现场算出来（三个来源取并集、带过期校验，
# 细节见那个文件的 docstring），结果打印到日志，可审计。
#
# 不做的事（边界）：散落在 GPU 测试类里的**个别** CPU 用例（如 test_perf.py 的
#   test_fusion_choice4_cpu）不在这里处理 —— 按类名收不住，只能各门禁自己按用例名加。
#   本文件的 ppu_cuda_only_k_expr 支持把这类片段当参数传进来一起 and 上。
#
# -----------------------------------------------------------------------------
# 用法
# -----------------------------------------------------------------------------
#   source .ci/ppu/cuda_only_filter.sh          # source 时即完成扫描与校验
#   python test/run_test.py --include ... -k "$(ppu_cuda_only_k_expr)" --verbose
#   # 叠加自己的排除项（任意多个片段，按 and 连接；空串会被忽略）：
#   python test/run_test.py ... -k "$(ppu_cuda_only_k_expr "not fp8 and not float8")"
#
# 注意：某个文件如果**只有** CPU 套件（本就该在文件级 --exclude 掉），被本过滤器全部
# 剔掉后 pytest 会以 exit code 5（no tests collected）结束，而 run_test.py 把 5 归一化
# 成 0 —— 表现为静默空跑，不会把 job 判红。
# =============================================================================

# 第 1 层：只实例化 CUDA 变体
export PYTORCH_TESTING_DEVICE_ONLY_FOR="cuda"

# 第 2 层：扫 test/ 现场算出类名排除表达式，缓存到 PPU_K_CPU_ONLY。
# 路径按脚本自身位置反推，不依赖调用方的 cwd。扫描器单独成 .py 而不是内联 heredoc：
# 内联在 $( ) 里的 heredoc 在 bash 3.2 上会被解析器当普通内容处理（正则里的 ['"] 会被
# 当成没闭合的引号，直接 syntax error），且独立文件也能被 lintrunner 检查、单独跑自查。
_ppu_cuda_only_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! PPU_K_CPU_ONLY="$(python "${_ppu_cuda_only_dir}/list_cpu_only_test_classes.py" \
    "${_ppu_cuda_only_dir}/../../test")"; then
    echo "[cuda-only][error] 计算 CPU 测试类排除表达式失败（原因见上）" >&2
    exit 1
fi
unset _ppu_cuda_only_dir

echo "[cuda-only] 第 1 层: PYTORCH_TESTING_DEVICE_ONLY_FOR=${PYTORCH_TESTING_DEVICE_ONLY_FOR}"
echo "[cuda-only] 第 2 层: -k '${PPU_K_CPU_ONLY}'"

# 把 CPU 类排除表达式与调用方自带的片段用 and 连成一条 -k 表达式打到 stdout。
# 用法：-k "$(ppu_cuda_only_k_expr [片段...])"；空串片段忽略，方便调用方写条件拼接。
ppu_cuda_only_k_expr() {
    local out="${PPU_K_CPU_ONLY}"
    local part
    for part in "$@"; do
        if [[ -n "${part}" ]]; then
            out="${out} and ${part}"
        fi
    done
    printf '%s' "${out}"
}
