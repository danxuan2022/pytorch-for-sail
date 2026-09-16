#!/usr/bin/env bash
# =============================================================================
# PPU pod 内：CUDA inductor 精度测试入口。
# 由 .github/workflows/ppu_ci_810/890.yml 的 accuracy job 经 flytiger-eco/ppu-distributed-action
# 在单卡 PPU pod 内执行（源码已由 action 解压到 pod 的 source_dir）。
#
# 单独抽成脚本而不是内联到 yaml 的 command：command 由 pod 的默认 shell 执行，
# 未必是 bash，而 sdk_env.sh 依赖 bash 语法（[[ ]] / BASH_SOURCE）且必须被 source。
#
# torch 来自 ppu-ci-810/890 的 build job 编译产出的 whl（workflow 侧已下载到 WHEEL_DIR，随源码一起
# 送进 pod），由 install_wheel.sh 安装：跑的是 PPU 基础镜像，镜像里没有 torch，
# 门禁必须测本 PR 编出来的那一份。
#
# 依赖环境变量：
#   WHEEL_DIR        - torch whl 所在目录（默认 <repo>/.ci/ppu/wheelhouse，供 install_wheel.sh 使用）
#   SDK_INSTALL_DIR  - PPU SDK 安装目录（默认 /usr/local，供 sdk_env.sh 使用）
#   PIP_INDEX        - 内部 pip 源（可选；不设则用镜像自带的 pip 默认源）
#   TRITON_INDEX     - 装 triton 的**唯一**源（默认取 PIP_INDEX，供 install_triton.sh 使用）
#   TRITON_VERSION   - 钉住的 triton 版本（默认 3.6.0，供 install_triton.sh 使用）
#   PR_NUMBER        - 仅用于日志溯源（可选）
# =============================================================================
set -euo pipefail

export SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"

# 切到源码根目录：action 的 source_dir 可配置，这里按脚本自身位置反推，不写死路径
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "[accuracy] 源码目录: $(pwd)"

# sdk_env.sh 会 source PPU SDK 的 envsetup.sh（设置 LD_LIBRARY_PATH / PATH）并校验 nvcc。
# 必须 source（而非执行），环境变量才能作用于后续的 pip / 测试进程；
# 也必须放在装 torch 之前 —— import torch 要能找到 SDK 里的运行时库。
source .ci/ppu/sdk_env.sh

# 安装被测的 torch：本 PR 由 ppu_ci_810/890.yml 的 build job 编出来的 whl
bash .ci/ppu/install_wheel.sh

# inductor 用例的 codegen 后端：钉版本、且只从内部源装（理由见 install_triton.sh 头注释）。
# 必须在 install_wheel.sh 之后；四条测试门禁用同一份脚本，改行为只改那一处。
bash .ci/ppu/install_triton.sh

echo "=== 环境自检 (CUDA) ==="
echo "pr=${PR_NUMBER:-none} hostname=$(hostname) node=${NODE_NAME:-unknown}"
echo "rank=${RANK:-0} nproc_per_node=${NPROC_PER_NODE:-1}"
python --version
# 自检必须在源码树外执行：`python -c` 会把 cwd（此时是源码根）放进 sys.path[0]，
# 未编译的源码目录 torch/ 会遮蔽刚装上的 torch，而 torch/version.py 是构建产物、
# 源码树里不存在，于是报 ModuleNotFoundError: No module named 'torch.version'。
# 打印 torch.__file__ 以便一眼确认导入的是 site-packages 里的那份。
(cd /tmp && python -c "import torch; print('torch', torch.__version__, torch.__file__); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())")
ppu-smi || echo "[warn] ppu-smi 不可用，请确认 pod 已分配 PPU 设备"

# 测试框架依赖（pytest 及其插件、expecttest、hypothesis）镜像里没预装对版本，必须在这里补
# （注意顺序：先装 torch 再装这些，torch 的 whl 不会反过来动这些测试期包的版本）：
# run_test.py 的 check_pip_packages() 硬校验 pytest-rerunfailures / pytest-flakefinder /
# pytest-xdist，缺任意一个直接 exit 1。
# 版本一律对齐 .ci/docker/requirements-ci.txt（官方 CI 跑 test case 的同一份约束），
# 安装逻辑（含 pip 源诊断与多源回退）抽到共享脚本，smoke_test.sh 用同一份。
bash .ci/ppu/install_test_deps.sh

# boto3 故意不装：它只被 tools/stats/upload_metrics.py 用于往官方 S3 上报指标，缺失时
# EMIT_METRICS=False 静默降级（日志里那条 "Unable to import boto3" 只是提示，不影响退出码），
# 自建集群也没有对应凭证。

# -----------------------------------------------------------------------------
# 只跑 CUDA：--include 只能挑**文件**，而这几个文件内部同时定义了 CPU 与 GPU 两套用例
# （test_torchinductor.py 的 CpuTests / SweepInputsCpuTest、
#   test_torchinductor_dynamic_shapes.py 的 DynamicShapesCpuTests 都是 CommonTemplate 的
#   整份 CPU 拷贝，各占所在文件一半用例）。用公共过滤器把它们收掉，
# 两层机制（PYTORCH_TESTING_DEVICE_ONLY_FOR + 类名级 -k）的缘由见 cuda_only_filter.sh 头注释。
# 必须 source：它要 export 环境变量并定义 ppu_cuda_only_k_expr 给下面用。
# shellcheck source=.ci/ppu/cuda_only_filter.sh
source .ci/ppu/cuda_only_filter.sh

echo "=== CUDA inductor 精度单测（run_test.py --include 白名单过滤，仅 CUDA 相关） ==="
# 下面的 include 白名单是"可调项"：先给一组有代表性的 CUDA 精度用例，
# 请按 PPU 实际支持情况增删（例如想加深 op 级覆盖可加 inductor/test_torchinductor_opinfo，
# 但该文件很重、耗时长，默认不放进门禁）。
# 用例名必须是 run_test.py 发现得到的测试名（即 tools/testing/discover_tests.py 的 TESTS，
# 大体对应 test/ 下去掉 .py 后缀的相对路径）：-i/--include 的 choices 就是这份清单，
# 写错一个名字 argparse 会直接 exit 2（invalid choice），一个用例也不会跑。
# 因此 rebase 上游后若有测试文件被重命名/删除，这里要跟着改。可在仓库根目录用
#   python -c "import sys;sys.path.insert(0,'.');from tools.testing.discover_tests import TESTS;print('inductor/test_gpu_select_algorithm' in TESTS)"
# 校验（不需要装 torch）。
# test_gpu_select_algorithm：上游 #163615 把 test_cuda_select_algorithm.py 泛化成兼容 XPU 后
# 改名而来，内容仍是 GPU 版 select_algorithm（instantiate_device_type_tests 的
# only_for=("cuda","xpu")），在 PPU 上跑的仍是 cuda 那一半。
# FP8：真武 PPU 当前不支持 FP8，两层都要。
#   - 文件级：inductor/test_fp8（全量 FP8 语义）不在上面白名单里，与 smoke job 一致。
#   - 用例级：下面 -k 里的 FP8 排除表达式，兜住白名单文件内散落的 FP8 用例 ——
#     test_torchinductor.py 的 test_generate_rand_fp8（在 CommonTemplate 里，会被 copy_tests
#     拷成 GPUTests.test_generate_rand_fp8_cuda 真跑）与 test_cuda_repro.py 的
#     test_float8_e8m0fnu。pytest 的 -k 大小写敏感，故 fp8/float8/e4m3/e5m2 各写一份。
# PPU 支持 FP8 后：把白名单里加回 inductor/test_fp8、删掉下面 K_FP8 即可。
K_FP8="not fp8 and not FP8 and not Fp8 \
and not float8 and not Float8 \
and not e4m3 and not E4M3 \
and not e5m2 and not E5M2"
# 按用例名排除的黑名单：--include 只能到文件级，下面这几个用例散在白名单文件内，
# 只能在 -k 里按名字收掉（与 K_FP8 同理，各门禁按需增删）：
#   - test_cuda_repro.py 的 test_not_disabling_ftz_yields_zero / test_triton_interpret；
#   - test_cudagraph_trees.py 的 test_graph_partition_user_defined_triton_kernel_reuse。
# pytest 的 -k 是子串匹配，这三个名字在对应文件内唯一，不会误伤其它用例。
K_SKIP_CASES="not test_not_disabling_ftz_yields_zero \
and not test_triton_interpret \
and not test_graph_partition_user_defined_triton_kernel_reuse"
# 不使用 --upload-artifacts-while-running：那是官方 S3 上传路径，自建集群上没有。
# -k 是 run_test.py 的 --pytest-k-expr，会原样透传给 pytest：这里把公共过滤器算出的
# 纯 CPU 类排除、K_FP8 与 K_SKIP_CASES 用 and 拼成一条。
python test/run_test.py \
    --include \
        inductor/test_cuda_repro \
        inductor/test_cudagraph_trees \
        inductor/test_gpu_select_algorithm \
    -k "$(ppu_cuda_only_k_expr "$K_FP8")" \
    --verbose

echo "[accuracy] 完成"
