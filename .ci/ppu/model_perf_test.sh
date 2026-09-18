#!/usr/bin/env bash
# =============================================================================
# PPU pod 内：dynamo benchmark 模型性能（model perf）入口。
# 由 .github/workflows/ppu_model_perf_810/890.yml 经 flytiger-eco/ppu-distributed-action
# 在单卡 PPU pod 内执行（源码已由 action 解压到 pod 的 source_dir）。
#
# 单独抽成脚本而不是内联到 yaml 的 command：command 由 pod 的默认 shell 执行，
# 未必是 bash，而 sdk_env.sh 依赖 bash 语法（[[ ]] / BASH_SOURCE）且必须被 source。
#
# torch 来自 ppu-ci-810/890 的 build job 编译产出的 whl（workflow 侧已下载到 WHEEL_DIR，随源码一起
# 送进 pod），由 install_wheel.sh 安装：跑的是 PPU 基础镜像，镜像里没有 torch，
# 门禁必须测本 PR 编出来的那一份。
#
# -----------------------------------------------------------------------------
# 跑什么（对齐 inductor-perf-test-nightly.yml 的 build job test-matrix）
# -----------------------------------------------------------------------------
# 上游 build job `cuda13.0-py3.10-gcc11-sm80` 的 test-matrix 里有若干类 config，全部落到
# .ci/pytorch/test.sh 的 benchmark 分支；本门禁只跑其中 3 类（torchbench 那一类已按需去掉）：
#   - huggingface  -> test_dynamo_benchmark huggingface  -> benchmarks/dynamo/huggingface.py
#   - timm_models  -> test_dynamo_benchmark timm_models  -> benchmarks/dynamo/timm_models.py
#   - cachebench   -> test_cachebench                    -> benchmarks/dynamo/cachebench.py
# 前两者在上游最终都走 test_single_dynamo_benchmark，执行 `python benchmarks/dynamo/$suite.py`；
# 注：cachebench 内部仍按 --benchmark 跑 torchbench / huggingface 两套模型集，所以 TORCHBENCHPATH
# 依然需要 —— 去掉的只是「把 torchbench 单独当一个 perf config 跑」那一类。
# 注：benchmarks/dynamo/*.py 只是入口，真正的模型列表来自它们各自加载的模型仓库
# （HuggingFace transformers / timm / torchbench），并非本仓库的测试文件。
#
# 与 full_ut 的分片差异：本 workflow 是「每个 config 一个分片」（NUM_TEST_SHARDS 默认 1），
# 即一个 config 的整套模型跑在**同一个**单卡 pod 里，不再按 partition 拆。若哪天要拆，
# 给 workflow 传 NUM_TEST_SHARDS>1 + SHARD_NUMBER，本脚本会自动补 --total-partitions/--partition-id
# （对齐 common.py 的 get_benchmark_indices 语义）。
#
# -----------------------------------------------------------------------------
# 模型从哪来（NAS 预置 + 内网可达）
# -----------------------------------------------------------------------------
# 这几个 config 都要加载「模型」：
#   - huggingface.py   由默认 config 构造随机权重（config_cls() / model_cls(config)），
#                      但 transformers 仍可能去 hub 拉 config/tokenizer；
#   - timm_models.py   create_model(pretrained=True) / list_models(pretrained=True) 要拉预训练权重；
#   - cachebench.py    内部按 --benchmark 跑 torchbench / huggingface 两套模型集；torchbench 那套
#                      还要 pytorch/benchmark 仓库本体（PYTHONPATH/TORCHBENCHPATH）与其模型依赖。
# 本集群里模型缓存与 torchbench 仓库已由运维预置到已挂载的 NAS（HOST_VOLUMES 里的 /nas_aisw、
# /wl_nas），pod 内也能走内网出口。因此这里把 HF_HOME / TORCHBENCHPATH 指向 NAS 上的预置目录
# （由 workflow 的 env 注入，见下），既命中缓存、又能在缺项时经内网回源，不会像公网那样卡死。
#
# 依赖环境变量：
#   BENCH_CONFIG     - huggingface | timm_models | cachebench（必填）
#   SHARD_NUMBER     - 当前分片编号，从 1 开始（默认 1）
#   NUM_TEST_SHARDS  - 总分片数（默认 1；=1 时不下发 partition 参数）
#   WHEEL_DIR        - torch whl 所在目录（默认 <repo>/.ci/ppu/wheelhouse，供 install_wheel.sh 使用）
#   SDK_INSTALL_DIR  - PPU SDK 安装目录（默认 /usr/local，供 sdk_env.sh 使用）
#   PIP_INDEX        - 内部 pip 源（可选；不设则用 pip_sources.sh 的内置候选源）
#   TRITON_INDEX     - 装 triton 的**唯一**源（默认取 PIP_INDEX，供 install_triton.sh 使用）
#   TRITON_VERSION   - 钉住的 triton 版本（默认 3.6.0，供 install_triton.sh 使用）
#   HF_HOME          - NAS 上预置的 HuggingFace 缓存目录（可选；不设则用 transformers 默认）
#   TORCHBENCHPATH   - NAS 上预置的 pytorch/benchmark 仓库目录（cachebench 的 torchbench 模型集需要）
#   BENCH_PIP_PACKAGES - 要装成指定版本的 benchmark 依赖（空格分隔的 pip spec）；默认见 ensure_bench_deps
#   MODEL_PERF_EXTRA_ARGS - 透传给 benchmark 脚本的额外参数（可选）
#   PR_NUMBER        - 仅用于日志溯源（可选）
# =============================================================================
set -euo pipefail

export SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"

BENCH_CONFIG="${BENCH_CONFIG:-}"
case "$BENCH_CONFIG" in
    huggingface | timm_models | cachebench) ;;
    *)
        echo "[model-perf][error] BENCH_CONFIG 只支持 huggingface / timm_models / cachebench，实际: '${BENCH_CONFIG}'" >&2
        exit 1
        ;;
esac

SHARD_NUMBER="${SHARD_NUMBER:-1}"
NUM_TEST_SHARDS="${NUM_TEST_SHARDS:-1}"
if (( SHARD_NUMBER < 1 || SHARD_NUMBER > NUM_TEST_SHARDS )); then
    echo "[model-perf][error] SHARD_NUMBER=${SHARD_NUMBER} 不在 1..${NUM_TEST_SHARDS} 内" >&2
    exit 1
fi

# 切到源码根目录：action 的 source_dir 可配置，这里按脚本自身位置反推，不写死路径
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
echo "[model-perf] 源码目录: $(pwd)"
echo "[model-perf] 配置: BENCH_CONFIG=${BENCH_CONFIG} 分片=${SHARD_NUMBER}/${NUM_TEST_SHARDS}"

# sdk_env.sh 会 source PPU SDK 的 envsetup.sh（设置 LD_LIBRARY_PATH / PATH）并校验 nvcc。
# 必须 source（而非执行），环境变量才能作用于后续的 pip / benchmark 进程；
# 也必须放在装 torch 之前 —— import torch 要能找到 SDK 里的运行时库。
source .ci/ppu/sdk_env.sh

# 安装被测的 torch：本 PR 由 ppu_ci_810/890.yml 的 build job 编出来的 whl
bash .ci/ppu/install_wheel.sh

# inductor 用例的 codegen 后端：钉版本、且只从内部源装（理由见 install_triton.sh 头注释）。
# 必须在 install_wheel.sh 之后；四条测试门禁用同一份脚本，改行为只改那一处。
# 对本门禁尤其关键：这里全是 benchmark 计时类跑批，triton 版本错位会直接改变生成的 kernel，
# 测出来的数与基线不可比。
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

# 测试框架依赖（pytest 及其插件、expecttest、hypothesis）镜像里没预装对版本，必须在这里补。
# 与 full_ut / perf 用同一份脚本；benchmarks/dynamo 的部分工具也会 import 到这些测试期包。
bash .ci/ppu/install_test_deps.sh

# -----------------------------------------------------------------------------
# benchmark 自身的 python 依赖：镜像不保证预装，且版本必须钉死 —— 这些包的版本直接影响
# benchmark 结果的可比性（transformers 决定 HF 模型清单，numpy/scipy/pandas 决定数值与统计
# 口径）。所以按**指定版本**无条件安装（不再用「import 成功就跳过」的老逻辑，那样无法把镜像
# 自带的错版本纠正过来），并用 constraints 把 torch 钉在当前 whl 版本 —— 否则 pip 解析
# transformers/timm 的依赖时可能顺手重装一个公网 torch，把上面刚装的被测 whl 覆盖掉
# （那等于门禁测了个别的包，比不测更糟）。装完再校验一次 torch 版本没变。
# -----------------------------------------------------------------------------
ensure_bench_deps() {
    # 默认清单：按 PPU 上实测能跑通 huggingface / timm_models / cachebench 的版本钉死；可用
    # BENCH_PIP_PACKAGES 覆盖（空格分隔的 pip spec）。timm 未钉版本 —— timm_models config 需要
    # 它但暂无指定版本，跟随源上的稳定版。
    local want="${BENCH_PIP_PACKAGES:-numpy==1.26.2 scipy==1.14.1 pandas==2.2.3 tqdm>=4.66.0 transformers==5.17.0 timm}"
    [[ -n "$want" ]] || return 0
    # spec 里带 ==/>= 等，不能直接 import；拆成数组原样交给 pip
    local -a specs
    read -r -a specs <<<"$want"
    echo "[model-perf] 安装 benchmark 依赖（钉版本）: ${specs[*]}"
    local torch_before
    torch_before="$(cd /tmp && python -c 'import torch;print(torch.__version__)')"
    local constraints="/tmp/model_perf_constraints.txt"
    echo "torch==${torch_before}" >"$constraints"
    # shellcheck source=.ci/ppu/pip_sources.sh
    source "$REPO_ROOT/.ci/ppu/pip_sources.sh"
    ppu_build_pip_candidates
    local index label installed=0
    local pip_args
    for index in "${PIP_CANDIDATES[@]}" __pip_default__; do
        pip_args=(--disable-pip-version-check --retries 1 --timeout 30 -c "$constraints")
        if [[ "${index}" == "__pip_default__" ]]; then
            label="pip 默认源"
        else
            label="${index}"
            pip_args+=(-i "${index}")
        fi
        echo "[model-perf] 尝试源: ${label}"
        if python -m pip install "${pip_args[@]}" "${specs[@]}"; then
            installed=1
            break
        fi
        echo "[model-perf][warn] 源 ${label} 失败，换下一个"
    done
    if [[ "$installed" != "1" ]]; then
        echo "[model-perf][error] 所有候选源都装不上: ${specs[*]}" >&2
        exit 1
    fi
    # 装完必须确认 torch 没被换掉（constraints 生效的证据）
    local torch_after
    torch_after="$(cd /tmp && python -c 'import torch;print(torch.__version__)')"
    if [[ "$torch_before" != "$torch_after" ]]; then
        echo "[model-perf][error] 装 benchmark 依赖后 torch 从 ${torch_before} 变成 ${torch_after}，被测 whl 已被覆盖" >&2
        exit 1
    fi
    # 打印实际装上的版本，便于事后核对是否与 pin 一致
    echo "[model-perf] 实际安装版本:"
    python -m pip freeze 2>/dev/null | grep -iE '^(numpy|scipy|pandas|tqdm|transformers|timm)=' || true
}
ensure_bench_deps

# 模型缓存 / torchbench 仓库路径：由 workflow 的 env 注入到 NAS 上的预置目录。
# 只在非空时导出，避免把空值写进环境覆盖掉镜像/默认行为。
if [[ -n "${HF_HOME:-}" ]]; then
    export HF_HOME
    echo "[model-perf] HF_HOME=${HF_HOME}"
fi
if [[ -n "${TORCHBENCHPATH:-}" ]]; then
    export TORCHBENCHPATH
    # 上游 test.sh 里 torchbench / cachebench 靠 PYTHONPATH=/torchbench 找到仓库本体
    export PYTHONPATH="${TORCHBENCHPATH}${PYTHONPATH:+:${PYTHONPATH}}"
    echo "[model-perf] TORCHBENCHPATH=${TORCHBENCHPATH} PYTHONPATH=${PYTHONPATH}"
fi

TEST_REPORTS_DIR="$REPO_ROOT/test/test-reports"
mkdir -p "$TEST_REPORTS_DIR"

# partition 参数：仅当 NUM_TEST_SHARDS>1 时下发（对齐 test_single_dynamo_benchmark）。
# 上游 shard_id 从 0 开始（id=SHARD_NUMBER-1），这里保持一致。
PARTITION_FLAGS=()
if (( NUM_TEST_SHARDS > 1 )); then
    PARTITION_FLAGS=(--total-partitions "$NUM_TEST_SHARDS" --partition-id "$((SHARD_NUMBER - 1))")
fi

# 透传的额外参数（workflow 可选注入，例如临时加 --exclude-exact 某模型）
EXTRA_FLAGS=()
if [[ -n "${MODEL_PERF_EXTRA_ARGS:-}" ]]; then
    read -r -a EXTRA_FLAGS <<<"${MODEL_PERF_EXTRA_ARGS}"
fi

# 把结果 csv/json 打到日志里：自建集群没有官方那套 S3 / benchmark database 上传通道，
# 文件本身在 test/test-reports 下、pod 销毁后就查不到了（action 不会带回来）。
dump_result() {
    local f="$1"
    echo "=== model-perf 结果: $(basename "$f") ==="
    if [[ -s "$f" ]]; then
        cat "$f"
    else
        echo "[model-perf][warn] 没有在 ${f} 拿到结果（命令以 0 退出但没写出文件）。"
    fi
}

# -----------------------------------------------------------------------------
# 跑批：按 BENCH_CONFIG 分派
# -----------------------------------------------------------------------------
# 两个 perf suite 的性能语义统一对齐用户本地跑法：--inference --inductor --performance --device cuda。
#   - huggingface：额外带上用户本地那份 --exclude "/" 与 5 个 --exclude-exact（这几个模型在
#     PPU 上跑不过/超大，先剔掉）。--exclude "/" 对 HF 模型名（不含 '/'）是 no-op，保留只为
#     与本地命令逐字一致；PPU 适配后可按需增删。
# 输出统一落 TEST_REPORTS_DIR 下的 csv，再 dump 到日志。
case "$BENCH_CONFIG" in
    huggingface)
        out="$TEST_REPORTS_DIR/inductor_huggingface_perf.csv"
        echo "=== CUDA dynamo benchmark: huggingface (inference/inductor/performance) ==="
        python benchmarks/dynamo/huggingface.py \
            --inference --inductor --performance --device cuda \
            --exclude "/" \
            --exclude-exact AllenaiLongformerBase \
            --exclude-exact T5Small \
            --exclude-exact DistillGPT2 \
            --exclude-exact GoogleFnet \
            --exclude-exact YituTechConvBert \
            "${PARTITION_FLAGS[@]}" "${EXTRA_FLAGS[@]}" \
            --output "$out"
        dump_result "$out"
        ;;
    timm_models)
        out="$TEST_REPORTS_DIR/inductor_timm_perf.csv"
        echo "=== CUDA dynamo benchmark: timm_models (inference/inductor/performance) ==="
        python benchmarks/dynamo/timm_models.py \
            --inference --inductor --performance --device cuda \
            "${PARTITION_FLAGS[@]}" "${EXTRA_FLAGS[@]}" \
            --output "$out"
        dump_result "$out"
        ;;
    cachebench)
        # 对齐 .ci/pytorch/test.sh 的 test_cachebench：modes=(training inference)，每个 mode 各跑
        # 静态与 --dynamic 两遍。上游用 SHARD_NUMBER 在 torchbench / huggingface 两个 benchmark 间
        # 二选一（shard1=torchbench, shard2=huggingface）；本 workflow「一个 config 一个分片」，
        # 所以这里把两个 benchmark 都跑一遍（等价于上游两分片的并集）。
        echo "=== CUDA dynamo cachebench (torchbench + huggingface) ==="
        for benchmark in torchbench huggingface; do
            for mode in training inference; do
                out="$TEST_REPORTS_DIR/cachebench_${benchmark}_${mode}.json"
                python benchmarks/dynamo/cachebench.py \
                    --mode "$mode" --device cuda --benchmark "$benchmark" --repeat 3 \
                    "${EXTRA_FLAGS[@]}" --output "$out"
                dump_result "$out"
                out_dyn="$TEST_REPORTS_DIR/cachebench_${benchmark}_${mode}_dynamic.json"
                python benchmarks/dynamo/cachebench.py \
                    --mode "$mode" --dynamic --device cuda --benchmark "$benchmark" --repeat 3 \
                    "${EXTRA_FLAGS[@]}" --output "$out_dyn"
                dump_result "$out_dyn"
            done
        done
        ;;
esac

echo "[model-perf] 完成 (config=${BENCH_CONFIG} shard=${SHARD_NUMBER}/${NUM_TEST_SHARDS})"
