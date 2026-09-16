#!/usr/bin/env python3
"""红线安全门禁：扫描 PR 是否把「内部身份 / 内网信息」泄漏进了这个公开 fork。

由 .github/workflows/ppu-security-gate.yml 调用，也可单独执行自查：

    # 扫描一段提交区间（提交元信息 + 该区间的 diff 新增行）
    python .ci/ppu/security_gate_scan.py --base-sha <base> --head-sha <head>

    # 只跑内置断言，验证规则本身没被改坏（不依赖 git 历史）
    python .ci/ppu/security_gate_scan.py --self-test

背景：本仓库是 PyTorch 的对外 fork。开发者日常在阿里内网提交，很容易把两类东西
带出来，一旦推到公开仓库就撤不回：

  1）提交人身份。git 的 author/committer 若写成阿里邮箱（@alibaba-inc.com）或阿里
     花名工号式的「名.姓」用户名（如 xx.xx），外部只要拼上阿里邮箱后缀就
     还原出真实邮箱——等于直接泄漏员工身份。故对「提交区间里每一个 commit 的
     author/committer 的 name 与 email」做检查。

  2）内网红线内容。diff 新增行里若出现内网 IP（10./172.16-31./192.168./100.64./
     169.254.）、代理配置（http_proxy 等）或阿里内部域名，都是典型的内网信息外泄。
     只扫「新增行」：删除行不算泄漏，存量内容不在本次 PR 的责任范围内。

设计取舍：
  * 门禁默认命中即硬失败（exit 1），逼开发者在合入前改掉。确有误报或无法避免时，
    走「行内豁免」或「PR 打 suppress-security-gate 标签」两条明路，而不是放宽规则。
  * 打印命中时对敏感值做掩码（10.**.**.** / da****.li****），既能让作者定位到
    file:line 自己去改，又不至于在 CI 日志里把原文再抄一遍、二次泄漏。
  * 纯正则 + git 文本解析，不 import torch、不需要构建环境，任何 runner 都能跑。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from typing import Iterable, Iterator, List, NamedTuple, Optional, Tuple


# ---------------------------------------------------------------------------
# 规则常量
# ---------------------------------------------------------------------------

# 阿里系邮箱域名：author/committer 的 email 命中即判定「真实身份直接泄漏」。
# 以 endswith 匹配，故 list.alibaba-inc.com 这类子域也能被 alibaba-inc.com 覆盖。
ALIBABA_EMAIL_DOMAINS = (
    "alibaba-inc.com",
    "alibaba.com",
    "taobao.com",
    "tmall.com",
    "alipay.com",
    "antgroup.com",
    "antfin.com",
    "aliyun.com",
    "alibabacloud.com",
    "cainiao.com",
    "lazada.com",
    "aliexpress.com",
)

# 阿里内部域名（会出现在配置 / 脚本 / 注释里）。命中即判定内网信息泄漏。
# 注意：这里只列「拼上去就是内部服务」的域名，不含已在本仓库构建里公开使用的
# 私有镜像仓库域名（那属于既定构建约定，扫描只看新增行，不在此追责）。
INTERNAL_DOMAIN_SUFFIXES = (
    ".alibaba-inc.com",
    ".aliyun-inc.com",
    ".alipay.net",
    ".antgroup.net",
    ".taobao.net",
)

# 阿里花名用户名形态：「小写.小写」，如 xx.xx。拼上 @alibaba-inc.com 即真实邮箱。
# 用于校验 git 的 name 字段与 email 的 local-part（@ 前那段）。
ALI_USERNAME_RE = re.compile(r"^[a-z][a-z0-9]*\.[a-z][a-z0-9]+$")

EMAIL_RE = re.compile(r"^\s*([^@\s]+)@([^@\s]+?)\s*$")

# 代理配置：环境变量式（http_proxy / HTTPS_PROXY / all_proxy / no_proxy / ftp_proxy）。
PROXY_RE = re.compile(r"(?i)\b(?:https?|all|no|ftp)_proxy\b")

# IPv4：先粗匹配四段点分，再逐段校验 0-255，避免把版本号等误当 IP。
IPV4_RE = re.compile(r"\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b")

# 非回环的 IPv6（含至少一个 ::，且不是 ::1）。粗匹配，命中后再排除白名单。
IPV6_RE = re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){2,}[0-9a-fA-F]{0,4}(?::[0-9a-fA-F]{1,4})*\b")

# IP 白名单：回环 / 未指定 / 广播 / 公共 DNS 等公开常量，出现在代码里不算内网泄漏。
IP_ALLOWLIST = frozenset(
    {
        "0.0.0.0",
        "127.0.0.1",
        "255.255.255.255",
        "1.1.1.1",  # Cloudflare 公共 DNS
        "8.8.8.8",  # Google 公共 DNS
        "8.8.4.4",
    }
)

# 行内豁免标记：确属误报（如文档举例、测试夹具）时，在该行加下述任一标记即跳过内容扫描。
SUPPRESS_MARKERS = ("security-gate: allow", "noqa: security-gate")


class Finding(NamedTuple):
    """一条命中记录。location 形如 'commit <sha>' 或 'path/to/file:123'。"""

    severity: str  # HIGH / MEDIUM
    kind: str  # 违规类别（人读）
    location: str
    detail: str  # 已掩码的说明


# ---------------------------------------------------------------------------
# 掩码
# ---------------------------------------------------------------------------


def _mask_email(email: str) -> str:
    """da****.li****@ali****  —— 保留头部字符便于作者辨认，其余打码。"""
    m = EMAIL_RE.match(email)
    if not m:
        return _mask_token(email)
    local, domain = m.group(1), m.group(2)
    return f"{_mask_token(local)}@{_mask_token(domain)}"


def _mask_token(tok: str) -> str:
    def mask_part(p: str) -> str:
        if len(p) <= 2:
            return p[:1] + "*" * (len(p) - 1) if p else p
        return p[:2] + "*" * (len(p) - 2)

    return ".".join(mask_part(p) for p in tok.split("."))


def _mask_ipv4(ip: str) -> str:
    head = ip.split(".", 1)[0]
    return f"{head}.**.**.**"


# ---------------------------------------------------------------------------
# 身份检查（提交元信息）
# ---------------------------------------------------------------------------


def check_identity(sha: str, name: str, email: str, role: str) -> List[Finding]:
    """检查单个 commit 的 author/committer 身份。role 取 'author' / 'committer'。"""
    out: List[Finding] = []
    loc = f"commit {sha[:12]} ({role})"

    email = (email or "").strip()
    name = (name or "").strip()

    m = EMAIL_RE.match(email)
    local = m.group(1).lower() if m else ""
    domain = m.group(2).lower() if m else ""

    if domain and any(
        domain == d or domain.endswith("." + d) for d in ALIBABA_EMAIL_DOMAINS
    ):
        out.append(
            Finding(
                "HIGH",
                "阿里邮箱作为提交人身份",
                loc,
                f"email={_mask_email(email)} 命中阿里邮箱域名，会直接泄漏真实身份",
            )
        )

    # local-part 或 name 是「名.姓」花名形态：拼上阿里邮箱后缀即真实邮箱。
    for field, value in (("email local-part", local), ("name", name.lower())):
        if value and ALI_USERNAME_RE.match(value):
            out.append(
                Finding(
                    "HIGH",
                    "阿里花名式用户名作为提交人身份",
                    loc,
                    f"{field}={_mask_token(value)} 形如「名.姓」，"
                    f"拼上阿里邮箱后缀即真实邮箱，请改用对外身份提交",
                )
            )

    return out


# ---------------------------------------------------------------------------
# 内容检查（diff 新增行）
# ---------------------------------------------------------------------------


def _valid_ipv4(parts: Tuple[str, str, str, str]) -> bool:
    try:
        return all(0 <= int(p) <= 255 for p in parts) and all(
            p == "0" or not p.startswith("0") for p in parts
        )
    except ValueError:
        return False


def _classify_ipv4(ip: str) -> Optional[Tuple[str, str]]:
    """返回 (severity, 说明) 或 None（白名单/非内网且非可疑则放行）。"""
    if ip in IP_ALLOWLIST:
        return None
    a, b = (int(x) for x in ip.split(".")[:2])
    if a == 10:
        return "HIGH", "10.0.0.0/8 内网地址"
    if a == 172 and 16 <= b <= 31:
        return "HIGH", "172.16.0.0/12 内网地址"
    if a == 192 and b == 168:
        return "HIGH", "192.168.0.0/16 内网地址"
    if a == 100 and 64 <= b <= 127:
        return "HIGH", "100.64.0.0/10 (CGNAT) 内网地址"
    if a == 169 and b == 254:
        return "HIGH", "169.254.0.0/16 链路本地地址"
    if a == 127:  # 非 127.0.0.1 的其它回环地址，提示但不作为内网泄漏
        return None
    # 其余为公网 IP：可能是内网出口/网关的公网地址，按中危提示，允许行内豁免。
    return "MEDIUM", "疑似硬编码公网 IP"


def scan_line(path: str, lineno: int, text: str) -> List[Finding]:
    out: List[Finding] = []
    if any(marker in text for marker in SUPPRESS_MARKERS):
        return out
    loc = f"{path}:{lineno}"

    if PROXY_RE.search(text):
        out.append(Finding("HIGH", "代理配置", loc, "出现 *_proxy 代理配置"))

    low = text.lower()
    for suffix in INTERNAL_DOMAIN_SUFFIXES:
        if suffix in low:
            out.append(
                Finding("HIGH", "阿里内部域名", loc, f"出现内部域名 *{suffix}")
            )
            break

    for m in IPV4_RE.finditer(text):
        parts = m.groups()
        if not _valid_ipv4(parts):
            continue
        ip = ".".join(parts)
        verdict = _classify_ipv4(ip)
        if verdict is None:
            continue
        severity, why = verdict
        out.append(Finding(severity, "IP 地址", loc, f"{why}: {_mask_ipv4(ip)}"))

    for m in IPV6_RE.finditer(text):
        token = m.group(0)
        if token.lower() in ("::1",) or "::" not in token:
            continue
        # 至少要有两段十六进制才当地址，排除 `a::` 这种代码里的作用域符号误报
        if len(re.findall(r"[0-9a-fA-F]{1,4}", token)) < 3:
            continue
        out.append(Finding("MEDIUM", "IPv6 地址", loc, "疑似硬编码 IPv6 地址"))
        break

    return out


# ---------------------------------------------------------------------------
# git 交互
# ---------------------------------------------------------------------------


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True)


def collect_commit_findings(base: str, head: str) -> List[Finding]:
    sep = "\x1f"
    fmt = sep.join(["%H", "%an", "%ae", "%cn", "%ce"])
    out_lines = _git("log", "--no-merges", f"--format={fmt}", f"{base}..{head}")
    findings: List[Finding] = []
    for row in out_lines.splitlines():
        if not row.strip():
            continue
        sha, an, ae, cn, ce = (row.split(sep) + [""] * 5)[:5]
        findings += check_identity(sha, an, ae, "author")
        # committer 与 author 完全一致时不重复报
        if (cn, ce) != (an, ae):
            findings += check_identity(sha, cn, ce, "committer")
    return findings


def iter_added_lines(diff_text: str) -> Iterator[Tuple[str, int, str]]:
    """解析 `git diff -U0` 输出，产出 (新文件路径, 新文件行号, 新增行文本)。"""
    path: Optional[str] = None
    new_lineno = 0
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            p = line[4:].strip()
            path = None if p == "/dev/null" else (p[2:] if p.startswith("b/") else p)
        elif line.startswith("@@"):
            m = re.search(r"\+(\d+)", line)
            new_lineno = int(m.group(1)) if m else 0
        elif line.startswith("+") and not line.startswith("+++"):
            if path is not None:
                yield path, new_lineno, line[1:]
            new_lineno += 1
        # -U0 下没有上下文行；删除行 / 元信息行不推进新文件行号


def collect_content_findings(base: str, head: str) -> List[Finding]:
    diff_text = _git("diff", "-U0", "--no-color", base, head)
    findings: List[Finding] = []
    for path, lineno, text in iter_added_lines(diff_text):
        findings += scan_line(path, lineno, text)
    return findings


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def report(findings: Iterable[Finding], warn_only: bool) -> int:
    findings = list(findings)
    if not findings:
        print("[security-gate] 通过：未发现内部身份 / 内网信息泄漏。")
        return 0

    print("[security-gate] 发现以下疑似泄漏项：\n")
    for f in findings:
        print(f"  [{f.severity}] {f.kind}")
        print(f"        位置: {f.location}")
        print(f"        说明: {f.detail}")
    print(
        "\n处理办法：\n"
        "  1) 修改后重新提交（身份类：用对外邮箱/用户名 amend 相关 commit；"
        "内容类：删除内网 IP/代理/内部域名）。\n"
        "  2) 确属误报：在该行加 `# security-gate: allow` 豁免，"
        "或给 PR 打 `suppress-security-gate` 标签。"
    )

    if warn_only:
        print("\n[security-gate] 已打 suppress 标签，仅告警不拦截（warn-only）。")
        return 0
    return 1


def _run_self_test() -> int:
    """内置断言：规则被改坏时立刻暴露，不依赖 git 历史。"""
    # 身份：命中
    assert check_identity("abc123", "xx.xx", "xx.xx@alibaba-inc.com", "author")
    assert check_identity("abc123", "Xx Xxx", "xx.xx@list.alibaba-inc.com", "author")
    assert check_identity("abc123", "xx.xx", "someone@gmail.com", "author")
    # 身份：放行
    assert not check_identity("abc123", "Jane Doe", "jane@users.noreply.github.com", "author")

    def hits(text: str) -> List[Finding]:
        return scan_line("f.py", 1, text)

    # 内容：命中
    assert hits("host = 10.1.2.3")
    assert hits("PROXY = '172.16.0.1'")
    assert hits("export https_proxy=http://x:8080")
    assert hits("url = 'http://gw.alibaba-inc.com/api'")
    assert hits("addr = 192.168.1.1")
    # 内容：放行
    assert not hits("bind 127.0.0.1")
    assert not hits("version = 2.11.0")  # 三段，非 IPv4
    assert not hits("dns = 8.8.8.8")  # 公共 DNS 白名单
    assert not hits("host = 10.1.2.3  # security-gate: allow")  # 行内豁免
    assert not hits("mask = 255.255.255.255")

    # diff 解析：行号与路径
    diff = (
        "diff --git a/x.py b/x.py\n"
        "--- a/x.py\n"
        "+++ b/x.py\n"
        "@@ -0,0 +5,1 @@\n"
        "+ip = 10.0.0.1\n"
    )
    added = list(iter_added_lines(diff))
    assert added == [("x.py", 5, "ip = 10.0.0.1")], added

    print("[security-gate] self-test 全部通过。")
    return 0


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="内部身份 / 内网信息泄漏门禁扫描")
    parser.add_argument("--base-sha", help="提交区间起点（不含）")
    parser.add_argument("--head-sha", help="提交区间终点（含）")
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="只告警不拦截（PR 打 suppress-security-gate 标签时由 workflow 传入）",
    )
    parser.add_argument("--self-test", action="store_true", help="只跑内置断言")
    args = parser.parse_args(argv[1:])

    if args.self_test:
        return _run_self_test()

    if not args.base_sha or not args.head_sha:
        print("用法: --base-sha <base> --head-sha <head>（或 --self-test）", file=sys.stderr)
        return 2

    findings = collect_commit_findings(args.base_sha, args.head_sha)
    findings += collect_content_findings(args.base_sha, args.head_sha)
    return report(findings, args.warn_only)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
