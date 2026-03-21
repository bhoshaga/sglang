"""Generate a Markdown dashboard for diffusion cross-framework comparisons.

Reads current comparison results + historical data from sglang-ci-data repo
and produces a Markdown report with tables and Mermaid trend charts.

Usage:
    python3 scripts/ci/utils/generate_diffusion_dashboard.py \
        --results comparison-results.json \
        --output dashboard.md \
        --history-dir history/           # optional, local history JSONs
        --fetch-history                  # fetch from GitHub API instead
"""

import argparse
import base64
import io
import json
import os
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# History fetching (from sglang-ci-data repo via GitHub API)
# ---------------------------------------------------------------------------

CI_DATA_REPO_OWNER = "sglang-bot"
CI_DATA_REPO_NAME = "sglang-ci-data"
CI_DATA_BRANCH = "main"
HISTORY_PREFIX = "diffusion-comparisons"
MAX_HISTORY_RUNS = 7


def _github_get(url: str, token: str) -> dict | list | None:
    """Simple GET to GitHub API."""
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    req = Request(url, headers=headers)
    try:
        with urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        print(f"  Warning: GitHub API request failed ({e.code}): {url}")
        return None
    except Exception as e:
        print(f"  Warning: GitHub API request error: {e}")
        return None


def fetch_history_from_github(token: str) -> list[dict]:
    """Fetch recent comparison result JSONs from sglang-ci-data repo."""
    print("Fetching historical comparison data from GitHub...")
    url = (
        f"https://api.github.com/repos/{CI_DATA_REPO_OWNER}/{CI_DATA_REPO_NAME}"
        f"/contents/{HISTORY_PREFIX}?ref={CI_DATA_BRANCH}"
    )
    listing = _github_get(url, token)
    if not listing or not isinstance(listing, list):
        print("  No historical data found.")
        return []

    # Filter JSON files and sort by name (date prefix) descending
    json_files = sorted(
        [f for f in listing if f["name"].endswith(".json")],
        key=lambda f: f["name"],
        reverse=True,
    )[:MAX_HISTORY_RUNS]

    history = []
    for entry in json_files:
        raw_url = entry.get("download_url")
        if not raw_url:
            continue
        data = _github_get(raw_url, token)
        if data and isinstance(data, dict):
            history.append(data)
    print(f"  Loaded {len(history)} historical run(s).")
    return history


def load_history_from_dir(history_dir: str) -> list[dict]:
    """Load historical JSONs from a local directory."""
    if not os.path.isdir(history_dir):
        return []
    files = sorted(
        [f for f in os.listdir(history_dir) if f.endswith(".json")],
        reverse=True,
    )[:MAX_HISTORY_RUNS]
    history = []
    for fname in files:
        try:
            with open(os.path.join(history_dir, fname)) as f:
                history.append(json.load(f))
        except Exception:
            pass
    return history


# ---------------------------------------------------------------------------
# Dashboard generation
# ---------------------------------------------------------------------------


def _fmt_latency(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{val:.2f}"


def _fmt_speedup(sglang_lat: float | None, other_lat: float | None) -> str:
    if sglang_lat is None or other_lat is None or sglang_lat <= 0:
        return "N/A"
    ratio = other_lat / sglang_lat
    return f"{ratio:.2f}x"


def _short_date(ts: str) -> str:
    """Extract short date from ISO timestamp."""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%b %d")
    except Exception:
        return ts[:10]


def _short_sha(sha: str) -> str:
    return sha[:7] if sha and sha != "unknown" else "?"


def _trend_emoji(current: float | None, previous: float | None) -> str:
    if current is None or previous is None:
        return ""
    diff_pct = (current - previous) / previous * 100
    if diff_pct < -2:
        return " :arrow_down:"  # faster (good)
    elif diff_pct > 2:
        return " :arrow_up:"  # slower (bad)
    return " :left_right_arrow:"


def _extract_case_results(run_data: dict) -> dict[str, dict[str, float | None]]:
    """Extract {case_id: {framework: latency}} from a run."""
    mapping: dict[str, dict[str, float | None]] = {}
    for r in run_data.get("results", []):
        cid = r["case_id"]
        fw = r["framework"]
        if cid not in mapping:
            mapping[cid] = {}
        mapping[cid][fw] = r.get("latency_s")
    return mapping


def generate_dashboard(
    current: dict,
    history: list[dict],
) -> str:
    """Generate full markdown dashboard."""
    lines: list[str] = []
    lines.append("# Diffusion Cross-Framework Performance Dashboard\n")
    ts = current.get("timestamp", datetime.now(timezone.utc).isoformat())
    sha = current.get("commit_sha", "unknown")
    lines.append(f"*Generated: {_short_date(ts)} | Commit: `{_short_sha(sha)}`*\n")

    current_cases = _extract_case_results(current)
    case_ids = list(current_cases.keys())

    # ---- Regression detection ----
    REGRESSION_THRESHOLD = 0.05  # 5%
    regressions: list[str] = []
    if history:
        prev_cases = _extract_case_results(history[0])
        for cid in case_ids:
            for fw in ("sglang", "vllm-omni"):
                cur = current_cases.get(cid, {}).get(fw)
                prev = prev_cases.get(cid, {}).get(fw)
                if cur and prev and prev > 0:
                    pct = (cur - prev) / prev
                    if pct > REGRESSION_THRESHOLD:
                        regressions.append(
                            f"**{cid}** ({fw}): {prev:.2f}s -> {cur:.2f}s "
                            f"(+{pct*100:.1f}%)"
                        )

    if regressions:
        lines.append("> [!WARNING]\n> **Performance Regression Detected**\n>")
        for reg in regressions:
            lines.append(f"> - {reg}")
        lines.append("\n")

    # Discover all frameworks present in results
    all_frameworks = []
    seen_fw = set()
    for r in current.get("results", []):
        fw = r["framework"]
        if fw not in seen_fw:
            all_frameworks.append(fw)
            seen_fw.add(fw)
    # Ensure sglang is first
    if "sglang" in all_frameworks:
        all_frameworks.remove("sglang")
        all_frameworks.insert(0, "sglang")
    other_frameworks = [fw for fw in all_frameworks if fw != "sglang"]

    # ---- Section 1: Cross-Framework Comparison (current run) ----
    lines.append("## Cross-Framework Performance Comparison\n")

    # Dynamic header
    header = "| Model |"
    sep = "|-------|"
    for fw in all_frameworks:
        header += f" {fw} (s) |"
        sep += "---------|"
    for ofw in other_frameworks:
        header += f" vs {ofw} |"
        sep += "---------|"
    lines.append(header)
    lines.append(sep)

    # One row per case (deduplicated by case_id)
    seen_cases = set()
    for r in current.get("results", []):
        cid = r["case_id"]
        if cid in seen_cases:
            continue
        seen_cases.add(cid)

        case_fws = current_cases.get(cid, {})
        sg_lat = case_fws.get("sglang")

        row = f"| {r['model'].split('/')[-1]} |"
        # Latency columns — bold the fastest
        lats = {fw: case_fws.get(fw) for fw in all_frameworks}
        valid_lats = [v for v in lats.values() if v is not None]
        min_lat = min(valid_lats) if valid_lats else None
        for fw in all_frameworks:
            lat = lats[fw]
            if lat is not None and min_lat is not None and lat == min_lat:
                row += f" **{_fmt_latency(lat)}** |"
            else:
                row += f" {_fmt_latency(lat)} |"
        # Speedup columns
        for ofw in other_frameworks:
            row += f" {_fmt_speedup(sg_lat, case_fws.get(ofw))} |"
        lines.append(row)

    # ---- Section 2: SGLang Performance Trend ----
    if history:
        lines.append("\n## SGLang Performance Trend (Last 7 Runs)\n")

        # Build header
        header = "| Date | Commit |"
        sep = "|------|--------|"
        for cid in case_ids:
            header += f" {cid} (s) |"
            sep += "---------|"
        header += " Trend |"
        sep += "-------|"
        lines.append(header)
        lines.append(sep)

        # Current run first
        all_runs = [current] + history
        for i, run in enumerate(all_runs):
            run_cases = _extract_case_results(run)
            date = _short_date(run.get("timestamp", ""))
            sha_s = _short_sha(run.get("commit_sha", ""))
            row = f"| {date} | `{sha_s}` |"
            for cid in case_ids:
                lat = run_cases.get(cid, {}).get("sglang")
                row += f" {_fmt_latency(lat)} |"
            # Trend vs next (older) run
            if i + 1 < len(all_runs):
                prev_cases = _extract_case_results(all_runs[i + 1])
                emojis = []
                for cid in case_ids:
                    cur = run_cases.get(cid, {}).get("sglang")
                    prev = prev_cases.get(cid, {}).get("sglang")
                    emojis.append(_trend_emoji(cur, prev))
                row += " ".join(emojis) + " |"
            else:
                row += " — |"
            lines.append(row)

    # ---- Section 3: Cross-Framework Speedup Trend (only if multiple frameworks) ----
    if history and other_frameworks:
        lines.append("\n## SGLang vs vLLM-Omni Speedup Over Time\n")

        header = "| Date |"
        sep = "|------|"
        for cid in case_ids:
            header += f" {cid} |"
            sep += "---------|"
        lines.append(header)
        lines.append(sep)

        all_runs = [current] + history
        for run in all_runs:
            run_cases = _extract_case_results(run)
            date = _short_date(run.get("timestamp", ""))
            row = f"| {date} |"
            for cid in case_ids:
                sg = run_cases.get(cid, {}).get("sglang")
                vl = run_cases.get(cid, {}).get("vllm-omni")
                row += f" {_fmt_speedup(sg, vl)} |"
            lines.append(row)

    # ---- Section 4: Matplotlib Trend Charts (embedded as base64 PNG) ----
    if history:
        all_runs = list(reversed([current] + history))  # chronological order

        def _chart_label(run: dict) -> str:
            d = _short_date(run.get("timestamp", ""))
            s = _short_sha(run.get("commit_sha", ""))
            return f"{d}\n({s})"

        def _fig_to_base64(fig) -> str:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("utf-8")

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Per-case latency trend charts
            for cid in case_ids:
                labels = []
                sg_vals = []
                vl_vals = []
                for run in all_runs:
                    run_cases = _extract_case_results(run)
                    sg = run_cases.get(cid, {}).get("sglang")
                    vl = run_cases.get(cid, {}).get("vllm-omni")
                    if sg is None:
                        continue
                    labels.append(_chart_label(run))
                    sg_vals.append(sg)
                    vl_vals.append(vl)

                if not sg_vals:
                    continue

                has_vl = any(v is not None for v in vl_vals)
                fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4))

                # SGLang line
                ax.plot(
                    range(len(sg_vals)),
                    sg_vals,
                    "o-",
                    color="#2563eb",
                    linewidth=2,
                    markersize=6,
                    label="SGLang",
                )
                for i, v in enumerate(sg_vals):
                    ax.annotate(
                        f"{v:.2f}s",
                        (i, v),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                        fontsize=8,
                        fontweight="bold",
                        color="#2563eb",
                    )

                # vLLM-Omni line (if data exists)
                if has_vl:
                    vl_clean = [v if v is not None else float("nan") for v in vl_vals]
                    ax.plot(
                        range(len(vl_clean)),
                        vl_clean,
                        "s--",
                        color="#dc2626",
                        linewidth=2,
                        markersize=5,
                        label="vLLM-Omni",
                    )
                    for i, v in enumerate(vl_vals):
                        if v is not None:
                            ax.annotate(
                                f"{v:.2f}s",
                                (i, v),
                                textcoords="offset points",
                                xytext=(0, -14),
                                ha="center",
                                fontsize=8,
                                color="#dc2626",
                            )

                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, fontsize=7)
                ax.set_ylabel("Latency (s)")
                ax.set_title(f"Latency Trend — {cid}", fontsize=11, fontweight="bold")
                ax.legend(loc="upper right", fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(bottom=0)

                b64 = _fig_to_base64(fig)
                plt.close(fig)

                lines.append(f"\n### Latency Trend: {cid}\n")
                lines.append(f"![Latency Trend {cid}](data:image/png;base64,{b64})\n")

            # Speedup trend chart (only if multiple frameworks)
            if other_frameworks:
                fig, ax = plt.subplots(figsize=(max(6, len(all_runs) * 1.2), 4))
                colors = ["#2563eb", "#dc2626", "#16a34a", "#ea580c"]
                for ci, cid in enumerate(case_ids):
                    speedups = []
                    run_labels = []
                    for run in all_runs:
                        run_cases = _extract_case_results(run)
                        sg = run_cases.get(cid, {}).get("sglang")
                        vl = run_cases.get(cid, {}).get("vllm-omni")
                        if sg and vl and sg > 0:
                            speedups.append(vl / sg)
                        else:
                            speedups.append(None)
                        run_labels.append(_chart_label(run))
                    clean = [v if v is not None else float("nan") for v in speedups]
                    ax.plot(
                        range(len(clean)),
                        clean,
                        "o-",
                        color=colors[ci % len(colors)],
                        linewidth=2,
                        markersize=5,
                        label=cid,
                    )

                ax.set_xticks(range(len(run_labels)))
                ax.set_xticklabels(run_labels, fontsize=7)
                ax.set_ylabel("Speedup (x)")
                ax.set_title(
                    "SGLang Speedup Over vLLM-Omni", fontsize=11, fontweight="bold"
                )
                ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
                ax.legend(loc="upper left", fontsize=7)
                ax.grid(True, alpha=0.3)

                b64 = _fig_to_base64(fig)
                plt.close(fig)

                lines.append("\n### Speedup Trend (SGLang vs vLLM-Omni)\n")
                lines.append(f"![Speedup Trend](data:image/png;base64,{b64})\n")

        except ImportError:
            lines.append(
                "\n*Charts unavailable (matplotlib not installed)*\n"
            )

    # Footer
    lines.append("\n---")
    lines.append(
        "*Generated by `generate_diffusion_dashboard.py` in SGLang nightly CI.*"
    )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate diffusion cross-framework comparison dashboard"
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to comparison-results.json from current run",
    )
    parser.add_argument(
        "--output",
        default="dashboard.md",
        help="Output markdown file path",
    )
    parser.add_argument(
        "--history-dir",
        default=None,
        help="Local directory containing historical comparison JSONs",
    )
    parser.add_argument(
        "--fetch-history",
        action="store_true",
        help="Fetch history from sglang-ci-data GitHub repo",
    )
    parser.add_argument(
        "--step-summary",
        action="store_true",
        help="Also write to $GITHUB_STEP_SUMMARY",
    )

    args = parser.parse_args()

    # Load current results
    with open(args.results) as f:
        current = json.load(f)
    print(f"Loaded current results: {len(current.get('results', []))} entries")

    # Load history
    history: list[dict] = []
    if args.fetch_history:
        token = os.environ.get("GH_PAT_FOR_NIGHTLY_CI_DATA") or os.environ.get(
            "GITHUB_TOKEN"
        )
        if token:
            history = fetch_history_from_github(token)
        else:
            print("Warning: No GitHub token available, skipping history fetch")
    elif args.history_dir:
        history = load_history_from_dir(args.history_dir)
        print(f"Loaded {len(history)} historical run(s) from {args.history_dir}")

    # Generate dashboard
    markdown = generate_dashboard(current, history)

    # Write output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write(markdown)
    print(f"Dashboard written to {args.output}")

    # Write to GitHub Step Summary
    if args.step_summary:
        summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary_file:
            with open(summary_file, "a") as f:
                f.write(markdown)
            print("Dashboard appended to $GITHUB_STEP_SUMMARY")
        else:
            print("Warning: $GITHUB_STEP_SUMMARY not set, skipping")


if __name__ == "__main__":
    main()
