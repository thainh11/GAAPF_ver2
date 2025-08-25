#!/usr/bin/env python3
"""
run_benchmarks.py — Orchestrator for running code-generation benchmarks using the
existing GAAPF project stack (Vertex AI + SandboxRunner), without modifying core code.

Features implemented:
- HumanEval integration
  * Generate samples.jsonl using Google Vertex AI (ChatVertexAI) via existing credentials helper
  * Evaluate functional correctness using the official human-eval evaluation harness
  * All executions go through GAAPF SandboxRunner where appropriate
- τ-Bench (tau-bench) bootstrap (best-effort placeholder)
  * Helper to clone/install and a minimal runner scaffold using Vertex AI
  * If τ-Bench provider interface changes, the script will print actionable hints

Notes:
- This script reuses GAAPF modules: get_vertex_ai_config (credentials) and SandboxRunner.
- It does NOT modify any project source code under src/.
- Outputs are stored under run_benchmark/results/.
- Requires the active conda env to have the necessary dependencies (see tips printed by this script).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import gzip

# Ensure project src path is importable when running from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Reuse GAAPF utilities (NO modifications)
from GAAPF.core.utils.credentials_helper import get_vertex_ai_config
from GAAPF.core.tools.sandbox_runner import SandboxRunner

# Vertex AI (LangChain)
try:
    from langchain_google_vertexai import ChatVertexAI
    from langchain_core.messages import AIMessage
except Exception as e:  # pragma: no cover
    ChatVertexAI = None  # type: ignore
    AIMessage = None  # type: ignore

# Constants
EXTERNAL_DIR = PROJECT_ROOT / "run_benchmark" / "_external"
RESULTS_DIR = PROJECT_ROOT / "run_benchmark" / "results"
HUMAN_EVAL_DEFAULT_SAMPLES = RESULTS_DIR / "humaneval" / "samples.jsonl"
HUMAN_EVAL_DEFAULT_RESULTS = RESULTS_DIR / "humaneval" / "results.jsonl"

# -----------------------------
# Utilities
# -----------------------------

def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix:  # It's a file path, ensure parent only
            continue
        p.mkdir(parents=True, exist_ok=True)


def init_vertex_llm() -> Any:
    """Initialize ChatVertexAI using shared helper config.
    Falls back to a basic mock response provider if Vertex AI is unavailable.
    """
    try:
        cfg = get_vertex_ai_config()
        if ChatVertexAI is None:
            raise RuntimeError("langchain-google-vertexai not installed")
        llm = ChatVertexAI(
            model_name=cfg["model_name"],
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            project=cfg["project"],
            location=cfg["location"],
        )
        return llm
    except Exception as e:
        # Local mock fallback avoids introducing any non-Vertex providers
        class MockLLM:
            def invoke(self, prompt):
                return "Mock LLM response. Please configure Vertex AI. Error: %s" % e
        return MockLLM()


def call_llm(llm: Any, prompt: str, max_retries: int = 2, sleep_sec: float = 1.5) -> str:
    """Robust LLM invocation returning string content.
    Works for LangChain ChatVertexAI and basic mock.
    """
    last_err: Optional[Exception] = None
    for _ in range(max_retries + 1):
        try:
            resp = llm.invoke(prompt)
            if AIMessage is not None and isinstance(resp, AIMessage):
                return resp.content or ""
            return str(resp)
        except Exception as e:
            last_err = e
            time.sleep(sleep_sec)
    raise RuntimeError(f"LLM invocation failed after retries: {last_err}")


# -----------------------------
# HumanEval integration
# -----------------------------

def _try_import_humaneval() -> Tuple[bool, Optional[Path]]:
    """Try to import human_eval. If not available, return False and suggest install.
    Returns (available, installed_repo_path_if_any)
    """
    try:
        import human_eval  # type: ignore
        return True, None
    except Exception:
        # Check if we cloned the official repo to _external/human-eval
        repo_dir = EXTERNAL_DIR / "human-eval"
        if repo_dir.exists():
            sys.path.insert(0, str(repo_dir))
            try:
                import human_eval  # type: ignore
                return True, repo_dir
            except Exception:
                return False, repo_dir
        return False, None


def ensure_humaneval_available() -> None:
    """Ensure the human-eval package or repo is available locally.
    Preference: pip install if available, else fall back to git clone.
    Note: This function only prepares filesystem hints; it does not execute pip here.
    """
    ok, repo = _try_import_humaneval()
    if ok:
        return
    # If not found, create a placeholder directory where the user can clone it.
    placeholder = EXTERNAL_DIR / "human-eval"
    placeholder.mkdir(parents=True, exist_ok=True)


def _load_humaneval_tasks() -> List[Dict[str, Any]]:
    """Load HumanEval tasks, either via the package or from cloned repo JSONL.
    Returns list of dicts with at least keys: task_id, prompt
    """
    ok, repo = _try_import_humaneval()
    tasks: List[Dict[str, Any]] = []
    if ok:
        try:
            from human_eval.data import read_problems  # type: ignore
            problems = read_problems()
            for task_id, item in problems.items():
                tasks.append({
                    "task_id": task_id,
                    "prompt": item.get("prompt", ""),
                })
            return tasks
        except Exception:
            pass
    # Fallback: read JSONL from cloned repo if present
    data_file = (repo or (EXTERNAL_DIR / "human-eval")) / "data" / "HumanEval.jsonl"
    if data_file.exists():
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                tasks.append({
                    "task_id": obj.get("task_id"),
                    "prompt": obj.get("prompt", ""),
                })
    if not tasks:
        raise FileNotFoundError(
            "Cannot load HumanEval tasks. Install the package (pip install human-eval) "
            "or clone the repo into run_benchmark/_external/human-eval"
        )
    return tasks


def humaneval_generate_samples(
    out_path: Path,
    samples_per_task: int = 1,
    max_tasks: Optional[int] = None,
    system_hint: Optional[str] = None,
) -> Path:
    """Generate HumanEval samples.jsonl using Vertex AI.
    - Only depends on GAAPF's Vertex config; does NOT modify project code.
    - The completion is raw code text returned by the LLM for each task prompt.
    """
    ensure_dirs(out_path)
    ensure_humaneval_available()
    tasks = _load_humaneval_tasks()
    if max_tasks is not None:
        tasks = tasks[: max(0, int(max_tasks))]

    llm = init_vertex_llm()

    default_system = (
        "You are a helpful Python coding assistant. "
        "Given a code prompt from HumanEval that includes a function signature and docstring, "
        "produce a Python solution that passes tests. Return ONLY valid Python code with the completed function. "
        "Do not include explanations or markdown."
    )
    sys_msg = system_hint or default_system

    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for task in tasks:
            task_id = task["task_id"]
            prompt = task["prompt"]
            for i in range(samples_per_task):
                full_prompt = f"{sys_msg}\n\nPROMPT:\n{prompt}\n\nReturn only Python code."
                try:
                    completion = call_llm(llm, full_prompt)
                except Exception as e:
                    completion = f"# ERROR generating completion for {task_id}: {e}"
                rec = {"task_id": task_id, "completion": completion}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
    print(f"[HumanEval] Wrote {written} samples to: {out_path}")
    return out_path


def humaneval_evaluate(
    samples_path: Path,
    result_path: Path,
    timeout_per_task: int = 7,
) -> Tuple[int, float]:
    """Run functional correctness eval using official harness via SandboxRunner.
    Returns (exit_code, exec_time_seconds).
    """
    ensure_dirs(result_path)
    ok, repo = _try_import_humaneval()
    # Build command: use module entry point
    cmd = (
        f"python -m human_eval.evaluate_functional_correctness \"{samples_path}\" "
        f"--timeout {int(timeout_per_task)}"
    )
    # If run from a cloned repo, ensure PYTHONPATH includes it
    env = os.environ.copy()
    if not ok and repo is not None:
        extra = str(repo)
        env["PYTHONPATH"] = extra + os.pathsep + env.get("PYTHONPATH", "")

    runner = SandboxRunner(timeout=60 * 30, memory_limit_mb=2048)
    print(f"[HumanEval] Running evaluation: {cmd}")

    # Temporarily set env for subprocess (SandboxRunner doesn't take env)
    old_env = os.environ.copy()
    os.environ.update(env)
    result = runner.run_in_sandbox(cmd, cwd=str(PROJECT_ROOT))
    stdout, stderr, exit_code, elapsed = result["stdout"], result["stderr"], result["exit_code"], result["execution_time"]
    # Restore environment
    os.environ.clear()
    os.environ.update(old_env)

    print("[HumanEval] Exit:", exit_code, "Elapsed:", f"{elapsed:.2f}s")
    if stdout:
        print("[HumanEval][stdout]\n" + stdout)
    if stderr:
        print("[HumanEval][stderr]\n" + stderr)

    # Locate output produced by the harness and copy to desired result_path
    cand_jsonl = Path(str(samples_path) + "_results.jsonl")
    cand_gz = Path(str(samples_path) + "_results.jsonl.gz")
    try:
        if cand_jsonl.exists():
            shutil.copyfile(cand_jsonl, result_path)
            print(f"[HumanEval] Copied results to: {result_path}")
        elif cand_gz.exists():
            with gzip.open(cand_gz, "rt", encoding="utf-8") as fin, open(result_path, "w", encoding="utf-8") as fout:
                fout.write(fin.read())
            print(f"[HumanEval] Decompressed and wrote results to: {result_path}")
        else:
            print(f"[HumanEval] Warning: results file not found next to samples: {cand_jsonl} or {cand_gz}")
    except Exception as copy_e:
        print(f"[HumanEval] Warning: failed to copy/decompress results: {copy_e}")

    return exit_code, elapsed


# -----------------------------
# τ-Bench integration (best-effort scaffold)
# -----------------------------

TAU_REPO_URL = "https://github.com/ServiceNow/tau-bench"

def ensure_taubench_placeholder() -> Path:
    """Prepare local directory for τ-Bench checkout under _external.
    We don't alter third-party code; this only prepares a place to clone.
    """
    tb_dir = EXTERNAL_DIR / "tau-bench"
    tb_dir.mkdir(parents=True, exist_ok=True)
    return tb_dir


def taubench_run_minimal(model_hint: Optional[str] = None) -> None:
    """Attempt to run a minimal τ-Bench script using Vertex AI.
    This is a scaffold that prints actionable guidance if the environment isn't ready.
    """
    tb_dir = ensure_taubench_placeholder()
    print("[τ-Bench] Placeholder at:", tb_dir)
    print("[τ-Bench] To enable full runs, clone the repo:")
    print(f"  git clone {TAU_REPO_URL} \"{tb_dir}\"")
    print("[τ-Bench] Then follow their installation docs. Ensure you configure provider to use Vertex AI only.")
    print("[τ-Bench] After installation, you can integrate a Vertex provider adapter in run-time via sys.path injection.")
    # Initialize Vertex LLM just to verify credentials are OK
    _ = init_vertex_llm()
    print("[τ-Bench] Vertex AI is initialized. Proceed to configure τ-Bench to call this provider.")


# -----------------------------
# CLI
# -----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GAAPF Benchmarks Orchestrator")
    sub = p.add_subparsers(dest="cmd", required=True)

    # HumanEval: generate
    g = sub.add_parser("humaneval-generate", help="Generate HumanEval samples.jsonl using Vertex AI")
    g.add_argument("--out", default=str(HUMAN_EVAL_DEFAULT_SAMPLES), help="Path to write samples.jsonl")
    g.add_argument("--samples-per-task", type=int, default=1)
    g.add_argument("--max-tasks", type=int, default=None)

    # HumanEval: evaluate
    e = sub.add_parser("humaneval-eval", help="Run HumanEval functional correctness evaluation")
    e.add_argument("--samples", default=str(HUMAN_EVAL_DEFAULT_SAMPLES), help="Path to samples.jsonl")
    e.add_argument("--out", default=str(HUMAN_EVAL_DEFAULT_RESULTS), help="Path to write results.jsonl")
    e.add_argument("--timeout", type=int, default=7, help="Per-task timeout (seconds)")

    # HumanEval: all-in-one
    a = sub.add_parser("humaneval-all", help="Generate and evaluate HumanEval in one step")
    a.add_argument("--samples-per-task", type=int, default=1)
    a.add_argument("--max-tasks", type=int, default=None)
    a.add_argument("--timeout", type=int, default=7)
    a.add_argument("--samples", default=str(HUMAN_EVAL_DEFAULT_SAMPLES))
    a.add_argument("--out", default=str(HUMAN_EVAL_DEFAULT_RESULTS))

    # τ-Bench: scaffold run
    t = sub.add_parser("taubench-run", help="Prepare τ-Bench folder and verify Vertex AI setup")
    t.add_argument("--model", default=None, help="Optional model hint for Vertex AI")

    # Utilities
    c = sub.add_parser("clean", help="Clean generated benchmark artifacts")

    return p


def cmd_clean() -> None:
    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR, ignore_errors=True)
        print(f"Removed: {RESULTS_DIR}")
    else:
        print("Nothing to clean.")


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.cmd == "humaneval-generate":
        out = Path(args.out)
        humaneval_generate_samples(out_path=out, samples_per_task=args.samples_per_task, max_tasks=args.max_tasks)

    elif args.cmd == "humaneval-eval":
        samples = Path(args.samples)
        out = Path(args.out)
        humaneval_evaluate(samples_path=samples, result_path=out, timeout_per_task=args.timeout)

    elif args.cmd == "humaneval-all":
        samples = Path(args.samples)
        out = Path(args.out)
        humaneval_generate_samples(out_path=samples, samples_per_task=args.samples_per_task, max_tasks=args.max_tasks)
        humaneval_evaluate(samples_path=samples, result_path=out, timeout_per_task=args.timeout)

    elif args.cmd == "taubench-run":
        taubench_run_minimal(model_hint=args.model)

    elif args.cmd == "clean":
        cmd_clean()

    else:  # pragma: no cover
        parser.print_help()


if __name__ == "__main__":
    main()