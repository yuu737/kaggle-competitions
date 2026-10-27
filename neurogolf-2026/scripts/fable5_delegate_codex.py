from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "FABLE5_WORK" / "codex-delegations"
SCHEMA = ROOT / "FABLE5" / "CODEX_RESULT_SCHEMA.json"
WORKER_SKILL = ROOT / "skills" / "neurogolf-fable5-worker" / "SKILL.md"
CONFIRMED = ROOT / "submissions" / "submission_7190_28_verified.zip"
MAIN = ROOT / "submission.zip"
CONFIRMED_SHA = "48f8381369be0e4474fd42b35cbe14b7ecc99d278fb1fd84d4ed1f74b3e0a515"


def digest(path: Path) -> str | None:
    if not path.exists():
        return None
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def codex_command() -> Path:
    appdata = os.environ.get("APPDATA")
    candidates = [
        Path(appdata) / "npm" / "codex.cmd" if appdata else None,
        Path(shutil.which("codex.cmd") or ""),
        Path(shutil.which("codex") or ""),
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate
    raise FileNotFoundError("Codex CLI was not found; expected codex.cmd on PATH")


def slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-").lower()
    return cleaned[:48] or "delegated-task"


def build_prompt(
    task: str, run_dir: Path, scope: str, allow_main_update: bool
) -> str:
    return f"""You are the Codex execution subagent reporting to the Fable5 lead.

Read and follow:
- {WORKER_SKILL}
- {ROOT / 'FABLE5' / 'CURRENT_STATE.json'}
- relevant project evidence required by the task

Delegation run directory: {run_dir}
Intended write scope: {scope}
User-authorized root submission update: {str(allow_main_update).lower()}

The confirmed baseline archive is immutable:
{CONFIRMED}
SHA-256: {CONFIRMED_SHA}

Write experiment artifacts under:
{run_dir / 'workspace'}

Do not access credentials, submit to Kaggle, publish notebooks, or perform
external communication. Do not modify the confirmed archive. Do not modify root
submission.zip unless the authorization flag above is true. Use the project
.venv for ONNX/ORT commands. Return the final result in Japanese and conform
exactly to the supplied JSON schema.

Fable5 delegated task:

{task}
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Codex as a bounded NeuroGolf worker delegated by Fable5."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--task-file", type=Path)
    source.add_argument("--task")
    parser.add_argument("--check-auth", action="store_true")
    parser.add_argument("--title", default="delegated-task")
    parser.add_argument("--scope", choices=("experiment", "project"), default="experiment")
    parser.add_argument("--sandbox", choices=("read-only", "workspace-write"), default="workspace-write")
    parser.add_argument("--model")
    parser.add_argument("--allow-main-update", action="store_true")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    executable = codex_command()
    if args.check_auth:
        status = subprocess.run(
            [str(executable), "login", "status"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
        api_key_configured = bool(
            os.environ.get("CODEX_API_KEY") or os.environ.get("OPENAI_API_KEY")
        )
        report = {
            "codex": str(executable),
            "authenticated": status.returncode == 0 or api_key_configured,
            "api_key_configured": api_key_configured,
            "status": (status.stdout + status.stderr).strip(),
            "main_sha256": digest(MAIN),
            "confirmed_sha256": digest(CONFIRMED),
            "confirmed_sha_match": digest(CONFIRMED) == CONFIRMED_SHA,
        }
        print(json.dumps(report, ensure_ascii=False))
        raise SystemExit(0 if report["authenticated"] else 1)
    if not args.task_file and args.task is None:
        parser.error("one of --task-file or --task is required unless --check-auth is used")

    task = (
        args.task_file.resolve().read_text(encoding="utf-8")
        if args.task_file
        else str(args.task)
    )
    run_id = (
        f"{time.strftime('%Y%m%d-%H%M%S')}-"
        f"{slug(args.title)}-{uuid.uuid4().hex[:8]}"
    )
    run_dir = RUNS / run_id
    workspace = run_dir / "workspace"
    workspace.mkdir(parents=True)
    (run_dir / "task.md").write_text(task, encoding="utf-8")
    prompt = build_prompt(task, run_dir, args.scope, args.allow_main_update)
    (run_dir / "prompt.md").write_text(prompt, encoding="utf-8")

    before = {"main": digest(MAIN), "confirmed": digest(CONFIRMED)}
    command = [
        str(executable),
        "exec",
        "-C",
        str(ROOT),
        "--sandbox",
        args.sandbox,
        "-c",
        'approval_policy="never"',
        "-c",
        'shell_environment_policy.inherit="core"',
        "-c",
        'shell_environment_policy.exclude=["OPENAI_API_KEY","CODEX_API_KEY","KAGGLE_KEY","KAGGLE_USERNAME"]',
        "--ephemeral",
        "--json",
        "--output-schema",
        str(SCHEMA),
        "--output-last-message",
        str(run_dir / "result.json"),
    ]
    if args.model:
        command.extend(["--model", args.model])
    command.append("-")

    manifest: dict[str, object] = {
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scope": args.scope,
        "sandbox": args.sandbox,
        "allow_main_update": args.allow_main_update,
        "task_file": str((run_dir / "task.md").resolve()),
        "result_file": str((run_dir / "result.json").resolve()),
        "baseline_before": before,
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        manifest["status"] = "dry-run"
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(manifest, ensure_ascii=False))
        return

    child_env = dict(os.environ)
    for key in ("KAGGLE_KEY", "KAGGLE_USERNAME"):
        child_env.pop(key, None)
    try:
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=child_env,
            input=prompt,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=args.timeout,
        )
        returncode = process.returncode
        stdout = process.stdout
        stderr = process.stderr
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + "\nCodex delegation timed out."

    (run_dir / "events.jsonl").write_text(stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(stderr, encoding="utf-8")
    after = {"main": digest(MAIN), "confirmed": digest(CONFIRMED)}
    guard_errors = []
    if after["confirmed"] != CONFIRMED_SHA or after["confirmed"] != before["confirmed"]:
        guard_errors.append("confirmed baseline archive changed")
    if not args.allow_main_update and after["main"] != before["main"]:
        guard_errors.append("root submission.zip changed without authorization")
    manifest.update(
        {
            "status": "complete" if returncode == 0 and not guard_errors else "failed",
            "returncode": returncode,
            "baseline_after": after,
            "guard_errors": guard_errors,
        }
    )
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "run_id": run_id,
                "result": manifest["result_file"],
                "manifest": str(manifest_path.resolve()),
                "guard_errors": guard_errors,
            },
            ensure_ascii=False,
        )
    )
    raise SystemExit(0 if manifest["status"] == "complete" else 1)


if __name__ == "__main__":
    main()
