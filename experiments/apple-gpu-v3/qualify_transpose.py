"""Qualify the exact transpose+pointwise matrix builder; no import-time GPU work.

The unchanged V2 qualifier uses before_compile/before as internal aliases for
the combined candidate and its original eager input-history reference. This is
state/numerical qualification only; no paced or performance sweep is added.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import sys

HERE = Path(__file__).resolve().parent
V2 = HERE.parent / "apple-gpu-v2"
TRUE_ARM = "transpose_pointwise_mm"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("Use a simple output name")
    output = HERE / (name + ".json")
    if output.exists():
        raise FileExistsError(output)
    for key in ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH", "TORCHINDUCTOR_USE_FAST_MATH"):
        if os.environ.get(key, "0") != "0":
            raise RuntimeError(key + " must be disabled before Torch import")
        os.environ[key] = "0"
    os.environ.pop("PYTORCH_MPS_PREFER_METAL", None)
    sys.path.insert(0, str(V2))
    import experiment as exp
    import qualify_candidates as qual
    import screen_transpose as screen

    sources = {str(path): sha(path) for path in
               (Path(__file__).resolve(), HERE / "screen_transpose.py", V2 / "qualify_candidates.py")}
    original_main, original_build = exp.main, exp.build

    def qualify_instead_of_screen(args):
        # screen.main has now installed its exact local builder, after running
        # the same CPU algebra gates. Do not copy or reimplement that builder.
        screen_builder = exp.build
        if args.arms != ["before_compile", "transpose_mm", TRUE_ARM]:
            raise RuntimeError("Transpose screen arm contract changed")

        def alias_build(arm):
            if arm == "before_compile":
                return screen_builder(TRUE_ARM)
            if arm == "before":
                return original_build("before")
            raise ValueError("Unexpected qualification alias: " + arm)

        exp.build = alias_build
        errors = []
        try:
            qual.main(output, ["before_compile"])
        finally:
            exp.build = screen_builder
            if output.exists():
                result = json.loads(output.read_text())
                result["qualification_adapter"] = dict(
                    true_arm=TRUE_ARM,
                    aliases={"before_compile": TRUE_ARM, "before": "eager_original_input_history"},
                    aliases_retained_in_raw_labels=True,
                    builder="screen_transpose.main local build; exact existing combined route",
                    expected_compilation_first_calls=4, maximum_unique_graphs=2,
                    counter_after_timing_means="After qualification, not a performance sweep",
                    files=sources, sources_unchanged=all(sha(Path(p)) == h for p, h in sources.items()))
                result.setdefault("files", {}).update(sources)
                if "protocol" in result:
                    result["protocol"]["state_reference"] = (
                        "Original eager decoder with the same input-history representation; "
                        "candidate and reference evolve their own states.")
                    result["protocol"]["exclusions"] = (
                        "Model construction and four candidate preparation calls are reported "
                        "separately; those call times include compilation plus completed decode.")
                preparation = result.get("preparation", [])
                if len(preparation) == 1:
                    def selected_counters(key):
                        return {k: v for k, v in preparation[0].get(key, {}).items()
                                if k in ("stats", "frames", "graph_break", "unimplemented")}
                    before = selected_counters("compiler_counters")
                    after = selected_counters("final_compiler_counters")
                    result["compile_counters_after_preparation"] = before
                    result["compile_counters_after_timing"] = after
                    if result.get("status") == "passed":
                        if not 0 < before.get("stats", {}).get("unique_graphs", 0) <= 2:
                            errors.append("Exceeded two fixed compilation graphs")
                        if before != after:
                            errors.append("Compilation counters changed after preparation")
                elif result.get("status") == "passed":
                    errors.append("Expected exactly one qualified candidate")
                if result.get("status") == "passed" and result.get("compilation_first_calls") != 4:
                    errors.append("Expected four separately accounted initial compiled calls")
                if not result["qualification_adapter"]["sources_unchanged"]:
                    errors.append("Qualification adapter source changed")
                if errors:
                    result.update(status="failed", adapter_errors=errors)
                if result.get("status") == "passed":
                    result["qualified_arms"] = [TRUE_ARM]
                output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        if errors:
            raise RuntimeError("; ".join(errors))

    exp.main = qualify_instead_of_screen
    try:
        screen.main(argparse.Namespace(name=name, cpu_only=False, combined=True))
    finally:
        exp.main, exp.build = original_main, original_build
        if output.exists():
            result = json.loads(output.read_text())
            if "transpose_screen" in result:
                # The outer screen appends its usual performance-harness label;
                # replace that label because its exp.main call was intercepted.
                result["transpose_screen"]["harness"] = (
                    "Unchanged V2 qualify_candidates.main: one combined arm, seven short "
                    "fixtures, 39 paired state checks, 93 total model/API calls; no timing sweep.")
                result["transpose_screen"]["compile_graph_limit"] = 2
            output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    main(parser.parse_args().name)
