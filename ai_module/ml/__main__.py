"""CLI: ``python -m ai_module.ml [--dataset mimic_icu] [--target icu_flag] [--no-llm] [--json]``

Runs the whole comparison in-process (same steps the Langflow flow runs node by node)."""
from __future__ import annotations

import argparse
import json
import sys

from ai_module.config.logging import configure_logging
from ai_module.config.settings import get_settings
from ai_module.ml.errors import MLAnalysisError
from ai_module.ml.service import MLAnalysisService, error_output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ai_module.ml", description=__doc__)
    parser.add_argument("--dataset", default="mimic_icu")
    parser.add_argument("--target")
    parser.add_argument("--folds", type=int)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--language", default="es", choices=("es", "en"))
    parser.add_argument("--json", action="store_true", help="Print the full JSON output.")
    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    payload = {"dataset": args.dataset, "target": args.target, "cv_folds": args.folds, "explain_with_llm": not args.no_llm}
    try:
        out = MLAnalysisService(settings).run({k: v for k, v in payload.items() if v is not None}, language=args.language)
    except MLAnalysisError as exc:
        print(json.dumps(error_output(exc), indent=2, ensure_ascii=False))
        return 1
    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
        return 0
    cols = ("model", "type", "accuracy", "precision", "recall", "f1", "roc_auc", "cv_mean", "cv_std", "overfitting_gap")
    scores = {e["key"]: e["final_score"] for e in out["ranking"]}
    print(f"\nRun {out['run_id']} | dataset {out['dataset']['dataset']} | target {out['dataset']['target']} | primary {out['dataset']['primary_metric']}")
    print(" | ".join(f"{c:>18}" if c == "model" else f"{c[:9]:>9}" for c in cols) + " | final_score")
    for row in out["comparison_table"]:
        cells = [f"{row['model']:>18}", f"{row['type']:>9}"] + [f"{row[c]:>9.4f}" if isinstance(row.get(c), float) else f"{str(row.get(c)):>9}" for c in cols[2:]]
        print(" | ".join(cells) + f" | {scores.get(row['key'], row['status'])}")
    print("\nRANKING:")
    for e in out["ranking"]:
        print(f"  {e['rank']}. {e['model']} ({e['type']}) final_score={e['final_score']}")
    print(f"\nMEJOR MODELO: {out['best_model']['name']} ({out['best_model']['type']}) final_score={out['best_model']['final_score']}")
    print(f"\n¿Por qué? {out['selection_reason']}")
    print(f"\nAnálisis ({out['analysis_source']}): {out['analysis']}")
    if out["warnings"]:
        print("\nAvisos:", out["warnings"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
