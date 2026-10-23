"""Command line: python -m pseudolabel <command>."""
import argparse
import json
import sys
from pathlib import Path

import yaml

from pseudolabel.data import load_universe
from pseudolabel.export import build_training_set, write_export
from pseudolabel.iteration import IterationConfig, finalize_iteration, plan_iteration, run_iteration
from pseudolabel.manual_test import import_human_labels, load_manual_test_csv, run_manual_test, select_manual_test
from pseudolabel.metrics import ThresholdRequired
from pseudolabel.profiles import load_profile
from pseudolabel.settings import load_settings
from pseudolabel.store import Store


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pseudolabel")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("select-manual-test", help="fix the manual-test set and its review queue")
    p.add_argument("--profile", required=True)
    p.add_argument("--pairs", required=True, help="pairs file: items for the review cards, and the pool to select from")
    p.add_argument("--from-csv", help="use an already fixed set (tools/manual_labeling output) instead of selecting")
    p.add_argument("--per-class", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("import-human", help="import human labels a_id,b_id,class[,comment]")
    p.add_argument("--profile", required=True)
    p.add_argument("--csv", required=True)
    p = sub.add_parser("run-manual-test", help="label the manual-test pairs and compare with humans")
    p.add_argument("--profile", required=True)
    p.add_argument("--pairs", required=True)
    p.add_argument("--threshold", type=int)
    p = sub.add_parser("iterate", help="plan, run and finalize one iteration from a YAML config")
    p.add_argument("--config", required=True)
    p = sub.add_parser("export", help="export the training set")
    p.add_argument("--profile", required=True)
    p.add_argument("--threshold", type=int)
    p.add_argument("--iteration", type=int, nargs="*")
    p.add_argument("--include-legacy", action="store_true")
    p.add_argument("--out", required=True)
    p = sub.add_parser("ablate", help="run one labeler variant of an ablation config over its eval set")
    p.add_argument("--config", required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--limit", type=int, help="only the first N eval pairs (smoke test)")
    p.add_argument("--retry-failed", action="store_true", help="label again the pairs that got no class")
    p = sub.add_parser("segment", help="cut the item out of every eval photo of an ablation config")
    p.add_argument("--config", required=True)
    p.add_argument("--method", required=True, choices=["sam3", "birefnet"])
    p.add_argument("--out", required=True, help="JSON {photo: cut-out} for a variant's image_map")
    p = sub.add_parser("ablate-report", help="compare the variants of an ablation config; writes report.md")
    p.add_argument("--config", required=True)
    p.add_argument("--baseline", default="base")
    return parser


def _labeler(settings, store, profile, version, model: str | None = None):
    from pseudolabel.labeler import Labeler
    from pseudolabel.vlm import OpenAIVLM

    vlm = OpenAIVLM(settings.vllm_base_url, model or settings.vllm_model, settings.vllm_api_key)
    return Labeler(vlm, store, profile, version)


def _ablate(args, settings) -> int:
    from pseudolabel.experiments import load_config, load_eval, report, run_variant, stderr_progress

    cfg, variants = load_config(Path(args.config))
    out_dir = Path(cfg["out_dir"])
    if args.cmd == "ablate-report":
        text = report(out_dir, args.baseline, cfg["threshold"], order=list(variants))
        (out_dir / "report.md").write_text(text, encoding="utf-8")
        print(text)
        return 0
    profile = load_profile(settings.profiles_dir / f"{cfg['profile']}.yaml")
    universe, human = load_eval(Path(cfg["eval_pairs"]), Path(cfg["eval_labels"]), profile)
    if args.cmd == "segment":
        from pseudolabel import segment

        photos = [it.image for it in [*universe.items_a.values(), *universe.items_b.values()]]
        out = Path(args.out)
        seg = segment.make_segmenter(args.method, prompt=profile.object_prompt or profile.category_label,
                                     device=settings.encoder_device)
        mapping, stats = segment.segment_images(seg, photos, out.parent / args.method)
        out.write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(stats))
        return 0
    store = Store(Path(cfg["db"])) if cfg.get("db") else Store(settings.db_path)
    m = run_variant(variants[args.variant], profile, universe, human, store, out_dir, cfg["threshold"],
                    limit=args.limit, progress=stderr_progress(args.variant), retry_failed=args.retry_failed)
    print(json.dumps({k: v for k, v in m.items() if k != "confusion"}, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings()
    if args.cmd in ("ablate", "ablate-report", "segment"):
        return _ablate(args, settings)
    store = Store(settings.db_path)
    try:
        if args.cmd == "iterate":
            cfg = IterationConfig(**yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))
            cfg.labeler_model = cfg.labeler_model or settings.vllm_model
            profile = load_profile(settings.profiles_dir / f"{cfg.profile_name}.yaml")
            from pseudolabel.encoder import SigLIP2Encoder

            encoder = SigLIP2Encoder(cfg.encoder_checkpoint, device=settings.encoder_device)
            it = plan_iteration(cfg, store, profile, load_universe(profile, Path(cfg.pairs_path)), encoder,
                                settings.work_dir / "embeddings")
            version = store.get_iteration(it)["profile_version"]
            labeler = _labeler(settings, store, store.profile_at(profile.name, version), version, cfg.labeler_model)
            run_iteration(it, store, labeler, settings.vlm_concurrency,
                          progress=lambda d, t: print(f"\r{d}/{t}", end="", file=sys.stderr))
            print(json.dumps({"iteration": it, **finalize_iteration(it, store)}, ensure_ascii=False, indent=2))
            return 0
        profile = load_profile(settings.profiles_dir / f"{args.profile}.yaml")
        if args.cmd == "select-manual-test":
            universe = load_universe(profile, Path(args.pairs))
            store.upsert_items(profile.name, list(universe.items_a.values()) + list(universe.items_b.values()))
            if args.from_csv:
                rows, shortfall = load_manual_test_csv(Path(args.from_csv)), {}
            else:
                rows, shortfall = select_manual_test(universe, args.per_class, args.seed)
            store.set_manual_test(profile.name, rows)
            store.queue_replace(f"manual_test:{profile.name}", profile.name, [(a, b) for a, b, _ in rows])
            print(f"manual test: {len(rows)} pairs, shortfall {shortfall}")
        elif args.cmd == "import-human":
            print(f"imported {import_human_labels(store, profile.name, Path(args.csv))} human labels")
        elif args.cmd == "run-manual-test":
            universe = load_universe(profile, Path(args.pairs))
            labeler = _labeler(settings, store, profile, store.register_profile_version(profile))
            print(json.dumps(run_manual_test(store, labeler, universe, profile.name, args.threshold,
                                             settings.vlm_concurrency), ensure_ascii=False, indent=2))
        elif args.cmd == "export":
            df = build_training_set(store, profile.name, args.threshold, iteration_ids=args.iteration or None,
                                    include_legacy=args.include_legacy)
            pq, csv = write_export(df, Path(args.out), f"{profile.name}_threshold{args.threshold}")
            print(f"exported {len(df)} rows to {pq} and {csv}")
        return 0
    except (ThresholdRequired, ValueError) as e:
        print(str(e), file=sys.stderr)
        return 2
