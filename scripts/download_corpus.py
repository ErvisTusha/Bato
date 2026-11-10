#!/usr/bin/env python3
"""Download a validated real corpus concurrently, or create explicit offline stress cases."""
import argparse
import json
import sys
from bato_tools.corpus import DEFAULT_BUDGET, audit_existing, load_catalog, prepare_corpus, select_assets, synthesize


def positive(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be positive")
    return parsed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output-dir", default="multi_corpus")
    parser.add_argument("-t", "--tier", choices=["standard", "large", "extreme"], default="standard")
    parser.add_argument("--target", action="append", default=[], help="Repeatable category or comma-separated categories")
    parser.add_argument("--manifest", action="append", default=[], help="Additional schema_version=1 asset catalog")
    parser.add_argument("--all", action="store_true", help="Include large suites within the byte budget")
    parser.add_argument("-f", "--force", action="store_true", help="Refresh managed assets after validating replacements")
    parser.add_argument("--synthetic", action="store_true", help="Strictly offline, separate deterministic stress cases")
    parser.add_argument("--jobs", type=positive, default=8)
    parser.add_argument("--per-host", type=positive, default=2)
    parser.add_argument("--extraction-jobs", type=positive, default=2)
    parser.add_argument("--max-bytes", type=positive, default=DEFAULT_BUDGET)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Show selection without downloads or writes")
    parser.add_argument("--audit-existing", action="store_true", help="Read-only structural/hash audit")
    parser.add_argument("--adopt-local", action="store_true", help="Index valid existing files with local provenance")
    args = parser.parse_args(argv)
    try:
        if args.audit_existing or args.adopt_local:
            report = audit_existing(args.output_dir, adopt=args.adopt_local)
        elif args.synthetic:
            if args.list or args.dry_run:
                report = {"origin": "synthetic", "categories": args.target or ["text", "structured", "binary", "images", "audio"]}
            else:
                report = synthesize(args.output_dir, args.target, args.max_bytes)
        else:
            assets = select_assets(load_catalog(args.manifest), args.target, args.tier, args.all)
            if args.list or args.dry_run:
                for item in assets:
                    print(f"{item['id']:<40} {item['category']:<12} <= {item['prepared_bytes'] / 2**20:8.2f} MiB  {item['destination']}")
                print(f"{len(assets)} assets; prepared upper bound {sum(item['prepared_bytes'] for item in assets):,} bytes; budget {args.max_bytes:,}")
                return 0
            report = prepare_corpus(args.output_dir, assets, jobs=args.jobs, max_bytes=args.max_bytes,
                                    force=args.force, per_host=args.per_host, extraction_jobs=args.extraction_jobs)
        print(json.dumps(report, indent=2))
        return 1 if report.get("failures") else 0
    except KeyboardInterrupt:
        print("Interrupted; resumable partials and completed checkpoints were preserved.", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
