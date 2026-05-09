"""Command-line interface for HCNP antibody aggregation screening."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

from .locked_hcnp import add_locked_hcnp_scores
from .pdb_surface import compute_surface_descriptors, infer_two_largest_protein_chains
from .scoring import DEFAULT_HCNP_THRESHOLD, add_batch_scores


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HCNP antibody aggregation screening from one PDB, a PDB directory, or a manifest."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pdb", type=Path, help="Single antibody PDB file.")
    source.add_argument("--pdb-dir", type=Path, help="Directory containing PDB files.")
    source.add_argument("--manifest", type=Path, help="CSV manifest with PDB paths and optional chain IDs.")
    source.add_argument(
        "--feature-manifest",
        type=Path,
        help="CSV of precomputed HCNP/PyRosetta-CNT descriptors for the locked Method 01 score path.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--heavy-chain", default=None, help="Heavy-chain ID for --pdb or all files in --pdb-dir.")
    parser.add_argument("--light-chain", default=None, help="Light-chain ID for --pdb or all files in --pdb-dir.")
    parser.add_argument("--temperature-K", type=float, default=313.0, help="Temperature for kBT conversion.")
    parser.add_argument(
        "--hcnp-threshold",
        dest="hcnp_threshold",
        type=float,
        default=DEFAULT_HCNP_THRESHOLD,
        help="HCNP risk-score threshold used for thermo_call.",
    )
    parser.add_argument(
        "--max-curvature-points",
        type=int,
        default=1800,
        help="Maximum solvent-surface points used in local curvature fitting per structure.",
    )
    parser.add_argument(
        "--clean-barrier",
        action="store_true",
        help="Experimental: omit the original electrostatic barrier regularizer in --feature-manifest mode.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.feature_manifest is not None:
            features = pd.read_csv(args.feature_manifest)
            scored = add_locked_hcnp_scores(
                features,
                temperature_K=args.temperature_K,
                hcnp_threshold=args.hcnp_threshold,
                include_electrostatic_regularizer=not args.clean_barrier,
            )
            args.out.parent.mkdir(parents=True, exist_ok=True)
            scored.to_csv(args.out, index=False)
            print(f"Wrote {args.out}")
            return 0

        descriptor_rows = []
        for item in load_inputs(args):
            descriptors = compute_surface_descriptors(
                item["pdb_path"],
                item["heavy_chain"],
                item["light_chain"],
                max_curvature_points=args.max_curvature_points,
            )
            descriptors.update(
                {
                    "antibody_id": item.get("antibody_id") or Path(item["pdb_path"]).stem,
                    "pdb_id": item.get("pdb_id") or Path(item["pdb_path"]).stem,
                }
            )
            descriptor_rows.append(descriptors)

        scored = add_batch_scores(
            pd.DataFrame(descriptor_rows),
            temperature_K=args.temperature_K,
            hcnp_threshold=args.hcnp_threshold,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(args.out, index=False)
        print(f"Wrote {args.out}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def load_inputs(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.pdb is not None:
        heavy_chain, light_chain = _chain_pair_for_path(args.pdb, args.heavy_chain, args.light_chain)
        return [
            {
                "pdb_path": str(args.pdb),
                "antibody_id": args.pdb.stem,
                "pdb_id": args.pdb.stem,
                "heavy_chain": heavy_chain,
                "light_chain": light_chain,
            }
        ]

    if args.pdb_dir is not None:
        pdb_paths = sorted(args.pdb_dir.glob("*.pdb"))
        if not pdb_paths:
            raise ValueError(f"No .pdb files found in {args.pdb_dir}")
        items = []
        for pdb_path in pdb_paths:
            heavy_chain, light_chain = _chain_pair_for_path(pdb_path, args.heavy_chain, args.light_chain)
            items.append(
                {
                    "pdb_path": str(pdb_path),
                    "antibody_id": pdb_path.stem,
                    "pdb_id": pdb_path.stem,
                    "heavy_chain": heavy_chain,
                    "light_chain": light_chain,
                }
            )
        return items

    manifest = pd.read_csv(args.manifest)
    required = {"pdb_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    items = []
    for row in manifest.to_dict(orient="records"):
        pdb_path = Path(str(row["pdb_path"]))
        if not pdb_path.is_absolute():
            pdb_path = (args.manifest.parent / pdb_path).resolve()
        heavy_chain = _clean_optional(row.get("heavy_chain"))
        light_chain = _clean_optional(row.get("light_chain"))
        if not heavy_chain or not light_chain:
            heavy_chain, light_chain = infer_two_largest_protein_chains(pdb_path)
        items.append(
            {
                "pdb_path": str(pdb_path),
                "antibody_id": _clean_optional(row.get("antibody_id")) or pdb_path.stem,
                "pdb_id": _clean_optional(row.get("pdb_id")) or pdb_path.stem,
                "heavy_chain": heavy_chain,
                "light_chain": light_chain,
            }
        )
    return items


def _chain_pair_for_path(path: Path, heavy_chain: str | None, light_chain: str | None) -> tuple[str, str]:
    if heavy_chain and light_chain:
        return heavy_chain, light_chain
    return infer_two_largest_protein_chains(path)


def _clean_optional(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none"}:
        return None
    return text


if __name__ == "__main__":
    raise SystemExit(main())
