#!/usr/bin/env python3
"""Fetch a PDB chain sequence as one-letter amino-acid codes.

The default path uses SEQRES records from a PDB file because CovaGen's ESM
conditioning is sequence-based. If SEQRES is missing for the requested chain,
the script falls back to ordered ATOM residues.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    # Common modified residues accepted by PDB structures.
    "MSE": "M",
    "SEC": "U",
    "PYL": "O",
    "HID": "H",
    "HIE": "H",
    "HIP": "H",
    "ASH": "D",
    "GLH": "E",
    "CYX": "C",
    "CYM": "C",
    "UNK": "X",
}


def parse_pair(pair: str) -> tuple[str, str]:
    if ":" not in pair:
        raise ValueError(f"Expected PDB:CHAIN, got {pair!r}")
    pdb_id, chain_id = pair.split(":", 1)
    pdb_id = pdb_id.strip()
    chain_id = chain_id.strip()
    if not re.fullmatch(r"[A-Za-z0-9]{4}", pdb_id):
        raise ValueError(f"Expected four-character PDB id, got {pdb_id!r}")
    if not chain_id:
        raise ValueError("Chain id cannot be empty")
    if len(chain_id) != 1:
        raise ValueError(
            f"PDB-format chain ids must be one character; got {chain_id!r}. "
            "Use a local/mmCIF-aware extractor for multi-character auth chains."
        )
    return pdb_id.upper(), chain_id


def candidate_local_paths(pdb_id: str, pdb_dir: Path | None) -> list[Path]:
    if pdb_dir is None:
        return []
    lower = pdb_id.lower()
    upper = pdb_id.upper()
    return [
        pdb_dir / f"{upper}.pdb",
        pdb_dir / f"{lower}.pdb",
        pdb_dir / f"pdb{lower}.ent",
        pdb_dir / upper / f"{upper}.pdb",
        pdb_dir / lower / f"{lower}.pdb",
    ]


def get_pdb_file(pdb_id: str, cache_dir: Path, pdb_dir: Path | None) -> Path:
    for path in candidate_local_paths(pdb_id, pdb_dir):
        if path.exists():
            return path

    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{pdb_id}.pdb"
    if cached.exists() and cached.stat().st_size > 0:
        return cached

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not download {url}: {exc}") from exc

    if not data:
        raise RuntimeError(f"Downloaded empty PDB file from {url}")
    cached.write_bytes(data)
    return cached


def parse_seqres(lines: list[str], chain_id: str) -> str:
    residues: list[str] = []
    for line in lines:
        if not line.startswith("SEQRES"):
            continue
        if len(line) < 20 or line[11].strip() != chain_id:
            continue
        residues.extend(line[19:].split())
    return "".join(AA3_TO_1.get(res.upper(), "X") for res in residues)


def parse_atom_residues(lines: list[str], chain_id: str) -> str:
    residues: dict[tuple[int, str, str], str] = {}
    for line in lines:
        if not line.startswith("ATOM"):
            continue
        if len(line) < 54 or line[21].strip() != chain_id:
            continue
        resname = line[17:20].strip().upper()
        if resname in {"HOH", "WAT"}:
            continue
        try:
            resnum = int(line[22:26])
        except ValueError:
            continue
        icode = line[26].strip()
        key = (resnum, icode, resname)
        residues.setdefault(key, AA3_TO_1.get(resname, "X"))

    def sort_key(item: tuple[tuple[int, str, str], str]) -> tuple[int, int, str]:
        resnum, icode, _ = item[0]
        return (resnum, 0 if not icode else 1, icode)

    return "".join(one for _, one in sorted(residues.items(), key=sort_key))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pair", help="PDB:CHAIN pair, e.g. 1ghy:H")
    parser.add_argument(
        "--cache-dir",
        default=".pdb_cache",
        help="Directory used for downloaded PDB files.",
    )
    parser.add_argument(
        "--pdb-dir",
        default=None,
        help="Optional directory of local PDB files to prefer over RCSB downloads.",
    )
    parser.add_argument(
        "--source",
        choices=("seqres", "atom"),
        default="seqres",
        help="Use SEQRES full-chain sequence or ordered ATOM residues.",
    )
    parser.add_argument("--output", default="-", help="Write sequence here, or '-' for stdout.")
    args = parser.parse_args()

    try:
        pdb_id, chain_id = parse_pair(args.pair)
        pdb_dir = Path(args.pdb_dir).expanduser() if args.pdb_dir else None
        pdb_path = get_pdb_file(pdb_id, Path(args.cache_dir).expanduser(), pdb_dir)
        lines = pdb_path.read_text(errors="ignore").splitlines()
        if args.source == "seqres":
            sequence = parse_seqres(lines, chain_id)
            if not sequence:
                sequence = parse_atom_residues(lines, chain_id)
        else:
            sequence = parse_atom_residues(lines, chain_id)
        if not sequence:
            raise RuntimeError(f"No sequence found for {pdb_id}:{chain_id} in {pdb_path}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.output == "-":
        print(sequence)
    else:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(sequence + os.linesep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
