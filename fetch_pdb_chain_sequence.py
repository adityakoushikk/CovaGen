#!/usr/bin/env python3
"""Fetch a PDB chain sequence as one-letter amino-acid codes.

The default path uses mmCIF polymer sequence records when possible, because
mmCIF preserves both canonical/label chain IDs and author chain IDs. If mmCIF
cannot be used, the script falls back to SEQRES and then ordered ATOM residues
from the legacy PDB file.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
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


def get_cif_file(pdb_id: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{pdb_id}.cif"
    if cached.exists() and cached.stat().st_size > 0:
        return cached

    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not download {url}: {exc}") from exc

    if not data:
        raise RuntimeError(f"Downloaded empty mmCIF file from {url}")
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


def iter_mmcif_loop_rows(lines: list[str], category: str):
    """Yield dict rows for a simple mmCIF loop category."""
    i = 0
    prefix = f"_{category}."
    while i < len(lines):
        if lines[i].strip() != "loop_":
            i += 1
            continue
        i += 1
        headers: list[str] = []
        while i < len(lines) and lines[i].lstrip().startswith("_"):
            headers.append(lines[i].strip().split()[0])
            i += 1
        if not headers or not any(header.startswith(prefix) for header in headers):
            while i < len(lines):
                stripped = lines[i].strip()
                if stripped == "loop_" or stripped == "#" or stripped.startswith("_"):
                    break
                i += 1
            continue

        tokens: list[str] = []
        while i < len(lines):
            stripped = lines[i].strip()
            if stripped == "loop_" or stripped == "#" or stripped.startswith("_"):
                break
            if stripped and not stripped.startswith(";"):
                tokens.extend(shlex.split(stripped, posix=True))
            i += 1

        width = len(headers)
        for start in range(0, len(tokens), width):
            row_tokens = tokens[start : start + width]
            if len(row_tokens) != width:
                continue
            yield dict(zip(headers, row_tokens))


def parse_mmcif_poly_sequences(lines: list[str]) -> dict[tuple[str, str], str]:
    """Return {(label_asym_id, auth_asym_id): sequence} from mmCIF."""
    rows = list(iter_mmcif_loop_rows(lines, "pdbx_poly_seq_scheme"))
    grouped: dict[tuple[str, str], list[tuple[int, str]]] = {}
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        label = row.get("_pdbx_poly_seq_scheme.asym_id", "").strip()
        auth = row.get("_pdbx_poly_seq_scheme.pdb_strand_id", "").strip()
        mon = row.get("_pdbx_poly_seq_scheme.mon_id", "").strip().upper()
        seq_id = row.get("_pdbx_poly_seq_scheme.seq_id", "").strip()
        if not label or label in {".", "?"} or not auth or auth in {".", "?"}:
            continue
        if mon in {".", "?", "HOH", "WAT"}:
            continue
        dedupe_key = (label, auth, seq_id, mon)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        try:
            seq_pos = int(seq_id)
        except ValueError:
            seq_pos = len(grouped.get((label, auth), [])) + 1
        grouped.setdefault((label, auth), []).append((seq_pos, AA3_TO_1.get(mon, "X")))

    sequences: dict[tuple[str, str], str] = {}
    for key, values in grouped.items():
        values.sort(key=lambda item: item[0])
        sequences[key] = "".join(one for _, one in values)
    return sequences


def resolve_mmcif_sequence(
    pdb_id: str,
    chain_id: str,
    cache_dir: Path,
    chain_id_kind: str,
) -> tuple[str, str]:
    cif_path = get_cif_file(pdb_id, cache_dir)
    sequences = parse_mmcif_poly_sequences(cif_path.read_text(errors="ignore").splitlines())
    matches: list[tuple[tuple[str, str], str]] = []
    for key, sequence in sequences.items():
        label, auth = key
        if chain_id_kind in {"auto", "label"} and chain_id == label:
            matches.append((key, sequence))
        if chain_id_kind in {"auto", "auth"} and chain_id == auth and key not in [m[0] for m in matches]:
            matches.append((key, sequence))

    if not matches:
        raise RuntimeError(
            f"No mmCIF polymer chain matched {chain_id!r} for {pdb_id}. "
            f"Available chains: {format_chain_mapping(sequences)}"
        )
    unique_keys = {key for key, _ in matches}
    if len(unique_keys) > 1:
        raise RuntimeError(
            f"Ambiguous chain id {chain_id!r} for {pdb_id}. Matches: "
            + ", ".join(f"label {label} auth {auth}" for label, auth in sorted(unique_keys))
            + ". Re-run with --chain-id-kind label or --chain-id-kind auth."
        )
    key, sequence = matches[0]
    label, auth = key
    return sequence, f"mmCIF label {label} auth {auth}"


def format_chain_mapping(sequences: dict[tuple[str, str], str]) -> str:
    if not sequences:
        return "<none>"
    return ", ".join(
        f"label {label} auth {auth} len {len(sequence)}"
        for (label, auth), sequence in sorted(sequences.items())
    )


def print_chain_report(pdb_id: str, cache_dir: Path, pdb_dir: Path | None) -> None:
    try:
        cif_path = get_cif_file(pdb_id, cache_dir)
        sequences = parse_mmcif_poly_sequences(cif_path.read_text(errors="ignore").splitlines())
        print(f"mmCIF chains for {pdb_id}:")
        for (label, auth), sequence in sorted(sequences.items()):
            print(f"  label={label} auth={auth} length={len(sequence)}")
    except Exception as exc:
        print(f"Could not read mmCIF chains: {exc}", file=sys.stderr)

    try:
        pdb_path = get_pdb_file(pdb_id, cache_dir, pdb_dir)
        lines = pdb_path.read_text(errors="ignore").splitlines()
        seqres_chains = sorted({line[11].strip() for line in lines if line.startswith("SEQRES") and len(line) > 11})
        atom_chains = sorted({line[21].strip() for line in lines if line.startswith("ATOM") and len(line) > 21})
        print(f"PDB SEQRES chain IDs: {', '.join(seqres_chains) if seqres_chains else '<none>'}")
        print(f"PDB ATOM chain IDs: {', '.join(atom_chains) if atom_chains else '<none>'}")
    except Exception as exc:
        print(f"Could not read PDB chains: {exc}", file=sys.stderr)


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
        choices=("auto", "mmcif", "seqres", "atom"),
        default="auto",
        help="Sequence source. auto uses mmCIF first, then PDB SEQRES, then PDB ATOM.",
    )
    parser.add_argument(
        "--chain-id-kind",
        choices=("auto", "label", "auth"),
        default="auth",
        help="How to interpret CHAIN for mmCIF. Default: auth. auto accepts either label or author chain IDs.",
    )
    parser.add_argument(
        "--list-chains",
        action="store_true",
        help="Print available mmCIF/PDB chain IDs for the PDB id and exit.",
    )
    parser.add_argument("--output", default="-", help="Write sequence here, or '-' for stdout.")
    args = parser.parse_args()

    try:
        pdb_id, chain_id = parse_pair(args.pair)
        pdb_dir = Path(args.pdb_dir).expanduser() if args.pdb_dir else None
        cache_dir = Path(args.cache_dir).expanduser()
        if args.list_chains:
            print_chain_report(pdb_id, cache_dir, pdb_dir)
            return 0

        sequence = ""
        source_desc = ""
        if args.source in {"auto", "mmcif"}:
            try:
                sequence, source_desc = resolve_mmcif_sequence(
                    pdb_id,
                    chain_id,
                    cache_dir,
                    args.chain_id_kind,
                )
            except Exception:
                if args.source == "mmcif":
                    raise
        if not sequence and args.source in {"auto", "seqres", "atom"}:
            pdb_path = get_pdb_file(pdb_id, cache_dir, pdb_dir)
            lines = pdb_path.read_text(errors="ignore").splitlines()
            if args.source in {"auto", "seqres"}:
                sequence = parse_seqres(lines, chain_id)
                source_desc = f"PDB SEQRES chain {chain_id}"
            if not sequence and args.source in {"auto", "atom"}:
                sequence = parse_atom_residues(lines, chain_id)
                source_desc = f"PDB ATOM chain {chain_id}"
        if not sequence:
            raise RuntimeError(f"No sequence found for {pdb_id}:{chain_id}")
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
