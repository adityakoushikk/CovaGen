"""
Attach a fixed warhead from warheads.yaml to CovaGen-generated SMILES.

This is the 2D analogue of the MEDSAGE fragment-connect logic:

1. Add explicit hydrogens to the warhead and CovaGen-generated molecule.
2. Use the YAML attachment_topology to find allowed heavy atoms on the warhead.
3. Convert those allowed heavy atoms into allowed warhead H atoms.
4. Enumerate symmetry-unique H atoms on requested CovaGen-molecule atom types.
5. For each H-H pair, bond their heavy neighbors and delete both H atoms.
6. Keep products that sanitize and still match the YAML QC SMARTS.

Example:
    python scripts/attach_warhead.py \
        --input ./samples/my_run_valid.pkl \
        --warheads_yaml /Users/adityakoushik/Downloads/warheads.yaml \
        --warhead_id 3723 \
        --output ./samples/my_run_acrylamide.csv

Random docking enumerations with LigPrep-compatible titles:
    python scripts/attach_warhead.py \
        --input ./samples/my_run_valid.pkl \
        --warheads_yaml ./configs/warheads.yaml \
        --warhead_id 3723 \
        --enumerations_per_input 50 \
        --random_seed 114514 \
        --output ./samples/my_run_acrylamide.smi

Keep a random sample of classifier-generated molecules that retain the warhead:
    python scripts/attach_warhead.py \
        --input ./samples/classifier_output.smi \
        --warheads_yaml ./configs/warheads.yaml \
        --warhead_id 3723 \
        --with_classifier \
        --classifier_keep_n 1000 \
        --random_seed 114514 \
        --output ./samples/classifier_output_warhead_sample.smi
"""

import argparse
import csv
import os
import pickle
import random
import sys

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover - user-facing dependency check
    raise SystemExit("networkx is required. Install the repo requirements or install networkx.") from exc

try:
    from rdkit import Chem
except ImportError as exc:  # pragma: no cover - user-facing dependency check
    raise SystemExit("RDKit is required. Install the repo requirements or install rdkit.") from exc

try:
    import yaml
except ImportError as exc:  # pragma: no cover - user-facing dependency check
    raise SystemExit("PyYAML is required. Install it with `pip install pyyaml`.") from exc


PERMISSIVE_TOPOLOGIES = {"permissive", "none", "allow_all", "", None}


def load_smiles(path):
    return [record["smiles"] for record in load_smiles_records(path)]


def load_smiles_records(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in {".pkl", ".pickle"}:
        with open(path, "rb") as handle:
            data = pickle.load(handle)
        records = []
        for idx, smiles in enumerate(flatten_smiles(data), start=1):
            if smiles:
                records.append({"smiles": smiles, "name": default_mol_name(idx)})
        return records

    if ext == ".csv":
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames:
                for candidate in ("smiles", "product_smiles", "parent_smiles"):
                    if candidate in reader.fieldnames:
                        smiles_col = candidate
                        break
                else:
                    smiles_col = reader.fieldnames[0]
                name_col = first_existing_field(
                    reader.fieldnames,
                    ("molecule_name", "name", "id", "ID", "title", "parent_name", "product_name"),
                )
                records = []
                for idx, row in enumerate(reader, start=1):
                    smiles = row.get(smiles_col, "").strip()
                    if not smiles:
                        continue
                    name = row.get(name_col, "").strip() if name_col else ""
                    records.append({"smiles": smiles, "name": sanitize_name(name or default_mol_name(idx))})
                return records
        return []

    records = []
    with open(path) as handle:
        for idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            name = parts[1] if len(parts) > 1 else default_mol_name(idx)
            records.append({"smiles": parts[0], "name": sanitize_name(name)})
    return records


def first_existing_field(fieldnames, candidates):
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    return None


def default_mol_name(index):
    return f"mol_{index:06d}"


def sanitize_name(name):
    return "_".join(str(name).split())


def parent_id(input_index):
    """Stable ID for one row/molecule in the CovaGen input file."""
    return f"covagen_{input_index:06d}"


def product_name(covagen_parent_id, warhead_id, enumeration_index):
    """LigPrep title carrying parent and per-parent enumeration identity."""
    return sanitize_name(
        f"{covagen_parent_id}__wh{warhead_id}__enum{enumeration_index:03d}"
    )


def flatten_smiles(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from flatten_smiles(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from flatten_smiles(item)


def load_warhead(path, warhead_id):
    with open(path) as handle:
        data = yaml.safe_load(handle) or {}
    warheads = data.get("warheads", {})
    key = str(warhead_id)
    if key not in warheads:
        available = ", ".join(sorted(warheads))
        raise ValueError(f"Warhead ID {key!r} not found. Available IDs: {available}")
    return warheads[key]


def mol_from_smiles(smiles, explicit_hs=False):
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if explicit_hs:
        mol = Chem.AddHs(mol)
    return mol


def rdkit_mol_to_nx(mol):
    graph = nx.Graph()
    for atom in mol.GetAtoms():
        graph.add_node(
            atom.GetIdx(),
            element=atom.GetSymbol(),
            atomic_num=atom.GetAtomicNum(),
            formal_charge=atom.GetFormalCharge(),
            aromatic=bool(atom.GetIsAromatic()),
        )
    for bond in mol.GetBonds():
        order = 1.5 if bond.GetIsAromatic() else float(bond.GetBondTypeAsDouble())
        graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), order=order)
    return graph


def get_hydrogens_graph(graph):
    return [n for n, data in graph.nodes(data=True) if data.get("element") == "H"]


def get_a_neighbor_id(graph, node_id):
    neighbors = list(graph.neighbors(node_id))
    if len(neighbors) != 1:
        return None
    return neighbors[0]


def node_match(left, right):
    return (
        left.get("element") == right.get("element")
        and left.get("formal_charge", 0) == right.get("formal_charge", 0)
        and bool(left.get("aromatic", False)) == bool(right.get("aromatic", False))
    )


def edge_match(left, right):
    return abs(float(left.get("order", 0.0)) - float(right.get("order", 0.0))) < 1e-6


def is_isomorphic(left, right):
    return nx.is_isomorphic(left, right, node_match=node_match, edge_match=edge_match)


def remaining_unique_attachments(graph, candidate_hydrogens, used_attachments=None):
    """
    Return one representative from each chemically equivalent candidate H class.

    This mirrors MEDSAGE's trick: mark one candidate H as a fake element, then
    compare graph isomorphism. Equivalent hydrogens produce isomorphic marked
    graphs and are collapsed.
    """
    used_attachments = set(used_attachments or [])
    unique = []
    graph_used = graph.copy()
    for used_id in used_attachments:
        if used_id in graph_used:
            graph_used.nodes[used_id]["element"] = str(used_id)

    for h_id in candidate_hydrogens:
        if h_id in used_attachments:
            continue
        graph_test = graph_used.copy()
        graph_test.nodes[h_id]["element"] = "XX"

        if not unique or not any(is_isomorphic(item["graph"], graph_test) for item in unique):
            unique.append({"graph": graph_test, "id": h_id})

    unique_ids = [item["id"] for item in unique]
    attached_elements = []
    for h_id in unique_ids:
        heavy = get_a_neighbor_id(graph, h_id)
        attached_elements.append(graph.nodes[heavy]["element"] if heavy is not None else "")
    return unique_ids, attached_elements


def _bond_order(data):
    try:
        return float(data.get("order", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _is_cc_double(graph, left, right):
    return (
        graph.nodes[left]["element"] == "C"
        and graph.nodes[right]["element"] == "C"
        and _bond_order(graph[left][right]) >= 1.6
    )


def _cc_is_triple(graph, left, right):
    return (
        graph.nodes[left]["element"] == "C"
        and graph.nodes[right]["element"] == "C"
        and _bond_order(graph[left][right]) >= 2.5
    )


def _carbonyl_carbons_all(graph):
    out = []
    for left, right, data in graph.edges(data=True):
        if _bond_order(data) < 1.6:
            continue
        left_el = graph.nodes[left]["element"]
        right_el = graph.nodes[right]["element"]
        if {left_el, right_el} == {"C", "O"}:
            out.append(left if left_el == "C" else right)
    return out


def _carbonyl_carbon(graph):
    carbons = _carbonyl_carbons_all(graph)
    return carbons[0] if carbons else None


def _has_h_neighbor(graph, node):
    return any(graph.nodes[n]["element"] == "H" for n in graph.neighbors(node))


def allowed_heavy_halo_acetamide_n(graph):
    """
    [H]N([H])C(CCl)=O / CBr: primary-only amide N-H, not alpha carbon.
    """
    for carbonyl in _carbonyl_carbons_all(graph):
        nitrogens = [n for n in graph.neighbors(carbonyl) if graph.nodes[n]["element"] == "N"]
        if not nitrogens:
            continue
        halomethyl = False
        for neighbor_c in graph.neighbors(carbonyl):
            if graph.nodes[neighbor_c]["element"] != "C":
                continue
            if any(graph.nodes[x]["element"] in {"Cl", "Br", "I", "F"} for x in graph.neighbors(neighbor_c)):
                halomethyl = True
                break
        if not halomethyl:
            continue
        with_h = [n for n in nitrogens if _has_h_neighbor(graph, n)]
        if with_h:
            return set(with_h)
    return None


def allowed_heavy_acrylamide_amide_n_mono(graph):
    """
    C=CC(N([H])[H])=O: primary-only amide N-H on alpha,beta-unsaturated amide.
    """
    for carbonyl in _carbonyl_carbons_all(graph):
        nitrogens = [n for n in graph.neighbors(carbonyl) if graph.nodes[n]["element"] == "N"]
        if not nitrogens:
            continue
        enone = False
        for neighbor_c in graph.neighbors(carbonyl):
            if graph.nodes[neighbor_c]["element"] != "C":
                continue
            for other in graph.neighbors(neighbor_c):
                if other == carbonyl:
                    continue
                if graph.nodes[other]["element"] == "C" and _is_cc_double(graph, neighbor_c, other):
                    enone = True
                    break
            if enone:
                break
        if not enone:
            continue
        with_h = [n for n in nitrogens if _has_h_neighbor(graph, n)]
        if with_h:
            return set(with_h)
    return None


def allowed_heavy_acrylester_hydroxyl_o(graph):
    """
    O=C(C=C)O[H]: primary-only hydroxyl O-H on alpha,beta-unsaturated ester.
    """
    for carbonyl in _carbonyl_carbons_all(graph):
        hydroxyl_o = None
        for neighbor in graph.neighbors(carbonyl):
            if graph.nodes[neighbor]["element"] == "O" and _bond_order(graph[carbonyl][neighbor]) < 1.6:
                if _has_h_neighbor(graph, neighbor):
                    hydroxyl_o = neighbor
                    break
        if hydroxyl_o is None:
            continue
        enone = False
        for neighbor_c in graph.neighbors(carbonyl):
            if graph.nodes[neighbor_c]["element"] != "C" or neighbor_c == hydroxyl_o:
                continue
            for other in graph.neighbors(neighbor_c):
                if other == carbonyl:
                    continue
                if graph.nodes[other]["element"] == "C" and _is_cc_double(graph, neighbor_c, other):
                    enone = True
                    break
            if enone:
                break
        if enone:
            return {hydroxyl_o}
    return None


def allowed_heavy_vinyl_sulfone_sulfur(graph):
    """
    [H]S(=O)(C=C)=O: sulfur bears the replaceable H.
    """
    for node, data in graph.nodes(data=True):
        if data.get("element") != "S" or not _has_h_neighbor(graph, node):
            continue
        double_o = sum(
            1
            for neighbor in graph.neighbors(node)
            if graph.nodes[neighbor]["element"] == "O" and _bond_order(graph[node][neighbor]) >= 1.6
        )
        if double_o < 2:
            continue
        vinyl = False
        for neighbor_c in graph.neighbors(node):
            if graph.nodes[neighbor_c]["element"] != "C":
                continue
            for other in graph.neighbors(neighbor_c):
                if other == node:
                    continue
                if graph.nodes[other]["element"] == "C" and _is_cc_double(graph, neighbor_c, other):
                    vinyl = True
                    break
            if vinyl:
                break
        if vinyl:
            return {node}
    return None


def allowed_heavy_alpha_ketoamide_aldehyde_c(graph):
    """
    [H]C(C(N([H])[H])=O)=O: formyl C-H, not amide N or amide C=O.
    """
    for left, right, data in graph.edges(data=True):
        if _bond_order(data) < 1.6:
            continue
        if {graph.nodes[left]["element"], graph.nodes[right]["element"]} != {"C", "O"}:
            continue
        carbon = left if graph.nodes[left]["element"] == "C" else right
        oxygen = right if carbon == left else left
        others = [n for n in graph.neighbors(carbon) if n != oxygen]
        if any(graph.nodes[n]["element"] == "N" for n in others):
            continue
        if not _has_h_neighbor(graph, carbon):
            continue
        for other_c in others:
            if graph.nodes[other_c]["element"] != "C":
                continue
            if not any(graph.nodes[n]["element"] == "N" for n in graph.neighbors(other_c)):
                continue
            if any(
                graph.nodes[n]["element"] == "O" and _bond_order(graph[other_c][n]) >= 1.6
                for n in graph.neighbors(other_c)
            ):
                return {carbon}
    return None


def allowed_heavy_sulfonyl_fluoride_sulfur(graph):
    """
    [H]S(=O)(F)=O: sulfur bears the replaceable H.
    """
    for node, data in graph.nodes(data=True):
        if data.get("element") != "S" or not _has_h_neighbor(graph, node):
            continue
        if not any(graph.nodes[n]["element"] == "F" for n in graph.neighbors(node)):
            continue
        double_o = sum(
            1
            for neighbor in graph.neighbors(node)
            if graph.nodes[neighbor]["element"] == "O" and _bond_order(graph[node][neighbor]) >= 1.6
        )
        if double_o >= 2:
            return {node}
    return None


def allowed_heavy_ketoalkynyl_aldehyde_c(graph):
    """
    [H]C(C#C)=O: aldehyde carbon only.
    """
    for left, right, data in graph.edges(data=True):
        if _bond_order(data) < 1.6:
            continue
        if {graph.nodes[left]["element"], graph.nodes[right]["element"]} != {"C", "O"}:
            continue
        carbon = left if graph.nodes[left]["element"] == "C" else right
        oxygen = right if carbon == left else left
        others = [n for n in graph.neighbors(carbon) if n != oxygen]
        if any(graph.nodes[n]["element"] == "N" for n in others):
            continue
        if not _has_h_neighbor(graph, carbon):
            continue
        for other_c in others:
            if graph.nodes[other_c]["element"] != "C":
                continue
            for terminal in graph.neighbors(other_c):
                if terminal == carbon:
                    continue
                if _cc_is_triple(graph, other_c, terminal):
                    return {carbon}
    return None


def allowed_heavy_maleimide_n_mono(graph):
    """
    Maleimide: imide N-H in the 5-membered ring.
    """
    for cycle in nx.cycle_basis(graph):
        if len(cycle) != 5:
            continue
        for node in cycle:
            if graph.nodes[node]["element"] != "N":
                continue
            ring_carbons = [n for n in graph.neighbors(node) if graph.nodes[n]["element"] == "C" and n in cycle]
            if len(ring_carbons) != 2:
                continue
            if not all(
                any(
                    graph.nodes[o]["element"] == "O" and _bond_order(graph[c][o]) >= 1.6
                    for o in graph.neighbors(c)
                )
                for c in ring_carbons
            ):
                continue
            if _has_h_neighbor(graph, node):
                return {node}
    return None


def allowed_heavy_nitrile_carbon(graph):
    """
    [H]C#N: C-H on the nitrile carbon.
    """
    for left, right, data in graph.edges(data=True):
        if _bond_order(data) < 2.0:
            continue
        if {graph.nodes[left]["element"], graph.nodes[right]["element"]} != {"C", "N"}:
            continue
        nitrile_c = left if graph.nodes[left]["element"] == "C" else right
        if _has_h_neighbor(graph, nitrile_c):
            return {nitrile_c}
    return None


def allowed_heavy_vinylnitrile_terminal_vinyl_c(graph):
    """
    N#CC=C: terminal alkene carbon farthest from C#N.
    """
    nitrile_c = None
    nitrile_n = None
    for left, right, data in graph.edges(data=True):
        if _bond_order(data) < 2.0:
            continue
        if {graph.nodes[left]["element"], graph.nodes[right]["element"]} != {"C", "N"}:
            continue
        nitrile_c = left if graph.nodes[left]["element"] == "C" else right
        nitrile_n = right if nitrile_c == left else left
        break
    if nitrile_c is None:
        return None

    internal = None
    for neighbor in graph.neighbors(nitrile_c):
        if neighbor == nitrile_n:
            continue
        if graph.nodes[neighbor]["element"] == "C":
            internal = neighbor
            break
    if internal is None:
        return None

    for terminal in graph.neighbors(internal):
        if terminal == nitrile_c:
            continue
        if graph.nodes[terminal]["element"] == "C" and _is_cc_double(graph, internal, terminal):
            if _has_h_neighbor(graph, terminal):
                return {terminal}
    return None


def allowed_heavy_cyanamide_primary_n(graph):
    """
    H2N-C#N: amine nitrogen only.
    """
    for left, right, data in graph.edges(data=True):
        if _bond_order(data) < 2.0:
            continue
        if {graph.nodes[left]["element"], graph.nodes[right]["element"]} != {"C", "N"}:
            continue
        nitrile_c = left if graph.nodes[left]["element"] == "C" else right
        nitrile_n = right if nitrile_c == left else left
        for neighbor in graph.neighbors(nitrile_c):
            if neighbor == nitrile_n:
                continue
            if graph.nodes[neighbor]["element"] == "N" and _has_h_neighbor(graph, neighbor):
                return {neighbor}
    return None


def allowed_heavy_cyanamide_amine_n_mono(graph):
    return allowed_heavy_cyanamide_primary_n(graph)


def allowed_heavy_carbamate_hydroxyl_o(graph):
    """
    [H]N([H])C(O[H])=O: hydroxyl O-H, not amide N-H.
    """
    for carbonyl in _carbonyl_carbons_all(graph):
        if not any(graph.nodes[n]["element"] == "N" for n in graph.neighbors(carbonyl)):
            continue
        for neighbor in graph.neighbors(carbonyl):
            if graph.nodes[neighbor]["element"] != "O":
                continue
            if _bond_order(graph[carbonyl][neighbor]) >= 1.6:
                continue
            if _has_h_neighbor(graph, neighbor):
                return {neighbor}
    return None


def allowed_heavy_cyanoacrylamide_terminal_vinyl_c(graph):
    """
    NC(=O)C(C#N)=C: terminal =CH2 carbon, not amide nitrogen.
    """
    for carbonyl in _carbonyl_carbons_all(graph):
        if not any(graph.nodes[n]["element"] == "N" for n in graph.neighbors(carbonyl)):
            continue
        alpha = None
        for neighbor in graph.neighbors(carbonyl):
            if graph.nodes[neighbor]["element"] == "C" and _bond_order(graph[carbonyl][neighbor]) < 1.6:
                alpha = neighbor
                break
        if alpha is None:
            continue

        nitrile_branch = None
        for branch in graph.neighbors(alpha):
            if branch == carbonyl or graph.nodes[branch]["element"] != "C":
                continue
            for maybe_n in graph.neighbors(branch):
                if graph.nodes[maybe_n]["element"] == "N" and _bond_order(graph[branch][maybe_n]) >= 2.0:
                    nitrile_branch = branch
                    break
        if nitrile_branch is None:
            continue

        for terminal in graph.neighbors(alpha):
            if terminal in {carbonyl, nitrile_branch}:
                continue
            if graph.nodes[terminal]["element"] == "C" and _is_cc_double(graph, alpha, terminal):
                if _has_h_neighbor(graph, terminal):
                    return {terminal}
    return None


def allowed_heavy_epoxide_ring_carbons(graph):
    """
    Epoxide: either ring carbon, not ether oxygen.
    """
    for oxygen, data in graph.nodes(data=True):
        if data.get("element") != "O":
            continue
        carbons = [n for n in graph.neighbors(oxygen) if graph.nodes[n]["element"] == "C"]
        if len(carbons) != 2:
            continue
        first, second = carbons
        if graph.has_edge(first, second):
            allowed = {first, second}
            with_h = {n for n in allowed if _has_h_neighbor(graph, n)}
            return with_h or allowed
    return None


TOPOLOGY_HANDLERS = {
    "acrylamide_amide_n_mono": allowed_heavy_acrylamide_amide_n_mono,
    "haloacetamide_amide_n_mono": allowed_heavy_halo_acetamide_n,
    "acrylester_hydroxyl_o": allowed_heavy_acrylester_hydroxyl_o,
    "vinyl_sulfone_sulfur": allowed_heavy_vinyl_sulfone_sulfur,
    "alpha_ketoamide_aldehyde_c": allowed_heavy_alpha_ketoamide_aldehyde_c,
    "sulfonyl_fluoride_sulfur": allowed_heavy_sulfonyl_fluoride_sulfur,
    "ketoalkynyl_aldehyde_c": allowed_heavy_ketoalkynyl_aldehyde_c,
    "maleimide_n_mono": allowed_heavy_maleimide_n_mono,
    "epoxide_ring_carbons": allowed_heavy_epoxide_ring_carbons,
    "nitrile_carbon": allowed_heavy_nitrile_carbon,
    "vinylnitrile_terminal_vinyl_c": allowed_heavy_vinylnitrile_terminal_vinyl_c,
    "cyanamide_amine_n_mono": allowed_heavy_cyanamide_amine_n_mono,
    "carbamate_hydroxyl_o": allowed_heavy_carbamate_hydroxyl_o,
    "cyanoacrylamide_terminal_vinyl_c": allowed_heavy_cyanoacrylamide_terminal_vinyl_c,
}


def list_registered_attachment_topologies():
    return sorted(set(TOPOLOGY_HANDLERS) | {str(x) for x in PERMISSIVE_TOPOLOGIES if x})


def topology_allowed_warhead_hydrogens(warhead_graph, topology):
    if topology in PERMISSIVE_TOPOLOGIES:
        candidates = get_hydrogens_graph(warhead_graph)
        return remaining_unique_attachments(warhead_graph, candidates)

    if topology not in TOPOLOGY_HANDLERS:
        registered = ", ".join(list_registered_attachment_topologies())
        raise ValueError(f"Unsupported attachment_topology {topology!r}. Registered topologies: {registered}")

    allowed_heavy = TOPOLOGY_HANDLERS[topology](warhead_graph)
    if not allowed_heavy:
        return [], []

    candidates = []
    for h_id in get_hydrogens_graph(warhead_graph):
        heavy = get_a_neighbor_id(warhead_graph, h_id)
        if heavy in allowed_heavy:
            candidates.append(h_id)
    return remaining_unique_attachments(warhead_graph, candidates)


def scaffold_attachment_hydrogens(scaffold_graph, allowed_atomic_nums):
    candidates = []
    for h_id in get_hydrogens_graph(scaffold_graph):
        heavy = get_a_neighbor_id(scaffold_graph, h_id)
        if heavy is None:
            continue
        if scaffold_graph.nodes[heavy].get("atomic_num") in allowed_atomic_nums:
            candidates.append(h_id)
    return remaining_unique_attachments(scaffold_graph, candidates)


def scaffold_attachment_hydrogens_from_mol(scaffold_mol, allowed_atomic_nums):
    """
    Fast scaffold-side equivalent-site collapse using RDKit symmetry ranks.

    The older graph-isomorphism path is exact but slow for generated molecules
    with many H atoms. Here we dedupe by the symmetry rank of each H's heavy
    atom and keep one representative H per equivalent heavy-atom site.
    """
    ranks = list(Chem.CanonicalRankAtoms(scaffold_mol, breakTies=False))
    seen_sites = set()
    unique_hydrogens = []
    attached_elements = []

    for atom in scaffold_mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        neighbors = list(atom.GetNeighbors())
        if len(neighbors) != 1:
            continue
        heavy = neighbors[0]
        if heavy.GetAtomicNum() not in allowed_atomic_nums:
            continue

        site_key = (heavy.GetAtomicNum(), ranks[heavy.GetIdx()])
        if site_key in seen_sites:
            continue
        seen_sites.add(site_key)
        unique_hydrogens.append(atom.GetIdx())
        attached_elements.append(heavy.GetSymbol())

    return unique_hydrogens, attached_elements


def select_scaffold_attachment_sites(hydrogens, elements, max_sites, selection, rng):
    """Select symmetry-unique scaffold sites, optionally in seeded random order."""
    sites = list(zip(hydrogens, elements))
    if selection == "random":
        rng.shuffle(sites)
    if max_sites:
        sites = sites[:max_sites]
    return sites


def random_sample_records(records, keep_n, rng):
    """Return at most keep_n randomly selected records, preserving input order."""
    if not keep_n or len(records) <= keep_n:
        return records
    selected_indices = sorted(rng.sample(range(len(records)), keep_n))
    return [records[index] for index in selected_indices]


def heavy_neighbor_in_mol(mol, hydrogen_idx):
    atom = mol.GetAtomWithIdx(hydrogen_idx)
    if atom.GetAtomicNum() != 1:
        return None
    neighbors = list(atom.GetNeighbors())
    if len(neighbors) != 1:
        return None
    return neighbors[0].GetIdx()


def attach_by_hydrogen_pair(scaffold_mol, scaffold_h_idx, warhead_mol, warhead_h_idx):
    scaffold_heavy_idx = heavy_neighbor_in_mol(scaffold_mol, scaffold_h_idx)
    warhead_heavy_idx = heavy_neighbor_in_mol(warhead_mol, warhead_h_idx)
    if scaffold_heavy_idx is None or warhead_heavy_idx is None:
        return None

    offset = scaffold_mol.GetNumAtoms()
    combined = Chem.CombineMols(scaffold_mol, warhead_mol)
    rw = Chem.RWMol(combined)
    rw.AddBond(scaffold_heavy_idx, offset + warhead_heavy_idx, Chem.BondType.SINGLE)

    for atom_idx in sorted([scaffold_h_idx, offset + warhead_h_idx], reverse=True):
        rw.RemoveAtom(atom_idx)

    product = rw.GetMol()
    Chem.SanitizeMol(product)
    product = Chem.RemoveHs(product, sanitize=True)
    Chem.SanitizeMol(product)
    return product


def compile_qc_smarts(smarts_list):
    queries = []
    for smarts in smarts_list:
        query = Chem.MolFromSmarts(smarts)
        if query is None:
            raise ValueError(f"Could not parse QC SMARTS: {smarts!r}")
        queries.append(query)
    return queries


def passes_qc(product_mol, qc_queries):
    for query in qc_queries:
        if not product_mol.HasSubstructMatch(query):
            return False
    return True


OUTPUT_FIELDS = [
    "product_smiles",
    "product_name",
    "parent_id",
    "parent_index",
    "enumeration_index",
    "parent_smiles",
    "parent_name",
    "warhead_id",
    "warhead_name",
    "warhead_class",
    "attachment_topology",
    "scaffold_h_idx",
    "scaffold_heavy_idx",
    "scaffold_heavy_element",
    "warhead_h_idx",
    "warhead_heavy_idx",
    "warhead_heavy_element",
]


def write_records_csv(records, output_path):
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def default_metadata_path(output_path):
    ext = os.path.splitext(output_path)[1].lower()
    if ext not in {".smi", ".txt"}:
        return None
    stem, _ = os.path.splitext(output_path)
    return stem + "_metadata.csv"


def output_records(records, output_path, as_records=False, metadata_path=None):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".csv":
        write_records_csv(records, output_path)
    elif ext in {".pkl", ".pickle"}:
        payload = records if as_records else [record["product_smiles"] for record in records]
        with open(output_path, "wb") as handle:
            pickle.dump(payload, handle)
    else:
        # LigPrep reads the first column as SMILES and the second as the title.
        with open(output_path, "w") as handle:
            for record in records:
                handle.write(f"{record['product_smiles']} {record['product_name']}\n")

    if metadata_path:
        metadata_dir = os.path.dirname(metadata_path)
        if metadata_dir:
            os.makedirs(metadata_dir, exist_ok=True)
        write_records_csv(records, metadata_path)


def parse_allowed_atoms(value):
    symbol_to_num = {"C": 6, "N": 7, "O": 8, "S": 16}
    nums = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item.isdigit():
            nums.add(int(item))
        elif item.upper() in symbol_to_num:
            nums.add(symbol_to_num[item.upper()])
        else:
            raise ValueError(f"Unsupported scaffold atom specifier: {item!r}")
    if not nums:
        raise ValueError("--scaffold_atoms must include at least one element")
    return nums


def main():
    parser = argparse.ArgumentParser(description="Attach fixed warheads to decoded CovaGen SMILES.")
    parser.add_argument("--input", required=True, help="Decoded SMILES file: .pkl/.pickle, .smi/.txt, or .csv.")
    parser.add_argument("--warheads_yaml", required=True, help="Path to warheads.yaml.")
    parser.add_argument("--warhead_id", required=True, help="Warhead ID in warheads.yaml, e.g. 3723.")
    parser.add_argument("--output", required=True, help="Output path: .csv, .pkl, .smi, or .txt.")
    parser.add_argument(
        "--covagen_atoms",
        "--scaffold_atoms",
        dest="scaffold_atoms",
        default="C",
        help="Comma-separated CovaGen-molecule atom symbols/numbers allowed for grafting. "
        "Default: C. Use C,N,O,S to include hetero-atom H sites.",
    )
    parser.add_argument(
        "--max_sites_per_mol",
        type=int,
        default=12,
        help="Maximum symmetry-unique scaffold sites to try per molecule. 0 means all sites.",
    )
    parser.add_argument(
        "--site_selection",
        choices=("first", "random"),
        default="first",
        help="Choose the first RDKit-ranked sites or sample sites in seeded random order.",
    )
    parser.add_argument("--random_seed", type=int, default=114514, help="Seed used by random site selection.")
    parser.add_argument(
        "--enumerations_per_input",
        type=int,
        default=0,
        help=(
            "Randomly select up to N symmetry-unique warhead attachments for each CovaGen molecule. "
            "This is the recommended option for docking enumeration and overrides --site_selection, "
            "--max_sites_per_mol, and --max_products_per_input. 0 uses the legacy controls."
        ),
    )
    parser.add_argument(
        "--max_products_per_input",
        type=int,
        default=25,
        help="Legacy cap on successful products per input. Use --enumerations_per_input for random docking sets.",
    )
    parser.add_argument("--max_total_products", type=int, default=0, help="0 means no global limit.")
    parser.add_argument("--no_dedupe", action="store_true", help="Keep duplicate product SMILES.")
    parser.add_argument("--pkl_records", action="store_true", help="For .pkl output, save full metadata records.")
    parser.add_argument(
        "--metadata_output",
        default=None,
        help="Companion metadata CSV path. For .smi/.txt, defaults to <output_stem>_metadata.csv.",
    )
    parser.add_argument(
        "--with_classifier",
        action="store_true",
        help=(
            "Assume the input SMILES already include the attached warhead; apply the same "
            "qc_required_smarts checks used after enumeration, do not enumerate attachments, "
            "and preserve the input title."
        ),
    )
    parser.add_argument(
        "--classifier_keep_n",
        "--keep_n",
        dest="classifier_keep_n",
        type=int,
        default=0,
        help=(
            "With --with_classifier, randomly retain at most N passing molecules after "
            "warhead QC and deduplication. Uses --random_seed. 0 keeps all molecules."
        ),
    )
    parser.add_argument("--progress_every", type=int, default=25, help="Print progress every N input molecules. 0 disables.")
    args = parser.parse_args()

    if args.enumerations_per_input < 0:
        parser.error("--enumerations_per_input must be 0 or greater")
    if args.classifier_keep_n < 0:
        parser.error("--classifier_keep_n must be 0 or greater")
    if args.classifier_keep_n and not args.with_classifier:
        parser.error("--classifier_keep_n requires --with_classifier")
    if args.classifier_keep_n and args.max_total_products:
        parser.error(
            "--classifier_keep_n cannot be combined with --max_total_products; "
            "random sampling must consider all classifier-passing molecules"
        )

    if args.with_classifier:
        args.enumerations_per_input = 0

    warhead = load_warhead(args.warheads_yaml, args.warhead_id)
    topology = warhead.get("attachment_topology", "")
    warhead_mol = mol_from_smiles(warhead.get("frag_smiles"), explicit_hs=True)
    if warhead_mol is None:
        raise ValueError(f"Could not parse warhead frag_smiles: {warhead.get('frag_smiles')!r}")

    warhead_graph = rdkit_mol_to_nx(warhead_mol)
    warhead_hydrogens, warhead_heavy_elements = topology_allowed_warhead_hydrogens(warhead_graph, topology)
    if not warhead_hydrogens:
        raise ValueError(
            f"No topology-allowed warhead hydrogens found for topology {topology!r} "
            f"and frag_smiles {warhead.get('frag_smiles')!r}."
        )

    input_records = load_smiles_records(args.input)
    allowed_atoms = parse_allowed_atoms(args.scaffold_atoms)
    qc_queries = compile_qc_smarts(warhead.get("qc_required_smarts", []))
    rng = random.Random(args.random_seed)
    metadata_path = args.metadata_output or default_metadata_path(args.output)

    records = []
    seen = set()
    failed_inputs = 0
    attempted_products = 0
    shortfall_inputs = 0
    classifier_checked = 0
    classifier_rejected = 0

    for input_idx, input_record in enumerate(input_records, start=1):
        if args.progress_every and (input_idx == 1 or input_idx % args.progress_every == 0):
            print(f"Processing input {input_idx}/{len(input_records)}; products so far: {len(records)}")

        parent_smiles = input_record["smiles"]
        parent_name = input_record["name"]
        covagen_parent_id = parent_id(input_idx)
        scaffold_mol = mol_from_smiles(parent_smiles, explicit_hs=True)
        if scaffold_mol is None:
            failed_inputs += 1
            if args.enumerations_per_input:
                shortfall_inputs += 1
            continue

        if args.with_classifier:
            attempted_products += 1
            classifier_checked += 1
            # Use exactly the same warhead integrity check as newly enumerated
            # products. The complete frag_smiles is not a reliable query here:
            # attachment replaces one of its hydrogens with the scaffold bond.
            if not passes_qc(scaffold_mol, qc_queries):
                classifier_rejected += 1
                continue
            if not args.no_dedupe and parent_smiles in seen:
                continue
            seen.add(parent_smiles)
            records.append(
                {
                    "product_smiles": parent_smiles,
                    "product_name": parent_name,
                    "parent_id": covagen_parent_id,
                    "parent_index": input_idx,
                    "enumeration_index": 0,
                    "parent_smiles": parent_smiles,
                    "parent_name": parent_name,
                    "warhead_id": str(args.warhead_id),
                    "warhead_name": warhead.get("name", ""),
                    "warhead_class": warhead.get("warhead_class", ""),
                    "attachment_topology": topology,
                    "scaffold_h_idx": "",
                    "scaffold_heavy_idx": "",
                    "scaffold_heavy_element": "",
                    "warhead_h_idx": "",
                    "warhead_heavy_idx": "",
                    "warhead_heavy_element": "",
                }
            )
            if args.max_total_products and len(records) >= args.max_total_products:
                output_records(
                    records,
                    args.output,
                    as_records=args.pkl_records,
                    metadata_path=metadata_path,
                )
                print_summary(
                    len(input_records),
                    failed_inputs,
                    attempted_products,
                    len(records),
                    args.output,
                    len(warhead_hydrogens),
                    metadata_path,
                    shortfall_inputs,
                    0,
                    classifier_mode=True,
                    classifier_checked=classifier_checked,
                    classifier_rejected=classifier_rejected,
                )
                return
            continue

        scaffold_hydrogens, scaffold_heavy_elements = scaffold_attachment_hydrogens_from_mol(
            scaffold_mol,
            allowed_atoms,
        )
        if args.enumerations_per_input:
            scaffold_pairs = select_scaffold_attachment_sites(
                scaffold_hydrogens,
                scaffold_heavy_elements,
                0,
                "random",
                rng,
            )
            product_limit = args.enumerations_per_input
        else:
            scaffold_pairs = select_scaffold_attachment_sites(
                scaffold_hydrogens,
                scaffold_heavy_elements,
                args.max_sites_per_mol,
                args.site_selection,
                rng,
            )
            product_limit = args.max_products_per_input

        attachment_candidates = [
            (scaffold_h_idx, scaffold_heavy_element, warhead_h_idx, warhead_heavy_element)
            for scaffold_h_idx, scaffold_heavy_element in scaffold_pairs
            for warhead_h_idx, warhead_heavy_element in zip(warhead_hydrogens, warhead_heavy_elements)
        ]
        if args.enumerations_per_input:
            rng.shuffle(attachment_candidates)

        made_for_input = 0
        seen_for_input = set()
        for (
            scaffold_h_idx,
            scaffold_heavy_element,
            warhead_h_idx,
            warhead_heavy_element,
        ) in attachment_candidates:
            if product_limit and made_for_input >= product_limit:
                break

            scaffold_heavy_idx = heavy_neighbor_in_mol(scaffold_mol, scaffold_h_idx)
            attempted_products += 1
            try:
                product = attach_by_hydrogen_pair(scaffold_mol, scaffold_h_idx, warhead_mol, warhead_h_idx)
            except Exception:
                continue
            if product is None or not passes_qc(product, qc_queries):
                continue

            product_smiles = Chem.MolToSmiles(product, canonical=True)
            dedupe_set = seen_for_input if args.enumerations_per_input else seen
            if not args.no_dedupe and product_smiles in dedupe_set:
                continue
            dedupe_set.add(product_smiles)

            enumeration_index = made_for_input + 1
            records.append(
                {
                    "product_smiles": product_smiles,
                    "product_name": product_name(
                        covagen_parent_id,
                        args.warhead_id,
                        enumeration_index,
                    ),
                    "parent_id": covagen_parent_id,
                    "parent_index": input_idx,
                    "enumeration_index": enumeration_index,
                    "parent_smiles": parent_smiles,
                    "parent_name": parent_name,
                    "warhead_id": str(args.warhead_id),
                    "warhead_name": warhead.get("name", ""),
                    "warhead_class": warhead.get("warhead_class", ""),
                    "attachment_topology": topology,
                    "scaffold_h_idx": scaffold_h_idx,
                    "scaffold_heavy_idx": scaffold_heavy_idx,
                    "scaffold_heavy_element": scaffold_heavy_element,
                    "warhead_h_idx": warhead_h_idx,
                    "warhead_heavy_idx": get_a_neighbor_id(warhead_graph, warhead_h_idx),
                    "warhead_heavy_element": warhead_heavy_element,
                }
            )
            made_for_input += 1

            if args.max_total_products and len(records) >= args.max_total_products:
                output_records(
                    records,
                    args.output,
                    as_records=args.pkl_records,
                    metadata_path=metadata_path,
                )
                print_summary(
                    len(input_records),
                    failed_inputs,
                    attempted_products,
                    len(records),
                    args.output,
                    len(warhead_hydrogens),
                    metadata_path,
                    shortfall_inputs,
                    args.enumerations_per_input,
                )
                return

        if args.enumerations_per_input and made_for_input < args.enumerations_per_input:
            shortfall_inputs += 1

    classifier_passing_before_sample = len(records) if args.with_classifier else 0
    if args.classifier_keep_n:
        records = random_sample_records(records, args.classifier_keep_n, rng)

    output_records(
        records,
        args.output,
        as_records=args.pkl_records,
        metadata_path=metadata_path,
    )
    print_summary(
        len(input_records),
        failed_inputs,
        attempted_products,
        len(records),
        args.output,
        len(warhead_hydrogens),
        metadata_path,
        shortfall_inputs,
        args.enumerations_per_input,
        classifier_mode=args.with_classifier,
        classifier_checked=classifier_checked,
        classifier_rejected=classifier_rejected,
        classifier_keep_n=args.classifier_keep_n,
        classifier_passing_before_sample=classifier_passing_before_sample,
    )


def print_summary(
    num_inputs,
    failed_inputs,
    attempted_products,
    num_records,
    output,
    warhead_h_count,
    metadata_path=None,
    shortfall_inputs=0,
    requested_enumerations=0,
    classifier_mode=False,
    classifier_checked=0,
    classifier_rejected=0,
    classifier_keep_n=0,
    classifier_passing_before_sample=0,
):
    print(f"Topology-allowed unique warhead H atoms: {warhead_h_count}")
    print(f"Input SMILES: {num_inputs}")
    print(f"Invalid input SMILES skipped: {failed_inputs}")
    print(f"Attachment attempts: {attempted_products}")
    print(f"Products written: {num_records}")
    print(f"Output: {output}")
    if metadata_path:
        print(f"Metadata: {metadata_path}")
    if requested_enumerations:
        print(
            f"Inputs with fewer than {requested_enumerations} valid unique attachments: "
            f"{shortfall_inputs}"
        )
    if classifier_mode:
        rejected_percent = (
            100.0 * classifier_rejected / classifier_checked
            if classifier_checked
            else 0.0
        )
        print(f"Valid input SMILES checked for warhead QC: {classifier_checked}")
        print(
            "Molecules rejected by warhead QC: "
            f"{classifier_rejected}/{classifier_checked} ({rejected_percent:.2f}%)"
        )
        if classifier_keep_n:
            print(
                "Classifier-passing molecules before random sampling: "
                f"{classifier_passing_before_sample}"
            )
            print(
                "Classifier molecules retained after random sampling: "
                f"{num_records} (limit: {classifier_keep_n})"
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
