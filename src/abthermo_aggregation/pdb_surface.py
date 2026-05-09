"""PDB parsing and molecular-surface descriptors.

This module intentionally avoids benchmark-specific files. It computes local
surface geometry directly from PDB atom coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


KD_HYDROPATHY = {
    "ALA": 1.8,
    "ARG": -4.5,
    "ASN": -3.5,
    "ASP": -3.5,
    "CYS": 2.5,
    "GLN": -3.5,
    "GLU": -3.5,
    "GLY": -0.4,
    "HIS": -3.2,
    "ILE": 4.5,
    "LEU": 3.8,
    "LYS": -3.9,
    "MET": 1.9,
    "PHE": 2.8,
    "PRO": -1.6,
    "SER": -0.8,
    "THR": -0.7,
    "TRP": -0.9,
    "TYR": -1.3,
    "VAL": 4.2,
}

RESIDUE_CHARGE = {"ASP": -1.0, "GLU": -1.0, "LYS": 1.0, "ARG": 1.0, "HIS": 0.25}
VDW_RADIUS = {"H": 1.2, "C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8, "P": 1.8}
PROTEIN_RESIDUES = set(KD_HYDROPATHY)


@dataclass(frozen=True)
class AtomRecord:
    coord: np.ndarray
    residue_name: str
    chain_id: str
    residue_number: int
    atom_name: str
    hydrophobicity: float
    charge: float
    radius: float


def fibonacci_sphere(n_points: int) -> np.ndarray:
    phi = math.pi * (3.0 - math.sqrt(5.0))
    points = []
    for i in range(n_points):
        y = 1.0 - (i / (n_points - 1)) * 2.0 if n_points > 1 else 0.0
        radius = math.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i
        points.append((math.cos(theta) * radius, y, math.sin(theta) * radius))
    return np.asarray(points, dtype=float)


SPHERE_POINTS = fibonacci_sphere(18)


def element_from_pdb_line(line: str) -> str:
    element = line[76:78].strip()
    if element:
        return element[0].upper()
    letters = "".join(ch for ch in line[12:16].strip() if ch.isalpha())
    return (letters[:1] or "C").upper()


def parse_pdb_atoms(path: str | Path, chains: set[str] | None = None) -> list[AtomRecord]:
    atoms: list[AtomRecord] = []
    seen = set()
    with Path(path).open() as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            chain_id = line[21].strip()
            if chains is not None and chain_id not in chains:
                continue
            residue_name = line[17:20].strip()
            if residue_name not in PROTEIN_RESIDUES:
                continue
            atom_name = line[12:16].strip()
            if atom_name.startswith("H"):
                continue
            altloc = line[16].strip()
            if altloc not in ("", "A"):
                continue
            key = (chain_id, line[22:26].strip(), line[26].strip(), atom_name)
            if key in seen:
                continue
            seen.add(key)
            element = element_from_pdb_line(line)
            atoms.append(
                AtomRecord(
                    coord=np.asarray(
                        [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                        dtype=float,
                    ),
                    residue_name=residue_name,
                    chain_id=chain_id,
                    residue_number=int(line[22:26]),
                    atom_name=atom_name,
                    hydrophobicity=KD_HYDROPATHY[residue_name] / 4.5,
                    charge=RESIDUE_CHARGE.get(residue_name, 0.0),
                    radius=VDW_RADIUS.get(element, 1.7),
                )
            )
    return atoms


def infer_two_largest_protein_chains(path: str | Path) -> tuple[str, str]:
    chain_residues: dict[str, set[tuple[int, str]]] = {}
    for atom in parse_pdb_atoms(path, chains=None):
        chain_residues.setdefault(atom.chain_id, set()).add((atom.residue_number, atom.residue_name))
    if len(chain_residues) < 2:
        raise ValueError(f"Could not infer two protein chains from {path}; provide a manifest.")
    ranked = sorted(chain_residues.items(), key=lambda item: len(item[1]), reverse=True)
    return ranked[0][0], ranked[1][0]


def _cdr_like_window(chain_position: int, chain_is_heavy: bool) -> bool:
    if chain_is_heavy:
        return (26 <= chain_position <= 35) or (50 <= chain_position <= 65) or (95 <= chain_position <= 112)
    return (24 <= chain_position <= 34) or (50 <= chain_position <= 56) or (89 <= chain_position <= 97)


def compute_surface_descriptors(
    pdb_path: str | Path,
    heavy_chain: str,
    light_chain: str,
    probe_radius: float = 1.4,
    max_curvature_points: int = 1800,
) -> dict[str, float | int | str]:
    atoms = parse_pdb_atoms(pdb_path, {heavy_chain, light_chain})
    if len(atoms) < 50:
        raise ValueError(f"Too few parsed antibody atoms for {pdb_path}")

    coords = np.asarray([atom.coord for atom in atoms], dtype=float)
    radii = np.asarray([atom.radius + probe_radius for atom in atoms], dtype=float)
    hyd = np.asarray([atom.hydrophobicity for atom in atoms], dtype=float)
    charge = np.asarray([atom.charge for atom in atoms], dtype=float)
    atom_tree = cKDTree(coords)
    max_radius = float(radii.max())

    surface_points = []
    normals = []
    area_weights = []
    point_hyd = []
    point_charge = []
    atom_indices = []

    for atom_index in range(len(atoms)):
        radius = radii[atom_index]
        exposed_area_per_point = 4.0 * math.pi * radius * radius / len(SPHERE_POINTS)
        candidates = coords[atom_index] + SPHERE_POINTS * radius
        neighbor_lists = atom_tree.query_ball_point(candidates, max_radius + 0.2)
        for point_index, point in enumerate(candidates):
            buried = False
            for neighbor_index in neighbor_lists[point_index]:
                if neighbor_index == atom_index:
                    continue
                if np.linalg.norm(point - coords[neighbor_index]) < radii[neighbor_index] - 0.05:
                    buried = True
                    break
            if not buried:
                surface_points.append(point)
                normals.append(SPHERE_POINTS[point_index])
                area_weights.append(exposed_area_per_point)
                point_hyd.append(hyd[atom_index])
                point_charge.append(charge[atom_index])
                atom_indices.append(atom_index)

    surface_points = np.asarray(surface_points, dtype=float)
    normals = np.asarray(normals, dtype=float)
    area_weights = np.asarray(area_weights, dtype=float)
    point_hyd = np.asarray(point_hyd, dtype=float)
    point_charge = np.asarray(point_charge, dtype=float)
    atom_indices = np.asarray(atom_indices, dtype=int)
    if len(surface_points) < 20:
        raise ValueError(f"Too few surface points for {pdb_path}")

    selected = _select_curvature_points(point_hyd, max_curvature_points)
    surface_tree = cKDTree(surface_points)
    curvedness_values = []
    mean_curvature_values = []
    selected_hyd = []
    selected_weights = []

    for idx in selected:
        curvedness, mean_curvature = _fit_local_curvature(
            surface_points, normals, radii, atom_indices, surface_tree, idx
        )
        curvedness_values.append(curvedness)
        mean_curvature_values.append(mean_curvature)
        selected_hyd.append(point_hyd[idx])
        selected_weights.append(area_weights[idx])

    curvedness_values = np.asarray(curvedness_values, dtype=float)
    mean_curvature_values = np.asarray(mean_curvature_values, dtype=float)
    selected_hyd = np.asarray(selected_hyd, dtype=float)
    selected_weights = np.asarray(selected_weights, dtype=float)
    selected_hyd_positive = np.maximum(selected_hyd, 0.0)

    hidden = _hidden_hydrophobicity(atoms, atom_indices, heavy_chain, light_chain)
    hcm_integral = float(np.sum(selected_weights * selected_hyd_positive * curvedness_values))
    hcm_density = hcm_integral / max(float(np.sum(selected_weights)), 1e-6)

    return {
        "pdb_path": str(pdb_path),
        "heavy_chain": heavy_chain,
        "light_chain": light_chain,
        "n_atoms": len(atoms),
        "n_surface_points": len(surface_points),
        "surface_area_proxy": float(area_weights.sum()),
        "hydrophobic_curvature_density": hcm_density,
        "hydrophobic_curvature_integral": hcm_integral,
        "hydrophobic_curvedness_mean": float(
            np.average(curvedness_values, weights=selected_weights * selected_hyd_positive + 1e-6)
        ),
        "hydrophobic_surface": float(np.sum(area_weights * np.maximum(point_hyd, 0.0))),
        "absolute_charge_surface": float(np.sum(area_weights * np.abs(point_charge))),
        "signed_mean_curvature_hydrophobic": float(
            np.average(mean_curvature_values, weights=selected_weights * selected_hyd_positive + 1e-6)
        ),
        **hidden,
    }


def _select_curvature_points(point_hyd: np.ndarray, max_curvature_points: int) -> np.ndarray:
    rng = np.random.default_rng(7)
    importance = np.maximum(point_hyd, 0.0) + 0.25
    n_selected = min(max_curvature_points, len(point_hyd))
    return rng.choice(len(point_hyd), size=n_selected, replace=False, p=importance / importance.sum())


def _fit_local_curvature(
    surface_points: np.ndarray,
    normals: np.ndarray,
    radii: np.ndarray,
    atom_indices: np.ndarray,
    surface_tree: cKDTree,
    idx: int,
) -> tuple[float, float]:
    point = surface_points[idx]
    normal = normals[idx] / np.linalg.norm(normals[idx])
    reference = np.asarray([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.8 else np.asarray([0.0, 1.0, 0.0])
    t1 = np.cross(normal, reference)
    t1 = t1 / np.linalg.norm(t1)
    t2 = np.cross(normal, t1)

    neighbor_indices = surface_tree.query_ball_point(point, 5.0)
    if len(neighbor_indices) < 8:
        k1 = k2 = 1.0 / max(radii[atom_indices[idx]], 1.0)
    else:
        delta = surface_points[neighbor_indices] - point
        u = delta @ t1
        v = delta @ t2
        z = delta @ normal
        mask = (u * u + v * v) > 1e-6
        u, v, z = u[mask], v[mask], z[mask]
        if len(u) < 6:
            k1 = k2 = 1.0 / max(radii[atom_indices[idx]], 1.0)
        else:
            design = np.column_stack([0.5 * u * u, u * v, 0.5 * v * v, u, v, np.ones_like(u)])
            try:
                coef = np.linalg.lstsq(design, z, rcond=None)[0]
                hessian = np.asarray([[coef[0], coef[1]], [coef[1], coef[2]]], dtype=float)
                eigenvalues = np.linalg.eigvalsh(hessian)
                k1, k2 = float(eigenvalues[1]), float(eigenvalues[0])
            except Exception:
                k1 = k2 = 1.0 / max(radii[atom_indices[idx]], 1.0)

    curvedness = min(math.sqrt((k1 * k1 + k2 * k2) / 2.0), 1.25)
    return curvedness, 0.5 * (k1 + k2)


def _hidden_hydrophobicity(
    atoms: list[AtomRecord],
    atom_indices: np.ndarray,
    heavy_chain: str,
    light_chain: str,
) -> dict[str, float]:
    surface_atom_counts = np.bincount(atom_indices, minlength=len(atoms))
    residues = []
    seen_residues = set()
    exposed_by_residue = {}
    total_by_residue = {}
    for atom_index, atom in enumerate(atoms):
        key = (atom.chain_id, atom.residue_number, atom.residue_name)
        exposed_by_residue[key] = exposed_by_residue.get(key, 0.0) + surface_atom_counts[atom_index]
        total_by_residue[key] = total_by_residue.get(key, 0.0) + len(SPHERE_POINTS)
        if key not in seen_residues:
            seen_residues.add(key)
            residues.append((atom.chain_id, atom.residue_number, atom.residue_name))

    hidden_cdr = 0.0
    cdr_total = 0.0
    buried_all = 0.0
    for chain_id in [heavy_chain, light_chain]:
        chain_residues = sorted([res for res in residues if res[0] == chain_id], key=lambda res: res[1])
        for chain_position, residue in enumerate(chain_residues, start=1):
            key = residue
            hydrophobicity = max(KD_HYDROPATHY[residue[2]] / 4.5, 0.0)
            exposure_fraction = exposed_by_residue.get(key, 0.0) / max(total_by_residue.get(key, 1.0), 1.0)
            hidden_component = hydrophobicity * max(0.0, 1.0 - exposure_fraction)
            buried_all += hidden_component
            is_heavy = chain_id == heavy_chain
            if _cdr_like_window(chain_position, is_heavy):
                hidden_cdr += hidden_component
                cdr_total += hydrophobicity
    return {
        "hidden_cdr_hydrophobicity": float(hidden_cdr),
        "cdr_hydrophobicity": float(cdr_total),
        "buried_hydrophobicity_all": float(buried_all),
    }
