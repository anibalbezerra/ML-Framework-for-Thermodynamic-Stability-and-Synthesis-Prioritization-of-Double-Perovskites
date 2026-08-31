#!/usr/bin/env python3
"""
run_phonon_MLIP_mattersim.py
Refactored phonon + stability workflow:
 - Prefer MatterSim calculator if available, else fall back to CHGNet (unchanged behavior).
 - Keep full phonon analysis, Δ-scan, thermal properties and enhanced reporting.
 - Add structural stability descriptors:
     * Goldschmidt Tolerance Factor
     * Octahedral Factor
     * Bond Valence Sum (BVS) - optional R0 params file
     * Polyhedral Distortion (B-O bond variations)
 - Options control whether descriptors are computed.
"""

import os
import json
import numpy as np
from ase import Atoms
from ase.io import write
from ase.optimize import BFGS
from ase.phonons import Phonons
from ase import units
from pymatgen.ext.matproj import MPRester
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
import argparse
from pathlib import Path
import warnings

# Prefer MatterSim if available
try:
    from mattersim.forcefield import MatterSimCalculator
    MATTERSIM_AVAILABLE = True
except Exception:
    MATTERSIM_AVAILABLE = False

# Fallback CHGNet imports (unchanged usage)
try:
    from chgnet.model.dynamics import CHGNetCalculator
    from chgnet.model import CHGNet
    CHGNET_AVAILABLE = True
except Exception:
    CHGNET_AVAILABLE = False

# Try to import user's get_atomic_property helper
import sys
from pathlib import Path
HAVE_GET_RADII = False
try:
    helper_path = Path("/scratch_drive/anibal/DFT_out_of_nothing/phonons")
    sys.path.insert(0, str(helper_path))
    from raddi_helper import get_atomic_property  # <-- your file name
    HAVE_GET_RADII = True
    print(f"[radii-debug] Successfully imported get_atomic_property from {helper_path}")
except Exception as e:
    print(f"[radii-debug] Could not import get_atomic_property helper: {e}")

# Default oxygen ionic radius fallback (approx Shannon r_ionic for O2- in VI)
DEFAULT_R_O_VI = 1.35  # Å (fallback)

# ---------- Helper: MP fetch and ASE conversion ----------
def get_structure_from_mp(mp_id: str, api_key: str):
    print(f"Fetching structure {mp_id} from Materials Project...")
    with MPRester(api_key) as mpr:
        doc = mpr.get_summary_by_material_id(mp_id, fields=["structure", "formation_energy_per_atom", "energy_above_hull"])
        struct = doc["structure"]
        formation_energy = doc["formation_energy_per_atom"]
        e_hull = doc["energy_above_hull"]
    print(f"✅ Retrieved: {struct.formula}")
    return struct, formation_energy, e_hull

def ase_from_pmg(struct):
    return Atoms(symbols=[str(sp) for sp in struct.species],
                 scaled_positions=struct.frac_coords,
                 cell=struct.lattice.matrix,
                 pbc=True)

# ---------- MatterSim / CHGNet calculator helper ----------
def create_mattersim_calculator(device='auto', model_size='1M'):
    if not MATTERSIM_AVAILABLE:
        raise ImportError("MatterSim is not installed.")
    if device == 'auto':
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    load_path = "MatterSim-v1.0.0-5M.pth" if model_size == '5M' else "MatterSim-v1.0.0-1M.pth"
    print(f"🔧 Initializing MatterSim calculator (device: {device}, model: {model_size})")
    return MatterSimCalculator(load_path=load_path, device=device)

def create_chgnet_calculator(device='cpu'):
    if not CHGNET_AVAILABLE:
        raise ImportError("CHGNet is not installed.")
    model = CHGNet.load()
    print("🔧 Initializing CHGNet calculator (fallback)")
    return CHGNetCalculator(model=model, use_device=device)

# ---------- Phonon helpers (kept from your original scripts) ----------
def compute_dynamical_matrix_from_fc(ph, fc, q=[0,0,0]):
    n_cells, ndof_i, ndof_j = fc.shape
    N = len(ph.atoms)
    assert ndof_i == ndof_j == 3 * N
    masses = np.array([a.mass for a in ph.atoms])
    M_inv = 1.0 / np.sqrt(np.outer(masses, masses))
    M_inv_full = np.repeat(np.repeat(M_inv, 3, axis=0), 3, axis=1)
    D_real = np.sum(fc, axis=0)
    D_massnorm = D_real * M_inv_full
    return D_massnorm

def try_manual_phonon_processing(ph):
    from numpy.linalg import eigh
    from pathlib import Path
    print("\nReading cached force constants (ASE public API)...")
    ph.read(acoustic=True)
    fc = ph.get_force_constant()
    if fc is None:
        raise RuntimeError("ph.get_force_constant() returned None — no data read!")
    print(f"✅ Force constants loaded, shape = {fc.shape}")
    calc_dir = Path(ph.name)
    disp_dirs = sorted(calc_dir.glob("disp*"))
    print(f"Found {len(disp_dirs)} displacement subfolders in cache directory '{calc_dir}'")
    D_q = compute_dynamical_matrix_from_fc(ph, fc, q=[0,0,0])
    w, v = eigh(D_q)
    print("Three lowest eigenvalues at Γ:", w[:6])
    return D_q, w, v

def analyze_gamma_point(D_q):
    w, v = np.linalg.eigh(D_q)
    s = units._hbar * 1e10 / np.sqrt(units._e * units._amu)
    freqs_ev = np.sign(w) * np.sqrt(np.abs(w)) * s
    idx = np.argsort(np.abs(freqs_ev))
    print("\nSix smallest (by abs) Γ frequencies (eV):")
    for i in idx[:6]:
        print(i, freqs_ev[i])
    if np.any(freqs_ev < 0):
        im_idx = np.argmin(freqs_ev)
        print("Imag mode vector (first 30 components):", v[:30, im_idx])
    return freqs_ev

def check_displacement_cache(ph):
    from pathlib import Path
    p = Path(ph.name)
    print("ph.name:", ph.name)
    print("Exists:", p.exists())
    if p.exists():
        entries = list(p.iterdir())
        print("Listing entries:", entries[:50])
        disp_dirs = sorted([d for d in p.iterdir() if d.name.startswith("disp")])
        print("Found displacement dirs:", len(disp_dirs))
        for d in disp_dirs[:10]:
            print(" ", d.name)
        return len(disp_dirs) > 0
    return False

def check_acoustic_modes(ph):
    print("\n--- Acoustic Mode Check ---")
    try:
        fc = ph.get_force_constant()
        D0 = np.sum(fc, axis=0)
        w, v = np.linalg.eigh(D0)
        print("raw D eigenvals (lowest 6):", w[:6])
        s = units._hbar * 1e10 / np.sqrt(units._e * units._amu)
        freqs_ev = np.sign(w) * np.sqrt(np.abs(w)) * s
        eV_to_cm = 8065.54429
        print("Lowest frequencies (eV and cm^-1):")
        for i, f in enumerate(freqs_ev[:6]):
            print(f"  {i}: {f:.6e} eV  {f*eV_to_cm:.2f} cm^-1")
    except Exception as e:
        print(f"Acoustic mode check failed: {e}")

def analyze_imaginary_mode(D_q, atoms):
    w, v = np.linalg.eigh(D_q)
    s = units._hbar * 1e10 / np.sqrt(units._e * units._amu)
    freqs_ev = np.sign(w) * np.sqrt(np.abs(w)) * s
    if np.any(freqs_ev < -1e-6):
        im_idx = np.argmin(freqs_ev)
        vec = v[:, im_idx].real
        vec_norm = np.linalg.norm(vec.reshape(-1,3), axis=1)
        print(f"\n--- Imaginary Mode Analysis ({freqs_ev[im_idx]:.6f} eV) ---")
        print("Per-atom participation (largest amplitudes):")
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        indices_sorted = np.argsort(vec_norm)[::-1]
        for i in indices_sorted[:10]:
            print(f"  {i:3d} {symbols[i]:3s} amp={vec_norm[i]:.4f} pos={positions[i]}")
        return vec.reshape(-1,3)
    return None

# ---------- Isolated cache (unchanged) ----------
def create_isolated_phonon_cache(mp_id, base_dir="./phonon_cache"):
    from pathlib import Path
    import tempfile
    cache_dir = Path(base_dir) / mp_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Using isolated cache directory: {cache_dir}")
    return str(cache_dir)

def cleanup_phonon_cache(mp_id, base_dir="./phonon_cache"):
    from pathlib import Path
    import shutil
    cache_dir = Path(base_dir) / mp_id
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"🧹 Cleaned up cache directory: {cache_dir}")

class IsolatedPhonons(Phonons):
    def __init__(self, atoms, calculator, mp_id, supercell=(1,1,1), delta=0.01, base_cache_dir="./phonon_cache", **kwargs):
        self.cache_dir = create_isolated_phonon_cache(mp_id, base_cache_dir)
        self.mp_id = mp_id
        super().__init__(atoms, calculator, name=self.cache_dir, supercell=supercell, delta=delta, **kwargs)
    def clean(self):
        cleanup_phonon_cache(self.mp_id)

def run_phonons_isolated(atoms, calculator, mp_id, supercell=(3,3,3), delta=0.01):
    print(f"🔒 Running phonons with isolated cache for {mp_id}")
    ph = IsolatedPhonons(atoms=atoms, calculator=calculator, mp_id=mp_id, supercell=supercell, delta=delta)
    ph.run()
    return ph

# ---------- Structural descriptor helpers ----------
def _get_ionic_radius(element: str, charge: str = '2', coordination: str = 'VI'):
    """
    Verbose version: Try to get ionic radius using provided get_atomic_property helper if available,
    otherwise return None so caller can use fallback defaults.

    Adds detailed debug information to help diagnose missing radii or file issues.
    """
    print(f"[radii-debug] Attempting to get radius for element={element}, charge={charge}, coordination={coordination}")
    if HAVE_GET_RADII:
        try:
            val = get_atomic_property(
                element=element,
                charge=charge,
                coordination=coordination,
                property_name='r_ionic'
            )
            if val is not None:
                print(f"[radii-debug] SUCCESS: Found r_ionic={val} Å for {element}^{charge}+ ({coordination})")
                return float(val)
            else:
                print(f"[radii-debug] WARNING: No r_ionic found in database for {element}^{charge}+ ({coordination})")
        except FileNotFoundError as fe:
            print(f"[radii-debug] ERROR: Shannon radii JSON file not found. Details: {fe}")
        except json.JSONDecodeError as je:
            print(f"[radii-debug] ERROR: Malformed JSON in shannon-radii file. Details: {je}")
        except Exception as e:
            print(f"[radii-debug] UNABLE to retrieve radius for {element}: {e}")
    else:
        print(f"[radii-debug] get_atomic_property helper not available; returning None")

    # If no value retrieved
    print(f"[radii-debug] Returning None for element={element}")
    return None


def detect_b_sites(structure_ase, max_B_O_dist=2.5, min_B_O_coord=4, prefer_coord=6):
    """
    Heuristic: identify cations that have at least `min_B_O_coord` oxygen neighbors
    within `max_B_O_dist`. Returns a dict: {element: [site indices...], ...}
    """
    atoms = structure_ase
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    oxy_indices = [i for i,s in enumerate(symbols) if s == "O"]
    N = len(atoms)
    from scipy.spatial import cKDTree
    kdt = cKDTree(positions)
    b_site_indices = []
    for i, s in enumerate(symbols):
        if s == "O":
            continue
        # query O neighbors within cutoff
        dists, idxs = kdt.query(positions[i], k=len(oxy_indices)+1, distance_upper_bound=max_B_O_dist)
        # count finite distances that correspond to O indices
        count = 0
        for d, idx in zip(dists, idxs):
            if np.isfinite(d) and idx < N and symbols[idx] == 'O':
                count += 1
        if count >= min_B_O_coord:
            b_site_indices.append(i)
    # group by element
    from collections import defaultdict
    grouped = defaultdict(list)
    for idx in b_site_indices:
        grouped[atoms.get_chemical_symbols()[idx]].append(idx)
    return dict(grouped)

def compute_B_O_bond_lengths(atoms, b_indices, cutoff=3.0):
    """Return list of bond lengths for each B-site index to O neighbors."""
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    from scipy.spatial import cKDTree
    kdt = cKDTree(positions)
    bonds = {}
    for idx in b_indices:
        dists, idxs = kdt.query(positions[idx], k=50, distance_upper_bound=cutoff)
        bl = []
        for d, j in zip(dists, idxs):
            if np.isfinite(d) and j < len(atoms) and symbols[j] == 'O':
                bl.append(float(d))
        bonds[idx] = bl
    return bonds

def goldschmidt_tolerance(r_A, r_B, r_O=DEFAULT_R_O_VI):
    # t = (r_A + r_O) / (sqrt(2) * (r_B + r_O))
    return (r_A + r_O) / (np.sqrt(2.0) * (r_B + r_O))

def octahedral_factor(r_B, r_O=DEFAULT_R_O_VI):
    return r_B / r_O

def compute_polyhedral_distortion(bond_lengths):
    """
    simple distortion index for a single polyhedron:
      DI = (1/N) * sum(|d_i - d_mean|) / d_mean
    Returns mean, std, DI
    """
    if not bond_lengths:
        return None
    arr = np.array(bond_lengths)
    dmean = arr.mean()
    dstd = arr.std()
    DI = np.mean(np.abs(arr - dmean)) / dmean
    return float(dmean), float(dstd), float(DI)

def read_bv_params(path):
    """
    Read bond-valence params JSON file expected format:
    {
      "pairs": {
         "Fe3+-O2-": {"R0": 1.759, "b": 0.37},
         ...
      }
    }
    """
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        warnings.warn(f"Could not read BVS params file {path}: {e}")
        return None

def bond_valence_sum_for_site(bonds_list, cation_symbol, oxidation_state, bv_params=None):
    """
    Verbose debug version.
    Compute BVS for a cation site using bond-valence parameters if available.

    Prints detailed lookup and computation steps so we can trace any mismatch.
    """
    if not bonds_list:
        print(f"[BVS-debug] No bond distances provided for {cation_symbol} (oxidation {oxidation_state}); returning None")
        return None

    CN = len(bonds_list)
    print(f"[BVS-debug] Computing BVS for {cation_symbol}^{oxidation_state}+ with {CN} bonds")

    if bv_params:
        # Generate possible lookup keys to match JSON format
        ks = [
            f"{cation_symbol}{oxidation_state}+-O2-",  # matches "La3+-O2-" etc.
            f"{cation_symbol}{oxidation_state:+d}-O2-".replace("+", "") + "+",
            f"{cation_symbol}{oxidation_state}-O2-",
            f"{cation_symbol}{oxidation_state:+d}-O2-",
            f"{cation_symbol}-O2-",
            f"{cation_symbol}-{ 'O' }",
            f"{cation_symbol}{oxidation_state}-O"
        ]

        print(f"[BVS-debug] Trying possible key patterns for lookup:")
        for k in ks:
            print(f"    → {k}")

        pair_key = None
        pairs_dict = bv_params.get('pairs', {})
        for k in ks:
            if k in pairs_dict:
                pair_key = k
                break

        if pair_key:
            R0 = float(pairs_dict[pair_key]['R0'])
            b = float(pairs_dict[pair_key].get('b', 0.37))
            print(f"[BVS-debug] Found parameters for {pair_key}: R0={R0:.4f} Å, b={b:.4f}")
            s_sum = 0.0
            for i, d in enumerate(bonds_list):
                s = np.exp((R0 - d) / b)
                s_sum += s
                print(f"      bond {i+1}: d={d:.4f} Å → s={s:.4f}")
            print(f"[BVS-debug] Total BVS({cation_symbol}^{oxidation_state}+) = {s_sum:.4f}")
            return float(s_sum)
        else:
            print(f"[BVS-debug] ❌ No matching key found in bv_params for {cation_symbol}^{oxidation_state}+")
            print(f"[BVS-debug] Available keys include {len(pairs_dict.keys())} entries; sample: {list(pairs_dict.keys())[:8]}")
    else:
        print(f"[BVS-debug] ❌ bv_params dictionary is None (no file loaded)")

    # ---- Fallback simple Pauling-like estimate ----
    warnings.warn("BVS params not provided or pair not found; returning coordination-based estimate (not a substitute for true BVS)")
    s_per_bond = float(oxidation_state) / max(1, CN)
    bvs_est = s_per_bond * CN
    print(f"[BVS-debug] Fallback estimate: {CN} bonds, oxidation {oxidation_state} → s_per_bond={s_per_bond:.3f}, total={bvs_est:.3f}")
    return float(bvs_est)


# ---------- Generate structural descriptors ----------
def compute_structural_descriptors(mp_id, atoms, double_perov_mode='mean', bv_params_path=None):
    """
    Compute requested structural descriptors and return a dict.
    """
    comp = Composition(atoms.get_chemical_formula())
    # identify likely A and B site radii heuristically:
    # We'll detect B-site indices and group elements
    descriptors = {}
    # detect B sites heuristically
    try:
        b_groups = detect_b_sites(atoms)
    except Exception as e:
        print("B-site detection failed (scipy required). Skipping B-site based descriptors:", e)
        b_groups = {}
    # compute B-O bond lengths
    b_indices = [idx for lst in b_groups.values() for idx in lst]
    bonds = compute_B_O_bond_lengths(atoms, b_indices)
    # Build element-level summaries
    element_radii = {}
    # get oxygen ionic radius (default to VI O2-)
    r_O = _get_ionic_radius('O', charge='-2', coordination='VI') or DEFAULT_R_O_VI
    descriptors['r_O_VI'] = float(r_O)
    # compute per-element ionic radii if available
    for el in set(atoms.get_chemical_symbols()):
        if el == 'O':
            element_radii[el] = r_O
        else:
            # try some common charges: +2, +3, +4, +1
            r = None
            for ch in ['2','3','4','1']:
                r = _get_ionic_radius(el, charge=ch, coordination='VI')
                if r is not None:
                    element_radii[el] = float(r)
                    break
            if el not in element_radii:
                element_radii[el] = None
    descriptors['element_radii_guess'] = element_radii

    # Decide A-site and B-site elements heuristically:
    # A-site ~ largest cation (non-O) by ionic radius guess, B-site ~ cations flagged earlier
    non_ox_elems = [el for el in set(atoms.get_chemical_symbols()) if el != 'O']
    # pick A as element with max radius (where radius known), fallback to first non-O
    A_candidate = None
    max_r = -1
    for el in non_ox_elems:
        r = element_radii.get(el)
        if r is not None and r > max_r:
            max_r = r
            A_candidate = el
    if A_candidate is None and non_ox_elems:
        A_candidate = non_ox_elems[0]
    descriptors['A_site_guess'] = A_candidate

    # B-site element list from detection or fallback: choose those elements that appear among detected b_groups
    B_elements = list(b_groups.keys()) if b_groups else [el for el in non_ox_elems if el != A_candidate]
    descriptors['B_site_elements_guess'] = B_elements

    # For double perovskite option, compute mean B radius if requested
    B_radii = []
    for b_el in B_elements:
        r = element_radii.get(b_el)
        if r is not None:
            B_radii.append(r)
    if not B_radii:
        # fallback: try atomic radii from pymatgen (not ionic)
        try:
            from pymatgen.core.periodic_table import Element
            B_radii = []
            for b_el in B_elements:
                try:
                    a = Element(b_el)
                    B_radii.append(a.atomic_radius or a.covalent_radius or 0.0)
                except Exception:
                    pass
        except Exception:
            pass

    r_B_effective = None
    if B_radii:
        if double_perov_mode == 'mean':
            r_B_effective = float(np.mean(B_radii))
        else:
            # in 'separate' mode we keep the list and will report per-element results
            r_B_effective = None

    # A radius
    r_A = element_radii.get(A_candidate) if A_candidate else None

    # Goldschmidt & octahedral
    descriptors['goldschmidt'] = {}
    if r_A is not None and (r_B_effective is not None or (len(B_radii)==1)):
        if r_B_effective is None and len(B_radii)==1:
            r_B_effective = B_radii[0]
        t = goldschmidt_tolerance(r_A, r_B_effective, r_O=r_O)
        descriptors['goldschmidt']['tolerance_factor'] = float(t)
        descriptors['goldschmidt']['r_A'] = float(r_A)
        descriptors['goldschmidt']['r_B_effective'] = float(r_B_effective)
        descriptors['goldschmidt']['r_O'] = float(r_O)
    else:
        descriptors['goldschmidt']['note'] = "Insufficient ionic radii to compute tolerance factor"

    # Octahedral factor(s)
    descriptors['octahedral'] = {}
    if B_radii:
        octs = {}
        for i, b_el in enumerate(B_elements):
            rB = B_radii[i] if i < len(B_radii) else None
            if rB is not None:
                octs[b_el] = float(octahedral_factor(rB, r_O))
        descriptors['octahedral'] = octs
    else:
        descriptors['octahedral']['note'] = "No B radii available"

    # Polyhedral distortion per B-site
    poly = {}
    for idx, bl in bonds.items():
        el = atoms.get_chemical_symbols()[idx]
        stats = compute_polyhedral_distortion(bl)
        if stats:
            poly[idx] = {
                'element': el,
                'coordination_count': len(bl),
                'mean_B_O': stats[0],
                'std_B_O': stats[1],
                'distortion_index': stats[2]
            }
    descriptors['polyhedral_distortion'] = poly

    # Bond Valence Sums (requires either bv_params or fallback)
    bv_params = read_bv_params(bv_params_path) if bv_params_path else None
    bvs = {}
    # Try to get oxidation states via pymatgen (fast but can fail); use as input to BVS
    # create pymatgen Structure (cart coords)
    try:
        pg_struct = Structure(atoms.get_cell(), atoms.get_chemical_symbols(), atoms.get_positions(), coords_are_cartesian=True)
        oxi_states = None
        try:
            oxi_states = pg_struct.oxi_state_guesses()
        except Exception:
            # fallback to guess_oxidation_states_from_composition (less precise)
            try:
                from pymatgen.analysis.bond_valence import BVAnalyzer
                # try assign oxidation states via BVAnalyzer if available
                analyzer = BVAnalyzer()
                pg_assigned = analyzer.analyze_structure(pg_struct)
                oxi_states = [site.specie.oxi_state for site in pg_assigned]
            except Exception:
                oxi_states = None
    except Exception:
        pg_struct = None
        oxi_states = None

    # If oxi_states is a list of floats for each site, use them to compute BVS per B-site
    for idx, bl in bonds.items():
        el = atoms.get_chemical_symbols()[idx]
        ox = None
        if oxi_states and idx < len(oxi_states):
            try:
                ox = int(round(oxi_states[idx]))
            except Exception:
                ox = None
        # fallback: estimate typical oxidation state: use pymatgen composition common oxidation states
        if ox is None:
            try:
                comp = Composition(atoms.get_chemical_formula())
                # get symbolic oxidation states by heuristic: use element's common oxidation states if available
                from pymatgen.core.periodic_table import Element
                el_obj = Element(el)
                # pick most common positive oxidation state
                candidates = [o for o in el_obj.common_oxidation_states if o > 0]
                ox = int(candidates[0]) if candidates else 2
            except Exception:
                ox = 2
        bvs_val = bond_valence_sum_for_site(bl, el, ox, bv_params=bv_params)
        bvs[idx] = {'element': el, 'oxidation_state': int(ox), 'bvs': bvs_val, 'coordination_count': len(bl)}

    descriptors['bond_valence_sums'] = bvs

    return descriptors

# ---------------------------------------------------------------------------
# Adaptive phonon resolution control
# ---------------------------------------------------------------------------

def choose_supercell(atoms, min_length=10.0, max_reps=5):
    """
    Choose a supercell such that each lattice direction exceeds min_length Å.
    Prevents overly large supercells via max_reps.
    """
    a, b, c = atoms.cell.lengths()
    reps = [max(1, min(max_reps, int(np.ceil(min_length / L)))) for L in (a, b, c)]
    print(f"[AUTO] Selected supercell {tuple(reps)} (min_length={min_length:.1f} Å)")
    return tuple(reps)



def summarize_final_verdict(thermal_results, delta_results, freqs_ev=None):
    """Extract a consistent, ML-ready verdict from phonon and thermal data."""
    verdict = {}

    # ---- helper: fuzzy key lookup ----
    def find_key(d, substrs):
        if not d:
            return None
        for k, v in d.items():
            key = k.lower()
            if all(s in key for s in substrs):
                try:
                    return float(v)
                except Exception:
                    pass
        return None

    # ---- extract values (handle your key style: free_energy_helmholtz, G_form_300K, etc.) ----
    helmholtz = find_key(thermal_results, ["helmholtz"])
    zpe = find_key(thermal_results, ["zero"])
    gibbs_form = find_key(thermal_results, ["form"]) or find_key(thermal_results, ["g_form"])
    gibbs_hull = find_key(thermal_results, ["hull"]) or find_key(thermal_results, ["g_hull"])

    # ---- thermodynamic stability ----
    if gibbs_hull is not None:
        if gibbs_hull < 0.02:
            thermo_state = "HIGHLY STABLE"
        elif gibbs_hull < 0.05:
            thermo_state = "MARGINALLY STABLE"
        else:
            thermo_state = "METASTABLE"
    else:
        thermo_state = "UNKNOWN"

    # ---- dynamical stability ----
    min_imag = None
    if freqs_ev is not None and len(freqs_ev) > 0:
        min_imag = float(np.min(freqs_ev))
    elif delta_results:
        try:
            min_imag = float(min(d.get("min_freq", 0) for d in delta_results))
        except Exception:
            min_imag = None

    if min_imag is None:
        dyn_state = "UNKNOWN"
    elif min_imag < -0.005:
        dyn_state = "DYNAMICALLY UNSTABLE"
    elif min_imag < 0:
        dyn_state = "POSSIBLE PHYSICAL INSTABILITY"
    else:
        dyn_state = "DYNAMICALLY STABLE"

    # ---- compose summary ----
    verdict["Helmholtz_Free_Energy_eV_per_atom"] = helmholtz
    verdict["Zero_Point_Energy_eV_per_atom"] = zpe
    verdict["Gibbs_Formation_Energy_eV_per_atom"] = gibbs_form
    verdict["Gibbs_Energy_Above_Hull_eV_per_atom"] = gibbs_hull
    verdict["Thermodynamic_Stability"] = thermo_state
    verdict["Smallest_Imaginary_Frequency_eV"] = min_imag
    verdict["Dynamic_Stability"] = dyn_state
    verdict["Overall_Assessment"] = (
        "REQUIRES FURTHER INVESTIGATION"
        if "POSSIBLE" in dyn_state or "UNSTABLE" in dyn_state
        else "STABLE"
    )
    return verdict


# ---------- JSON save helper (enhanced, integrates descriptors) ----------
def save_phonon_results_to_json(mp_id, atoms, formation_energy, e_hull, thermal_results,
                                delta_results, analysis_results, cache_dir, output_dir="./phonon_results",
                                structural_descriptors=None):
    import json
    from datetime import datetime
    from pathlib import Path
    from thermal import get_qmesh_from_cell
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    structure_info = {
        'mp_id': mp_id,
        'formula': atoms.get_chemical_formula(),
        'lattice_parameters': [float(x) for x in atoms.cell.lengths()],
        'lattice_angles': [float(x) for x in atoms.cell.angles()],
        'volume': float(atoms.get_volume()),
        'num_atoms': len(atoms)
    }

    qmesh = get_qmesh_from_cell(atoms.cell)

    thermodynamic_info = {
        'formation_energy_per_atom': float(formation_energy),
        'energy_above_hull': float(e_hull),
        'analysis_timestamp': datetime.now().isoformat(),
        'qmesh': qmesh
    }
    cache_info = {'cache_directory': cache_dir, 'cache_exists': Path(cache_dir).exists() if cache_dir else False}
    thermal_info = {}
    if thermal_results:
        for k,v in thermal_results.items():
            if isinstance(v, (int,float)):
                thermal_info[k] = float(v)
            else:
                thermal_info[k] = str(v)
    delta_info = []
    if delta_results:
        for d in delta_results:
            delta_info.append({
                'delta': float(d.get('delta', 0)),
                'min_frequency_ev': float(d.get('min_freq', 0)),
                'min_frequency_cm': float(d.get('min_freq_cm', d.get('min_frequency_cm', 0))),
                'trend': d.get('trend', '')
            })
    final = {
        'structure': structure_info,
        'thermodynamics': thermodynamic_info,
        'cache': cache_info,
        'thermal_properties': thermal_info,
        'delta_scan': delta_info,
        'analysis_summary': analysis_results if analysis_results else {},
        'structural_descriptors': structural_descriptors if structural_descriptors else {}
    }
    filename = output_path / f"phonon_analysis_{mp_id}.json"
    with open(filename, 'w') as f:
        json.dump(final, f, indent=2, default=str)
    print(f"💾 Results saved to: {filename}")
    return filename

# ---------- Main workflow ----------
def main():
    parser = argparse.ArgumentParser(description="Phonon stability via MatterSim or CHGNet (refactored)")
    parser.add_argument("--mp-id", default="mp-1391671")
    parser.add_argument("--api-key", default="IDScwWdmFrlYPrCFBI31h23dSNlaIEvE")
    parser.add_argument("--supercell", default="3,3,3")
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--delta-scan", action="store_true", help="Run comprehensive Δ-scan with analysis")
    parser.add_argument("--tight-relax", action="store_true", help="Use tighter relaxation (fmax=0.005)")
    parser.add_argument("--full-analysis", action="store_true", help="Run comprehensive numerical stability analysis")
    parser.add_argument("--output-dir", default="./phonon_results")
    parser.add_argument("--cache-dir", default="./phonon_cache")
    parser.add_argument("--cleanup-cache", action="store_true")
    # MatterSim specific
    parser.add_argument("--use-mattersim", action="store_true", help="Prefer MatterSim (if available)")
    #parser.add_argument("--mattersim-device", default="cpu", choices=["auto","cpu","cuda"])
    parser.add_argument("--mattersim-model", default="5M", choices=["1M","5M"])
    # Structural descriptors options
    parser.add_argument("--compute-structural-indicators", action="store_true",
                        help="Compute Goldschmidt / Octahedral / BVS / Polyhedral Distortion")
    parser.add_argument("--bv-params", default="/scratch_drive/anibal/DFT_out_of_nothing/data/bv_params.json",
                        help="Optional JSON file with bond-valence parameters (R0,b) for pairs")
    parser.add_argument("--double-perovskite-mode", default="mean", choices=["mean","separate"],
                        help="How to treat B-site radii for double perovskites")
    args = parser.parse_args()

    
    cache_dir = None

    try:
        # 1. Fetch structure
        struct, formation_energy, e_hull = get_structure_from_mp(args.mp_id, args.api_key)
        atoms = ase_from_pmg(struct)
        print(f"\nMaterial: {struct.formula}")
        print(f"Formation Energy: {formation_energy:.4f} eV/atom")
        print(f"Energy Above Hull: {e_hull:.4f} eV/atom")

        # 2. Calculator selection (prefer MatterSim if requested and available)
        calc = None
        if args.use_mattersim and MATTERSIM_AVAILABLE:
            calc = create_mattersim_calculator(device="cpu", model_size=args.mattersim_model)
        else:
            # If MatterSim requested but not available, warn and fallback
            if args.use_mattersim and not MATTERSIM_AVAILABLE:
                warnings.warn("MatterSim requested but not available; falling back to CHGNet if installed.")
            if CHGNET_AVAILABLE:
                calc = create_chgnet_calculator(device='cpu')
            else:
                raise RuntimeError("No calculator available: install mattersim or chgnet.")

        atoms.calc = calc

        # 3. Relaxation
        print("\n--- Relaxing geometry with calculator ---")
        fmax = 0.005 if args.tight_relax else 0.01
        print(f"Using fmax = {fmax} for relaxation")
        relax = BFGS(atoms, logfile=None)
        relax.run(fmax=fmax)
        print("✅ Geometry relaxed.\n")

        # 4. Phonons with isolated cache
        if args.supercell:
            supercell = tuple(int(x) for x in args.supercell.split(","))
        else:
            supercell = choose_supercell(atoms, min_length=10.0)


        print(f"Supercell: {supercell}, Displacement: {args.delta}")
        ph = IsolatedPhonons(atoms=atoms, calculator=calc, mp_id=args.mp_id, supercell=supercell, delta=args.delta, base_cache_dir=args.cache_dir)
        cache_dir = ph.cache_dir
        ph.run()
        print("✅ Displacement calculations completed with isolated cache")

        # 5. Post-processing (manual)
        out = try_manual_phonon_processing(ph)
        if out is None:
            print("❌ Incomplete displacement data — cannot continue.")
            return
        D_q, w, v = out

        freqs_ev = None

        # 6. Analysis: either full or basic
        delta_results = None
        analysis_results = None
        if args.full_analysis or args.delta_scan:
            # reuse analyze_numerical_stability from your MLIP script if present (we import it if module provided),
            # else perform a smaller delta-scan loop here
            try:
                # Try to call existing comprehensive analysis if it exists in environment (keeps original behaviour)
                from utils.run_phonon_MLIP_patched import analyze_numerical_stability  # try to reuse original if present
                analysis_results = analyze_numerical_stability(ph, atoms, formation_energy, e_hull)
                delta_results = analysis_results
            except Exception:
                # simple Δ-scan fallback (keeps behaviour but minimal)
                print("Running fallback Δ-scan (simpler) ...")
                delta_values = [0.0025, 0.005, 0.01, 0.02]
                delta_results = []
                eV_to_cm = 8065.54429
                for delta in delta_values:
                    try:
                        # Reuse IsolatedPhonons to ensure consistent cache directory
                        ph_delta = IsolatedPhonons(
                            atoms=atoms,
                            calculator=atoms.calc,
                            mp_id=args.mp_id,
                            supercell=ph.supercell,
                            delta=delta,
                            base_cache_dir=args.cache_dir
                        )

                        print(f"Reading/using cached force constants for Δ={delta} ...")
                        ph_delta.run()

                        D_q_delta, w_delta, v_delta = try_manual_phonon_processing(ph_delta)
                        freqs_ev = analyze_gamma_point(D_q_delta)
                        min_freq = float(np.min(freqs_ev))
                        delta_results.append({
                            'delta': delta,
                            'min_freq': min_freq,
                            'min_freq_cm': min_freq * eV_to_cm,
                            'trend': ''
                        })
                    except Exception as e:
                        print(f"Δ={delta} failed in IsolatedPhonons context: {e}")
                analysis_results = {'delta_scan': delta_results}

        else:
            # Basic checks
            check_displacement_cache(ph)
            check_acoustic_modes(ph)
            mode_vector = analyze_imaginary_mode(D_q, atoms)
            freqs_ev = analyze_gamma_point(D_q)
            neg = np.sum(freqs_ev < 0)
            print(f"Imaginary modes at Gamma: {neg}")
            eV_to_cm = 8065.54429
            print("\nSix smallest frequencies (eV and cm^-1):")
            sorted_freqs = sorted(freqs_ev, key=abs)[:6]
            for i, f in enumerate(sorted_freqs):
                print(f"  {i}: {f:.6e} eV  {f*eV_to_cm:.2f} cm^-1")

        # 7. Thermal properties (reuse your thermal module pattern)
        try:
            from thermal import calculate_thermal_properties_ase, calculate_thermal_properties_manual
        except Exception:
            calculate_thermal_properties_ase = None
            calculate_thermal_properties_manual = None

        print("\n--- THERMAL PROPERTIES ---")
        thermal_results = None
        if calculate_thermal_properties_ase:
            try:
                thermal_ase = calculate_thermal_properties_ase(phonons=ph, atoms=atoms, formation_energy=formation_energy, e_hull=e_hull, temperature=300.0)
                if thermal_ase is not None:
                    thermal_results = thermal_ase
                    print("Thermal properties computed via ASE CrystalThermo.")
            except Exception as e:
                print("ASE thermal method failed:", e)
        if thermal_results is None and calculate_thermal_properties_manual:
            try:
                thermal_manual = calculate_thermal_properties_manual(phonons=ph, atoms=atoms, formation_energy=formation_energy, e_hull=e_hull, temperature=300.0, method='harmonic')
                thermal_results = thermal_manual
            except Exception as e:
                print("Manual thermal method failed:", e)

        # 8. Structural indicators (optional)
        structural_descriptors = None
        if args.compute_structural_indicators:
            print("\n--- Computing structural indicators (Goldschmidt, Octahedral, BVS, Polyhedral distortion) ---")
            structural_descriptors = compute_structural_descriptors(args.mp_id, atoms, double_perov_mode=args.double_perovskite_mode, bv_params_path=args.bv_params)
            print("Structural descriptors computed.")

        # 9. Save results to JSON (includes descriptors)
        freqs_ev = locals().get("freqs_ev", None)
        final_verdict = summarize_final_verdict(thermal_results, delta_results, freqs_ev)

        json_filename = save_phonon_results_to_json(
            mp_id=args.mp_id,
            atoms=atoms,
            formation_energy=formation_energy,
            e_hull=e_hull,
            thermal_results=thermal_results,
            delta_results=delta_results,
            analysis_results=final_verdict,
            cache_dir=cache_dir,
            output_dir=args.output_dir,
            structural_descriptors=structural_descriptors
        )

    except Exception as e:
        print(f"❌ Error processing {args.mp_id}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if args.cleanup_cache and cache_dir:
            print(f"🧹 Cleaning up cache directory: {cache_dir}")
            cleanup_phonon_cache(args.mp_id, args.cache_dir)

if __name__ == "__main__":
    main()
