# Phonons: Phonon stability and thermal analysis (MatterSim / CHGNet)

## source ~/envs/tblite312/bin/activate

python run_batch_phonons_analysis.py   --csv /scratch_drive/anibal/DFT_out_of_nothing/DoublePerovskitesScreening/double_perovskites_results.csv  --script ./run_phonon_MATERSIM.py   --outdir ./batch_results   --extra-args --use-mattersim --full-analysis --compute-structural-indicators --tight-relax


This folder contains scripts and helpers to perform phonon calculations, numerical stability analysis, thermal-property estimation and structural descriptors for perovskite-like materials. The main entrypoint is `run_phonon_MATERSIM.py` which was refactored to prefer the MatterSim potential (if available) and fall back to CHGNet.

## Key capabilities

- Fetch crystal structures from the Materials Project using a Material ID (via `MPRester`).
- Convert Pymatgen `Structure` to ASE `Atoms` and relax geometry using an ML interatomic potential (MatterSim preferred, CHGNet fallback).
- Create an isolated phonon displacement cache per-material to avoid collisions between runs.
- Run finite-displacement phonon calculations with ASE `Phonons` and postprocess force constants.
- Manual and programmatic analysis of dynamical matrices, including Γ-point diagonalization, identification of imaginary modes and eigenvector inspection.
- Δ-scan analysis (varying displacement amplitude) to detect numerical artifacts in low-frequency imaginary modes.
- Thermal-property estimation using ASE `CrystalThermo` (`calculate_thermal_properties_ase`) and a manual harmonic approximation implementation (`calculate_thermal_properties_manual`).
- Structural descriptors for perovskites: Goldschmidt Tolerance Factor, Octahedral Factor, Bond Valence Sums (BVS, optional parameters file), and simple polyhedral distortion metrics.
- Save a consolidated JSON report with structure, thermodynamics, delta-scan results, analysis summaries and structural descriptors.

## Files of interest

- `run_phonon_MATERSIM.py` — main CLI script. Implements the full workflow: fetch → relax → phonons → analysis → thermal → descriptors → save JSON.
- `thermal.py` — thermal property calculations. Exposes:
  - `calculate_thermal_properties_ase(phonons, atoms, formation_energy, e_hull, temperature=300)`
  - `calculate_thermal_properties_manual(phonons, atoms, formation_energy, e_hull, temperature=300, mesh_density=4, method='harmonic')`
  - utility functions for computing phonon frequencies and dynamical matrices.
- `raddi_helper.py` — helper to read ionic radii from `data/shannon-radii.json` and expose `get_atomic_property(element, charge, coordination, property_name)`.
- `phonon_cache/` — created runtime cache directory for per-material displacement calculations.
- `phonon_results/` — default output directory for JSON summary files.

## Main CLI usage (script: `run_phonon_MATERSIM.py`)

Basic invocation (uses CHGNet if MatterSim is not installed):

```bash
python3 phonons/run_phonon_MATERSIM.py --mp-id mp-1391671 --api-key <MATERIALS_PROJECT_API_KEY>
```

Common arguments:

- `--mp-id` : Materials Project ID (default `mp-1391671`).
- `--api-key` : Materials Project API key (default embedded in script for convenience in tests).
- `--supercell` : Supercell size as comma-separated integers, e.g., `3,3,3`.
- `--delta` : Displacement amplitude in Å (default `0.01`).
- `--delta-scan` : Run the Δ-scan analysis (multiple delta values).
- `--tight-relax` : Use tighter geometry relaxation (fmax=0.005 Å).
- `--full-analysis` : Run comprehensive numerical stability analysis (more diagnostics + Δ-scan).
- `--output-dir` : Where to save JSON results (default `./phonon_results`).
- `--cache-dir` : Base cache dir for phonon displacement jobs (default `./phonon_cache`).
- `--cleanup-cache` : Remove the isolated phonon cache after the run.
- `--use-mattersim` : Prefer MatterSim calculator (if installed). Default is fallback behavior.
- `--mattersim-device` : `auto`, `cpu`, or `cuda` for MatterSim device selection.
- `--mattersim-model` : `1M` or `5M` pre-trained Mattersim model sizes (script uses specific checkpoint names).
- `--compute-structural-indicators` : Compute Goldschmidt, Octahedral factor, BVS, and polyhedral distortion metrics.
- `--bv-params` : Path to `bv_params.json` (bond-valence parameters) for BVS calculations (default `data/bv_params.json`).
- `--double-perovskite-mode` : `mean` or `separate` (how to treat two B-site radii).

Examples:

- Minimal (fetch + relax + phonons + basic checks):
  python3 phonons/run_phonon_MATERSIM.py --mp-id mp-1391671 --api-key $MP_KEY

- Full analysis with structural descriptors and MatterSim (if installed):
  python3 phonons/run_phonon_MATERSIM.py --mp-id mp-1391671 --api-key $MP_KEY --use-mattersim --mattersim-device auto --full-analysis --compute-structural-indicators

- Run Δ-scan only:
  python3 phonons/run_phonon_MATERSIM.py --mp-id mp-1391671 --api-key $MP_KEY --delta-scan

## Output

On successful execution the script writes a JSON file to `./phonon_results/phonon_analysis_<mp-id>.json` containing:

- `structure` : formula, lattice params, volume, number of atoms
- `thermodynamics` : formation energy, energy above hull, timestamps
- `thermal_properties` : Helmholtz free energy per atom, zero-point energy, Gibbs formation/hull at 300K, etc.
- `delta_scan` : list of delta runs and the minimal Γ frequency for each
- `analysis_summary` : compact verdict (`Thermodynamic_Stability`, `Dynamic_Stability`, `Overall_Assessment`) and key numeric fields
- `structural_descriptors` : if requested, includes Goldschmidt tolerance factor, octahedral factors, BVS per-site (if params provided) and polyhedral distortion metrics

## Internal methods and what they do

- `get_structure_from_mp(mp_id, api_key)`
  - Fetches an MP summary (includes structure) and returns a Pymatgen `Structure`, formation energy per atom and energy above hull.

- `ase_from_pmg(struct)`
  - Converts a Pymatgen `Structure` to ASE `Atoms` (cartesian positions, pbc True).

- Calculator constructors:
  - `create_mattersim_calculator(device='auto', model_size='1M')` – create a MatterSim calculator (if installed).
  - `create_chgnet_calculator(device='cpu')` – create CHGNet calculator (fallback if MatterSim not available).

- Phonon helpers:
  - `IsolatedPhonons` (subclass of ASE `Phonons`) – creates a per-material cache dir and runs phonon displacements there.
  - `try_manual_phonon_processing(ph)` – reads cached force constants and computes the Γ dynamical matrix and eigenvalues.
  - `compute_dynamical_matrix_from_fc(ph, fc, q=[0,0,0])` – assemble mass-normalized dynamical matrix from force-constants array.
  - `analyze_gamma_point(D_q)` – convert eigenvalues to frequencies and report small/imaginary modes.
  - `analyze_imaginary_mode(D_q, atoms)` – report per-atom participation in the most negative eigenmode.
  - `check_displacement_cache(ph)` – validate phonon cache presence and list displacement directories.
  - `check_acoustic_modes(ph)` – basic acoustic mode diagnostics (translational invariance check).

- Structural helpers:
  - `detect_b_sites(atoms)` – heuristic detection of B-site cations by counting nearby O neighbors using KD-tree.
  - `compute_B_O_bond_lengths(atoms, b_indices)` – return lists of B–O bond lengths for each B-site.
  - `goldschmidt_tolerance(r_A, r_B, r_O=1.35)` – Goldschmidt tolerance factor.
  - `octahedral_factor(r_B, r_O=1.35)` – simple octahedral factor.
  - `compute_polyhedral_distortion(bond_lengths)` – distortion index of B–O bond lengths.
  - `bond_valence_sum_for_site(bonds_list, cation_symbol, oxidation_state, bv_params=None)` – computes BVS using R0/b pairs or falls back to a coordination-based estimate.
  - `compute_structural_descriptors(mp_id, atoms, double_perov_mode='mean', bv_params_path=None)` – orchestrates structure descriptor calculations and packages results into a dict.

## Dependencies

- Python 3.10+ (scripts use features from modern ASE and pymatgen)
- ase
- pymatgen
- numpy
- scipy (for KD-tree neighbor searches)
- chgnet (optional fallback calculator)
- mattersim (optional preferred calculator)
- Optional but useful: torch (for mattersim device selection), pymatgen.analysis.bond_valence for BVAnalyzer

Install minimal requirements (example):

```bash
pip install ase pymatgen numpy scipy
# optional
pip install torch chgnet mattersim
```

## Configuration and data files

- `data/shannon-radii.json` — Shannon ionic radii used by `raddi_helper.get_atomic_property()`.
- `data/bv_params.json` — Optional bond-valence params used for BVS calculations. Example keys: `"Fe3+-O2-": {"R0": 1.759, "b": 0.37}`

## Troubleshooting

- Missing calculators: If neither MatterSim nor CHGNet are installed, the script will raise a RuntimeError. Install one of them or mock an ASE-compatible calculator.
- Shannon radii import errors: `raddi_helper` expects `data/shannon-radii.json` relative to the repo root. Ensure that file exists and is valid JSON.
- ASE CrystalThermo failures: The ASE DOS sampling may return NaNs if phonon DOS is pathological; fallback to `calculate_thermal_properties_manual` with `method='harmonic'`.
- Imaginary modes: small imaginary modes (< 30 cm^-1) that change with Δ indicate numerical artifacts. Use `--delta-scan` or `--full-analysis` to classify the cause.

## Notes and caveats

- The script uses some heuristic and fallback methods for ionic radii and oxidation states. These are not substitutes for careful chemical assignment.
- Bond Valence Sum calculations require a correctly formatted `bv_params.json` for reliable values; otherwise a simple estimate is provided.
- The script uses an embedded default MP API key for convenience in testing; replace it with your own key for production use.

## Next steps / suggestions

- Add unit tests for small helper functions (e.g., Goldschmidt, octahedral factor, detect_b_sites boundaries).
- Parameterize the MatterSim model checkpoint path and support downloading/checkpoint management.
- Improve B-site detection by using coordination or Voronoi analysis from pymatgen.
- Add an option to skip relaxation and use the raw MP structure (useful for reproducible comparisons).

---

Generated from source files: `phonons/run_phonon_MATERSIM.py`, `phonons/thermal.py`, and `phonons/raddi_helper.py`.
