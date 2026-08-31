import numpy as np
import ase.units as units
from ase.thermochemistry import CrystalThermo
import traceback


def get_qmesh_from_cell(cell, density=1.0, max_points=40):
    """
    Choose a q-mesh to keep a roughly constant reciprocal-space sampling density.
    density: approximate number of q-points per Å⁻¹.
    max_points: maximum number of divisions per reciprocal vector.
    """
    rec = 2 * np.pi * np.linalg.inv(cell).T
    rec_lengths = np.linalg.norm(rec, axis=1)  # |b1|, |b2|, |b3|
    qmesh = [max(1, min(max_points, int(np.ceil(density * L)))) for L in rec_lengths]
    print(f"[AUTO] Selected q-mesh {tuple(qmesh)} (density={density:.2f} Å⁻¹)")
    return tuple(qmesh)


def calculate_thermal_properties_ase(phonons, atoms, formation_energy, e_hull, temperature=300):
    """Calculate thermal properties using ASE CrystalThermo."""
    print("\n--- Thermal Properties (ASE CrystalThermo) ---")

    qmesh = get_qmesh_from_cell(atoms.cell)
    
    try:
        # Get phonon density of states
        dos = phonons.get_dos(kpts=qmesh).sample_grid(npts=500, width=1e-3)
        phonon_energies = dos.get_energies()
        phonon_DOS = dos.get_weights()
        
        # Filter out any problematic energies
        valid_mask = (phonon_energies > 1e-6) & (phonon_DOS > 0)
        if np.sum(valid_mask) == 0:
            print("❌ No valid phonon energies for ASE calculation")
            return None
            
        phonon_energies = phonon_energies[valid_mask]
        phonon_DOS = phonon_DOS[valid_mask]
        
        # Create CrystalThermo object with ZERO potential energy
        # We only want the vibrational contributions, not the total energy
        thermo = CrystalThermo(
            phonon_energies=phonon_energies,
            phonon_DOS=phonon_DOS,
            potentialenergy=0.0,  # CRITICAL FIX: Use 0 to get only vibrational contributions
            formula_units=1
        )
        
        # Get Helmholtz free energy at temperature (vibrational only)
        F_vib = thermo.get_helmholtz_energy(temperature=temperature)
        
        # Check for NaN results
        if np.isnan(F_vib):
            print("❌ ASE calculation resulted in NaN values")
            return None
        
        # Convert to per-atom basis
        n_atoms = len(atoms)
        F_vib_per_atom = F_vib / n_atoms
        
        # Calculate Gibbs free energy contributions
        # Only add vibrational contribution to formation energy
        G_form_300K = formation_energy + F_vib_per_atom
        G_hull_300K = e_hull + F_vib_per_atom
        
        # Calculate zero-point energy manually
        zpe = 0.5 * np.sum(phonon_energies * phonon_DOS) / np.sum(phonon_DOS) / n_atoms
        
        thermal_info = {
            'free_energy_helmholtz': float(F_vib_per_atom),
            'zero_point_energy': float(zpe),
            'G_form_300K': float(G_form_300K),
            'G_hull_300K': float(G_hull_300K),
            'temperature': temperature
        }
        
        print(f"Helmholtz Free Energy ({temperature}K): {F_vib_per_atom:.6f} eV/atom")
        print(f"Zero-Point Energy: {zpe:.6f} eV/atom")
        print(f"Gibbs Formation Energy ({temperature}K): {G_form_300K:.6f} eV/atom")
        print(f"Gibbs Energy Above Hull ({temperature}K): {G_hull_300K:.6f} eV/atom")
        
        return thermal_info
        
    except Exception as e:
        print(f"❌ Error in ASE thermal properties: {e}")
        return None


def calculate_thermal_properties_manual(phonons, atoms, formation_energy, e_hull, temperature=300, 
                                      mesh_density=4, method='harmonic'):
    """Calculate thermal properties manually with method selector."""
    print(f"\n--- Thermal Properties (Manual - {method}) ---")
    
    try:
        # Method selector for different approximations
        if method == 'harmonic':
            return _harmonic_thermal_properties(phonons, atoms, formation_energy, e_hull, temperature, mesh_density)
        elif method == 'quasiharmonic':
            return _quasiharmonic_thermal_properties(phonons, atoms, formation_energy, e_hull, temperature, mesh_density)
        else:
            print(f"Unknown method: {method}, using harmonic")
            return _harmonic_thermal_properties(phonons, atoms, formation_energy, e_hull, temperature, mesh_density)
            
    except Exception as e:
        print(f"❌ Error in manual thermal properties: {e}")
        traceback.print_exc()
        return None

def _harmonic_thermal_properties(phonons, atoms, formation_energy, e_hull, temperature, mesh_density):
    """Harmonic approximation for thermal properties."""
    
    # Generate q-point mesh
    
    
    qmesh = get_qmesh_from_cell(atoms.cell, density=mesh_density, max_points=40)

    kpts = [(x/qmesh[0], y/qmesh[1], z/qmesh[2]) 
            for x in range(qmesh[0]) 
            for y in range(qmesh[1]) 
            for z in range(qmesh[2])]

    
    all_frequencies = []
    
    # Collect frequencies over Brillouin zone
    for q in kpts:
        freqs = calculate_phonon_frequencies(phonons, q)
        if freqs is not None:
            all_frequencies.extend(freqs)
    
    if not all_frequencies:
        print("❌ Could not calculate phonon frequencies")
        return None
    
    all_frequencies = np.array(all_frequencies)
    positive_freqs = all_frequencies[all_frequencies > 1e-6]
    
    if len(positive_freqs) == 0:
        print("❌ No positive frequencies for thermal properties")
        return None
    
    n_atoms = len(atoms)
    n_qpoints = len(kpts)
    kT = units.kB * temperature  # eV
    beta = 1.0 / kT
    
    # Initialize thermodynamic quantities
    zpe_total = 0.0
    F_vib_total = 0.0
    entropy_total = 0.0
    cv_total = 0.0
    
    # Calculate per mode
    for freq in positive_freqs:
        if freq > 1e-6:  # Only positive frequencies
            x = freq * beta
            
            # Zero-point energy
            zpe_total += 0.5 * freq
            
            # Vibrational free energy: F_vib = kT * ln[2 * sinh(hω/2kT)]
            # For harmonic oscillator: F_vib = 0.5ħω + kT ln(1 - exp(-ħω/kT))
            F_vib_total += (0.5 * freq + kT * np.log(1 - np.exp(-x)))
            
            # Entropy: S = -∂F/∂T
            if x < 1e-10:
                s_term = 0.0
            else:
                s_term = (x / (np.exp(x) - 1)) - np.log(1 - np.exp(-x))
            entropy_total += units.kB * s_term
            
            # Heat capacity at constant volume
            if x < 1e-10:
                cv_term = 0.0
            else:
                cv_term = x**2 * np.exp(x) / (np.exp(x) - 1)**2
            cv_total += units.kB * cv_term
    
    # Normalize by number of atoms and q-points
    zpe_per_atom = zpe_total / (n_atoms * n_qpoints)
    F_vib_per_atom = F_vib_total / (n_atoms * n_qpoints)
    entropy_per_atom = entropy_total / (n_atoms * n_qpoints)
    cv_per_atom = cv_total / (n_atoms * n_qpoints)
    
    # Calculate Gibbs-like quantities (constant volume approximation)
    G_form_300K = formation_energy + F_vib_per_atom
    G_hull_300K = e_hull + F_vib_per_atom
    
    thermal_info = {
        'free_energy_helmholtz': float(F_vib_per_atom),
        'zero_point_energy': float(zpe_per_atom),
        'entropy': float(entropy_per_atom),
        'heat_capacity': float(cv_per_atom),
        'G_form_300K': float(G_form_300K),
        'G_hull_300K': float(G_hull_300K),
        'temperature': temperature,
        'method': 'harmonic',
        'n_qpoints': n_qpoints,
        'n_phonon_modes': len(positive_freqs)
    }
    
    print(f"Helmholtz Free Energy ({temperature}K): {F_vib_per_atom:.6f} eV/atom")
    print(f"Zero-Point Energy: {zpe_per_atom:.6f} eV/atom")
    print(f"Entropy: {entropy_per_atom:.6e} eV/(K·atom)")
    print(f"Heat Capacity: {cv_per_atom/units.kB:.6f} k_B/atom")
    print(f"Gibbs Formation Energy ({temperature}K): {G_form_300K:.6f} eV/atom")
    print(f"Gibbs Energy Above Hull ({temperature}K): {G_hull_300K:.6f} eV/atom")
    print(f"Q-points used: {n_qpoints}, Phonon modes: {len(positive_freqs)}")
    
    return thermal_info

def _quasiharmonic_thermal_properties(phonons, atoms, formation_energy, e_hull, temperature, mesh_density):
    """Quasiharmonic approximation (placeholder for volume dependence)."""
    print("⚠️ Quasiharmonic requires volume dependence - using harmonic")
    return _harmonic_thermal_properties(phonons, atoms, formation_energy, e_hull, temperature, mesh_density)

def calculate_phonon_frequencies(phonons, q_point=[0, 0, 0]):
    """Calculate phonon frequencies at a given q-point manually."""
    try:
        # Get the dynamical matrix at q-point
        D_q = compute_dynamical_matrix(phonons, q_point)
        
        # Diagonalize to get eigenvalues (squared frequencies)
        eigenvalues = np.linalg.eigvalsh(D_q)
        
        # Convert to frequencies (taking square root of eigenvalues)
        # Handle negative eigenvalues (imaginary frequencies)
        frequencies = np.sqrt(np.abs(eigenvalues)) * np.sign(eigenvalues)
        
        # Convert from sqrt(eV/Å²/amu) to eV
        # The conversion factor is: hbar * 1e10 / sqrt(e * amu)
        s = units._hbar * 1e10 / np.sqrt(units._e * units._amu)
        frequencies_ev = frequencies * s
        
        return frequencies_ev
        
    except Exception as e:
        print(f"Error calculating frequencies: {e}")
        return None

def compute_dynamical_matrix(phonons, q_scaled):
    """Computation of the dynamical matrix in momentum space D_ab(q)."""
    try:
        # Evaluate fourier sum
        R_cN = phonons._lattice_vectors_array
        phase_N = np.exp(-2.j * np.pi * np.dot(q_scaled, R_cN))
        D_q = np.sum(phase_N[:, np.newaxis, np.newaxis] * phonons.D_N, axis=0)
        return D_q
    except:
        # Fallback: use Gamma point only (q=0)
        return np.sum(phonons.D_N, axis=0)
