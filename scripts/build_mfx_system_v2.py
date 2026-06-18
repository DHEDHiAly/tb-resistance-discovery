"""
Build complete gyrB + MFX OpenMM system manually.
Constructs all bonded and nonbonded forces for MFX via Python API.
"""
import openmm as mm
import openmm.app as app
import openmm.unit as u
from openmm.app import Modeller
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import math

# ========== 1. Build protein system ==========
print("Building protein system...")
pdb = app.PDBFile('data/pdb/alphafold/gyrB_P9WG45_alphafold.pdb')
modeller = Modeller(pdb.topology, pdb.positions)
ff_prot = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
modeller.addHydrogens(ff_prot)
print(f"  After H: {modeller.topology.getNumAtoms()} atoms")
modeller.addSolvent(ff_prot, boxSize=u.Quantity(np.array([8,8,8]), u.nanometer))
n_solvated = modeller.topology.getNumAtoms()
print(f"  Solvated: {n_solvated} atoms")

# Create the base system
system = ff_prot.createSystem(
    modeller.topology,
    nonbondedMethod=app.PME,
    nonbondedCutoff=1.0*u.nanometer,
    constraints=app.HBonds,
    rigidWater=True,
)

# Save references
n_prot_atoms = system.getNumParticles()
print(f"  System particles: {n_prot_atoms}")

# ========== 2. Prepare MFX from RDKit ==========
print("\nPreparing MFX ligand...")
mfx_smiles = 'CC1CN(CCN1C2=C(C=C3C(=C2OC)N(C(=O)C(=C3F)C(=O)O)CC4CC4)F)C'
mol = Chem.MolFromSmiles(mfx_smiles)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, AllChem.ETKDG())
AllChem.MMFFOptimizeMolecule(mol)
conf = mol.GetConformer(0)

# MMFF94 charges
props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94')
if props is None:
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')
charges = [props.GetMMFFPartialCharge(i) for i in range(mol.GetNumAtoms())]
charge_sum = sum(charges)
charges = [q - charge_sum/len(charges) for q in charges]

n_mfx = mol.GetNumAtoms()
print(f"  {n_mfx} atoms, total charge {sum(charges):.4f}")

# GAFF-like LJ parameters
gaff_lj = {
    'C': {'c3': (0.33997, 0.457730), 'ca': (0.33997, 0.457730), 'c2': (0.33997, 0.457730)},
    'N': {'n3': (0.32500, 0.711280), 'na': (0.32500, 0.711280)},
    'O': {'o': (0.29599, 0.878640), 'oh': (0.30664, 0.880474)},
    'F': {'f': (0.31181, 0.278236)},
    'H': {'hc': (0.25997, 0.065688)},
}

def get_gaff_type(atom):
    symb = atom.GetSymbol()
    if symb == 'H': return 'hc'
    if symb == 'F': return 'f'
    if symb == 'O':
        nbrs = [n.GetAtomicNum() for n in atom.GetNeighbors()]
        return 'oh' if 1 in nbrs else 'o'
    if symb == 'N':
        bo = sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())
        return 'na' if atom.GetIsAromatic() else ('n2' if bo > 3 else 'n3')
    if symb == 'C':
        if atom.GetIsAromatic(): return 'ca'
        bo = sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())
        return 'c2' if bo > 4 else 'c3'
    return 'c3'

# Get atom types and properties
mfx_types = []
mfx_lj = []
for i, atom in enumerate(mol.GetAtoms()):
    atype = get_gaff_type(atom)
    mfx_types.append(atype)
    elem = atom.GetSymbol()
    sigma, eps = gaff_lj[elem].get(atype, (0.34, 0.45773))
    mfx_lj.append((sigma, eps))

# ========== 3. Add MFX particles to the system ==========
print("Adding MFX particles to system...")

# Get MFX coordinates (aligned to docked pose)
# Load the docked pose for alignment
mfx_pdb = app.PDBFile('data/pdb/mfx_WT_pose.pdb')
mfx_dock_pos = list(mfx_pdb.positions)
dock_coords = np.array([[p[0]._value, p[1]._value, p[2]._value] for p in mfx_dock_pos])

# Align RDKit heavy atoms to docked pose
rdkit_coords = conf.GetPositions()
heavy_idx = [i for i, a in enumerate(mol.GetAtoms()) if a.GetAtomicNum() > 1]
heavy_rdkit = rdkit_coords[heavy_idx]
heavy_center = np.mean(heavy_rdkit, axis=0)
dock_center = np.mean(dock_coords, axis=0)
dock_c = dock_coords - dock_center
heavy_c = heavy_rdkit - heavy_center
H = heavy_c.T @ dock_c
U, S, Vt = np.linalg.svd(H)
R = Vt.T @ U.T
if np.linalg.det(R) < 0:
    Vt[-1,:] *= -1
    R = Vt.T @ U.T
transformed = (rdkit_coords - heavy_center) @ R.T + dock_center

# Create positions for MFX (in nm)
mfx_positions = [u.Quantity(np.array([transformed[i,0], transformed[i,1], transformed[i,2]]), u.nanometer) for i in range(n_mfx)]

# ========== 4. Add MFX particles to System ==========
print("Adding MFX particles to System...")
elem_mass = {'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'H': 1.008}
for i, atom in enumerate(mol.GetAtoms()):
    mass = elem_mass.get(atom.GetSymbol(), 12.0) * u.amu
    system.addParticle(mass)
print(f"  System particles now: {system.getNumParticles()}")

# ========== 5. Create forces for MFX ==========
print("Creating MFX bonded forces...")

# HarmonicBondForce
bf = mm.HarmonicBondForce()
for bond in mol.GetBonds():
    b1, b2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    t1, t2 = mfx_types[b1], mfx_types[b2]
    # Get MMFF parameters
    try:
        sp = props.GetMMFFBondStretchParams(mol, b1, b2)
        if sp:
            _, k_mdyn, r0_a = sp
            k_kcal = k_mdyn * 143.88
        else:
            k_kcal, r0_a = 300.0, 1.5
    except:
        k_kcal, r0_a = 300.0, 1.5
    # Convert: r0 from A to nm, k from kcal/(mol*A^2) to kJ/(mol*nm^2)
    r0_nm = r0_a / 10.0
    k_kj = k_kcal * 4.184 * 100.0
    p1 = n_prot_atoms + b1
    p2 = n_prot_atoms + b2
    bf.addBond(p1, p2, r0_nm, k_kj)
print(f"  HarmonicBondForce: {bf.getNumBonds()} bonds")

# HarmonicAngleForce
af = mm.HarmonicAngleForce()
for angle in mol.GetBonds():
    # Angles are identified by 3 consecutive atoms (1-2, 2-3, etc.)
    # We need to find all triplets of connected atoms
    pass

# Build angle list from topology
# For each central atom, find pairs of neighbors
from collections import defaultdict
adj = defaultdict(list)
for bond in mol.GetBonds():
    b1, b2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    adj[b1].append(b2)
    adj[b2].append(b1)

angles_found = set()
for center, neighbors in adj.items():
    for i in range(len(neighbors)):
        for j in range(i+1, len(neighbors)):
            a1, a2, a3 = neighbors[i], center, neighbors[j]
            key = tuple(sorted([a1, a3])) + (a2,)
            if key not in angles_found:
                angles_found.add(key)
                t1, t2, t3 = mfx_types[a1], mfx_types[a2], mfx_types[a3]
                # Generic angle parameter: 109.5 deg, k=60 kcal/mol
                angle_rad = 109.5 * math.pi / 180.0
                k_kj = 60.0 * 4.184
                p1 = n_prot_atoms + a1
                p2 = n_prot_atoms + a2
                p3 = n_prot_atoms + a3
                af.addAngle(p1, p2, p3, angle_rad, k_kj)
print(f"  HarmonicAngleForce: {af.getNumAngles()} angles")

# PeriodicTorsionForce
tf = mm.PeriodicTorsionForce()
# Find torsions (4 consecutive atoms following dihedral pattern)
# Simple approach: find all paths of length 4
torsions_found = set()
for a1 in adj:
    for a2 in adj[a1]:
        for a3 in adj[a2]:
            if a3 == a1: continue
            for a4 in adj[a3]:
                if a4 == a2: continue
                key = (a1, a2, a3, a4)
                if key not in torsions_found:
                    torsions_found.add(key)
                    # Generic torsion
                    t1, t2, t3, t4 = mfx_types[a1], mfx_types[a2], mfx_types[a3], mfx_types[a4]
                    if t2 == 'ca' and t3 == 'ca':
                        periodicity = 2
                        k_kcal = 3.5
                    elif (t2 == 'c3' and t3 == 'c3') or (t2 == 'n3' and t3 == 'c3'):
                        periodicity = 1 if t2 == 'n3' else 3
                        k_kcal = 0.5 if t2 == 'n3' else 0.15
                    else:
                        periodicity = 3
                        k_kcal = 0.15
                    k_kj = k_kcal * 4.184
                    p1 = n_prot_atoms + a1
                    p2 = n_prot_atoms + a2
                    p3 = n_prot_atoms + a3
                    p4 = n_prot_atoms + a4
                    tf.addTorsion(p1, p2, p3, p4, periodicity, 0.0, k_kj)
print(f"  PeriodicTorsionForce: {tf.getNumTorsions()} torsions")

# NonbondedForce - extend the existing one
nbf = None
for i in range(system.getNumForces()):
    f = system.getForce(i)
    if type(f).__name__ == 'NonbondedForce':
        nbf = f
        break

if nbf is not None:
    for i in range(n_mfx):
        q = charges[i] * u.elementary_charge
        s = mfx_lj[i][0] * u.nanometer
        e = mfx_lj[i][1] * u.kilojoule_per_mole
        nbf.addParticle(q, s, e)
    print(f"  NonbondedForce: {nbf.getNumParticles()} total particles")

# Add bonded forces to system (they reference atom indices correctly)
system.addForce(bf)
system.addForce(af)
system.addForce(tf)

# Verify particle count consistency
n_total = system.getNumParticles()
n_nb = nbf.getNumParticles()
assert n_total == n_nb, f"System has {n_total} particles but NonbondedForce has {n_nb}"
print(f"  Particle counts verified: {n_total}")

# Combine positions
all_positions = list(modeller.positions) + mfx_positions

# ========== 5. Test system ==========
print("\nTesting system...")
integrator = mm.LangevinIntegrator(300*u.kelvin, 1.0/u.picosecond, 0.002*u.picosecond)
platform = mm.Platform.getPlatformByName('CPU')
context = mm.Context(system, integrator, platform)
context.setPositions(all_positions)

print(f"  Total particles: {system.getNumParticles()}")
print(f"  Total forces: {system.getNumForces()}")
for i in range(system.getNumForces()):
    f = system.getForce(i)
    name = type(f).__name__
    nb = f.getNumBonds() if hasattr(f, 'getNumBonds') else 'N/A'
    na = f.getNumAngles() if hasattr(f, 'getNumAngles') else 'N/A'
    nt = f.getNumTorsions() if hasattr(f, 'getNumTorsions') else 'N/A'
    np_ = f.getNumParticles() if hasattr(f, 'getNumParticles') else 'N/A'
    print(f"  Force {i}: {name} bonds={nb} angles={na} torsions={nt} particles={np_}")

pe = context.getState(getEnergy=True).getPotentialEnergy()
print(f"  Initial PE: {pe}")

mm.LocalEnergyMinimizer.minimize(context, maxIterations=500)
pe_min = context.getState(getEnergy=True).getPotentialEnergy()
print(f"  After 500 steps min: {pe_min}")

# Check for NaN
pos_check = context.getState(getPositions=True).getPositions(asNumpy=True)
print(f"  NaN positions: {sum(1 for p in pos_check if math.isnan(p[0]))}")

if pe_min < 1e7:  # reasonable energy
    print("\nSUCCESS! System is stable.")
else:
    print("\nEnergy still high - may need more minimization or parameter tuning")
    print("Trying 5000 more steps...")
    mm.LocalEnergyMinimizer.minimize(context, maxIterations=5000)
    pe_min2 = context.getState(getEnergy=True).getPotentialEnergy()
    print(f"  After 5500 steps min: {pe_min2}")

context.setVelocitiesToTemperature(300*u.kelvin)
integrator.step(100)
pe_final = context.getState(getEnergy=True).getPotentialEnergy()
print(f"  After 100 steps MD: {pe_final}")
print("DONE!")
