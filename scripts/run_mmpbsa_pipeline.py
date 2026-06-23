"""
End-to-end MMPBSA pipeline for gyrB + moxifloxacin.

1. Build protein system (Modeller → amber14)
2. Build MFX ligand topology + manual forces
3. Combine, minimize, extract PDBs for MMPBSA.py
"""
import openmm as mm
import openmm.app as app
import openmm.unit as u
from openmm.app import Modeller
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import math, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.extract_mmpbsa_inputs import write_mmpbsa_pdbs, write_mmpbsa_input

# ======= CONFIG =======
LIGAND_RSMILES = 'CC1CN(CCN1C2=C(C=C3C(=C2OC)N(C(=O)C(=C3F)C(=O)O)CC4CC4)F)C'
PROT_PDB = 'data/pdb/alphafold/gyrB_P9WG45_alphafold.pdb'
LIGAND_DOCKED_PDB = 'data/pdb/mfx_WT_pose.pdb'
OUT_DIR = 'analysis/results'
PLATFORM = 'CPU'
N_STEPS_MIN = 5000
# ======================

# ---- 1. Protein system ----
print('=== 1. Building protein system ===')
pdb = app.PDBFile(PROT_PDB)
modeller = Modeller(pdb.topology, pdb.positions)
ff_prot = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
modeller.addHydrogens(ff_prot)
modeller.addSolvent(ff_prot, boxSize=u.Quantity(np.array([8,8,8]), u.nanometer))
n_prot_top = modeller.topology.getNumAtoms()

prot_system = ff_prot.createSystem(
    modeller.topology,
    nonbondedMethod=app.PME,
    nonbondedCutoff=1.0*u.nanometer,
    constraints=app.HBonds,
    rigidWater=True,
)
n_prot = prot_system.getNumParticles()
print(f'  Protein system: {n_prot} particles')

# ---- 2. Ligand preparation ----
print('\n=== 2. Preparing MFX ligand ===')
lig_smiles = LIGAND_RSMILES
mol = Chem.MolFromSmiles(lig_smiles)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, AllChem.ETKDG())
AllChem.MMFFOptimizeMolecule(mol)
conf = mol.GetConformer(0)

# Charges
props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94')
if props is None:
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')
charges = [props.GetMMFFPartialCharge(i) for i in range(mol.GetNumAtoms())]
cs = sum(charges); charges = [q - cs/len(charges) for q in charges]

# GAFF-like typing
def gaff_type(atom):
    s = atom.GetSymbol()
    if s == 'H': return 'hc'
    if s == 'F': return 'f'
    if s == 'O': return 'oh' if any(n.GetAtomicNum()==1 for n in atom.GetNeighbors()) else 'o'
    if s == 'N': return 'n3'
    if s == 'C': return 'ca' if atom.GetIsAromatic() else 'c3'
    return 'c3'

lj_sig = {'c3':0.33997,'ca':0.33997,'n3':0.32500,'o':0.29599,'oh':0.30664,'f':0.31181,'hc':0.25997}
lj_eps = {'c3':0.45773,'ca':0.45773,'n3':0.71128,'o':0.87864,'oh':0.880474,'f':0.278236,'hc':0.065688}
n_lig = mol.GetNumAtoms()

# Build ligand topology
lig_topology = app.Topology()
lig_chain = lig_topology.addChain('L')
lig_res = lig_topology.addResidue('MFX', lig_chain, 1)
for i, atom in enumerate(mol.GetAtoms()):
    elem_symbol = atom.GetSymbol()
    elem = app.Element.getBySymbol(elem_symbol)
    lig_topology.addAtom(f'{elem_symbol}{i}', elem, lig_res)
for bond in mol.GetBonds():
    atoms_list = list(lig_topology.atoms())
    lig_topology.addBond(atoms_list[bond.GetBeginAtomIdx()],
                         atoms_list[bond.GetEndAtomIdx()])
print(f'  Ligand topology: {n_lig} atoms, {lig_topology.getNumBonds()} bonds')

# ---- 3. Add ligand to system ----
print('\n=== 3. Adding ligand forces ===')
for atom in mol.GetAtoms():
    mass = {6:12.011,7:14.007,8:15.999,9:18.998,1:1.008}[atom.GetAtomicNum()] * u.amu
    prot_system.addParticle(mass)

# Extend NonbondedForce
for i in range(prot_system.getNumForces()):
    f = prot_system.getForce(i)
    if type(f).__name__ == 'NonbondedForce':
        nbf = f; break
for i, atom in enumerate(mol.GetAtoms()):
    g = gaff_type(atom)
    nbf.addParticle(charges[i]*u.elementary_charge, lj_sig[g]*u.nanometer, lj_eps[g]*u.kilojoule_per_mole)

# Bonded forces
from collections import defaultdict
adj = defaultdict(list)
for bond in mol.GetBonds():
    b1,b2 = bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
    adj[b1].append(b2); adj[b2].append(b1)

bf = mm.HarmonicBondForce()
for bond in mol.GetBonds():
    b1,b2 = bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
    sp = props.GetMMFFBondStretchParams(mol, b1, b2)
    k_mdyn, r0_a = (sp[1], sp[2]) if sp else (300.0/143.88, 1.5)
    k_kcal = k_mdyn * 143.88
    bf.addBond(n_prot+b1, n_prot+b2, r0_a/10.0, k_kcal*4.184*100.0)

af = mm.HarmonicAngleForce()
for center, nbrs in adj.items():
    for i in range(len(nbrs)):
        for j in range(i+1, len(nbrs)):
            af.addAngle(n_prot+nbrs[i], n_prot+center, n_prot+nbrs[j],
                        109.5*math.pi/180, 60.0*4.184)

tf = mm.PeriodicTorsionForce()
for a1 in adj:
    for a2 in adj[a1]:
        for a3 in adj[a2]:
            if a3 == a1: continue
            for a4 in adj[a3]:
                if a4 == a2: continue
                t2,t3 = gaff_type(mol.GetAtomWithIdx(a2)), gaff_type(mol.GetAtomWithIdx(a3))
                if t2 == 'ca' and t3 == 'ca': per,k = 2,3.5
                elif t2 == 'n3' or t3 == 'n3': per,k = 1,0.5
                else: per,k = 3,0.15
                tf.addTorsion(n_prot+a1, n_prot+a2, n_prot+a3, n_prot+a4, per, 0.0, k*4.184)

prot_system.addForce(bf); prot_system.addForce(af); prot_system.addForce(tf)
print(f'  Bonds: {bf.getNumBonds()}, Angles: {af.getNumAngles()}, Torsions: {tf.getNumTorsions()}')

# ---- 4. Ligand coordinates (align to docked pose) ----
print('\n=== 4. Aligning ligand coordinates ===')
mfx_pose = app.PDBFile(LIGAND_DOCKED_PDB)
dock_coords = np.array([[p[0]._value,p[1]._value,p[2]._value] for p in list(mfx_pose.positions)])
dock_center = np.mean(dock_coords, axis=0)
rdkit_coords = conf.GetPositions()
heavy_idx = [i for i,a in enumerate(mol.GetAtoms()) if a.GetAtomicNum()>1]
heavy_rdkit = rdkit_coords[heavy_idx]
heavy_center = np.mean(heavy_rdkit, axis=0)
hc = heavy_rdkit - heavy_center
dc = dock_coords - dock_center
Hmat = hc.T @ dc
U, S, Vt = np.linalg.svd(Hmat)
R = Vt.T @ U.T
if np.linalg.det(R) < 0:
    Vt[-1,:] *= -1
    R = Vt.T @ U.T
transformed = (rdkit_coords - heavy_center) @ R.T + dock_center
lig_positions = [u.Quantity(np.array(transformed[i]), u.nanometer)
                 for i in range(n_lig)]

# ---- 5. Minimization ----
print('\n=== 5. Energy minimization ===')
all_positions = list(modeller.positions) + lig_positions
integrator = mm.LangevinIntegrator(300*u.kelvin, 1.0/u.picosecond, 0.002*u.picosecond)
platform = mm.Platform.getPlatformByName(PLATFORM)
context = mm.Context(prot_system, integrator, platform)
context.setPositions(all_positions)

e0 = context.getState(getEnergy=True).getPotentialEnergy()
print(f'  Initial PE: {e0}')

mm.LocalEnergyMinimizer.minimize(context, maxIterations=N_STEPS_MIN)
state = context.getState(getPositions=True, enforcePeriodicBox=True)
e1 = context.getState(getEnergy=True).getPotentialEnergy()
print(f'  After {N_STEPS_MIN} steps: {e1}')

# ---- 6. Extract MMPBSA inputs ----
print('\n=== 6. Writing MMPBSA inputs ===')
min_pos = state.getPositions()

write_mmpbsa_input(ligand_mask=':676',
                   out_dir=OUT_DIR)

write_mmpbsa_pdbs(
    positions=min_pos,
    prot_topology=modeller.topology,
    ligand_topology=lig_topology,
    ligand_positions=lig_positions,
    out_dir=OUT_DIR,
)

print('\n=== Pipeline complete ===')
print('Next steps:')
print(f'  1. Run production MD to produce trajectory.dcd in {OUT_DIR}/')
print(f'  2. pip install gmx_MMPBSA')
print(f'  3. MMPBSA.py -O -i {OUT_DIR}/mmpbsa.in \\')
print(f'            -p {OUT_DIR}/complex.pdb \\')
print(f'            -y {OUT_DIR}/trajectory.dcd')
print('  4. Repeat for Q538L mutant to get DeltaDeltaG')
