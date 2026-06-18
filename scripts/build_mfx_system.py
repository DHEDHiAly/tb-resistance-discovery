"""
Build OpenMM system for gyrB + MFX manually.
Step 1: protein system (amber14)
Step 2: add MFX atoms with bonded and nonbonded forces
"""
import openmm as mm
import openmm.app as app
import openmm.unit as u
from openmm.app import Modeller
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

# ========== 1. Build protein system ==========
pdb = app.PDBFile('data/pdb/alphafold/gyrB_P9WG45_alphafold.pdb')
modeller = Modeller(pdb.topology, pdb.positions)
ff_prot = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
modeller.addHydrogens(ff_prot)
print(f"Protein after H: {modeller.topology.getNumAtoms()} atoms")

modeller.addSolvent(ff_prot, boxSize=u.Quantity(np.array([8,8,8]), u.nanometer))
print(f"Solvated: {modeller.topology.getNumAtoms()} atoms")

# Create protein system
prot_system = ff_prot.createSystem(
    modeller.topology,
    nonbondedMethod=app.PME,
    nonbondedCutoff=1.0*u.nanometer,
    constraints=app.HBonds,
    rigidWater=True,
)

# Save the protein-only system and topology for later
prot_positions = modeller.positions
prot_topology = modeller.topology
n_prot_particles = prot_system.getNumParticles()

# ========== 2. Prepare MFX molecule ==========
mfx_smiles = 'CC1CN(CCN1C2=C(C=C3C(=C2OC)N(C(=O)C(=C3F)C(=O)O)CC4CC4)F)C'
mol = Chem.MolFromSmiles(mfx_smiles)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, AllChem.ETKDG())
AllChem.MMFFOptimizeMolecule(mol)
conf = mol.GetConformer(0)

# Get MMFF94 charges
props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94')
if props is None:
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')
charges = [props.GetMMFFPartialCharge(i) for i in range(mol.GetNumAtoms())]
charge_sum = sum(charges)
charges = [q - charge_sum/len(charges) for q in charges]

atomic_num = [a.GetAtomicNum() for a in mol.GetAtoms()]
n_mfx_atoms = mol.GetNumAtoms()

# MFX positions (not yet aligned to docked pose)
mfx_positions_rdkit = [u.Quantity(np.array(conf.GetPositions()[i]), u.nanometer) for i in range(n_mfx_atoms)]

# ========== 3. Build MFX forces ==========
# FFXML building approach - use a minimal custom XML
# We'll construct a ForceField object with just the MFX parameters
# and use it to create forces for the MFX molecule alone

mfx_ffxml = '''<?xml version="1.0"?>
<ForceField>
  <AtomTypes>
'''
types_seen = set()
for i, atom in enumerate(mol.GetAtoms()):
    elem = atom.GetSymbol()
    if elem == 'C': mass = 12.011
    elif elem == 'N': mass = 14.007
    elif elem == 'O': mass = 15.999
    elif elem == 'F': mass = 18.998
    elif elem == 'H': mass = 1.008
    else: mass = 12.0
    
    # GAFF-like type
    if elem == 'H': atype = 'hc'
    elif elem == 'F': atype = 'f'
    elif elem == 'O':
        neighbors = [n.GetAtomicNum() for n in atom.GetNeighbors()]
        atype = 'oh' if 1 in neighbors else 'o'
    elif elem == 'N':
        bo = sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())
        atype = 'n3' if bo <= 3 else 'n2'
    elif elem == 'C':
        if atom.GetIsAromatic(): atype = 'ca'
        else: atype = 'c3'
    else: atype = 'c3'
    
    tname = f'MFX-{atype}'
    if tname not in types_seen:
        types_seen.add(tname)
        mfx_ffxml += f'    <Type name="{tname}" class="{atype}" element="{elem}" mass="{mass}"/>\n'

mfx_ffxml += '''  </AtomTypes>
  <Residues>
    <Residue name="MFX">
'''
# Add atoms
for i, atom in enumerate(mol.GetAtoms()):
    elem = atom.GetSymbol()
    if elem == 'H': atype = 'hc'
    elif elem == 'F': atype = 'f'
    elif elem == 'O':
        neighbors = [n.GetAtomicNum() for n in atom.GetNeighbors()]
        atype = 'oh' if 1 in neighbors else 'o'
    elif elem == 'N':
        bo = sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())
        atype = 'n3' if bo <= 3 else 'n2'
    elif elem == 'C':
        atype = 'ca' if atom.GetIsAromatic() else 'c3'
    else: atype = 'c3'
    mfx_ffxml += f'      <Atom name="{elem}{i}" type="MFX-{atype}" charge="{charges[i]:.6f}"/>\n'

# Add bonds in residue template  
for bond in mol.GetBonds():
    b1, b2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    mfx_ffxml += f'      <Bond from="{b1}" to="{b2}"/>\n'

mfx_ffxml += '''    </Residue>
  </Residues>
'''

# Estimate bond parameters
gaff_lj = {
    'c3': (0.33997, 0.457730), 'ca': (0.33997, 0.457730),
    'n3': (0.32500, 0.711280), 'n2': (0.32500, 0.711280),
    'o': (0.29599, 0.878640), 'oh': (0.30664, 0.880474),
    'f': (0.31181, 0.278236), 'hc': (0.25997, 0.065688),
}

mfx_ffxml += '  <Bonds>\n'
bond_types_done = set()
for bond in mol.GetBonds():
    b1, b2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    # Get atom types
    def get_atype(atom):
        if atom.GetSymbol() == 'H': return 'hc'
        elif atom.GetSymbol() == 'F': return 'f'
        elif atom.GetSymbol() == 'O':
            return 'oh' if 1 in [n.GetAtomicNum() for n in atom.GetNeighbors()] else 'o'
        elif atom.GetSymbol() == 'N':
            return 'n3' if sum(b.GetBondTypeAsDouble() for b in atom.GetBonds()) <= 3 else 'n2'
        elif atom.GetSymbol() == 'C':
            return 'ca' if atom.GetIsAromatic() else 'c3'
        return 'c3'
    t1 = get_atype(mol.GetAtomWithIdx(b1))
    t2 = get_atype(mol.GetAtomWithIdx(b2))
    key = tuple(sorted([t1, t2]))
    if key in bond_types_done: continue
    bond_types_done.add(key)
    
    # Get MMFF bond params
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
    k_kj = k_kcal * 4.184 * 100  # 4.184 kcal->kJ, *100 for A^2 -> nm^2
    mfx_ffxml += f'      <Bond type1="MFX-{t1}" type2="MFX-{t2}" length="{r0_nm:.4f}" k="{k_kj:.1f}"/>\n'

mfx_ffxml += '  </Bonds>\n  <Angles>\n'
# Generic angle params (convert: 109.5 deg -> rad, 60 kcal->kJ)
angle_rad = 109.5 * 3.14159 / 180
k_angle = 60.0 * 4.184
angle_types = [
    ('c3','c3','c3'),('c3','c3','hc'),('hc','c3','hc'),
    ('c3','c3','o'),('c3','c3','n3'),('c3','c3','ca'),
    ('c3','ca','ca'),('ca','ca','ca'),('ca','ca','hc'),
    ('ca','ca','n3'),('c3','n3','c3'),('c3','n3','hc'),
    ('c3','o','hc'),('c3','c3','f'),('c3','c3','n2'),
    ('ca','ca','n2'),('ca','ca','o'),('c3','n3','n3'),
    ('c3','c2','o'),('c2','c2','o'),('c2','c3','n3'),
]
for t1, t2, t3 in angle_types:
    mfx_ffxml += f'      <Angle type1="MFX-{t1}" type2="MFX-{t2}" type3="MFX-{t3}" angle="{angle_rad:.4f}" k="{k_angle:.1f}"/>\n'

mfx_ffxml += '  </Angles>\n  <Torsions>\n'
# Generic torsion params (convert: kcal->kJ)
torsion_params = [
    ('c3','c3','c3','c3', 3, 0.15), ('c3','c3','c3','hc', 3, 0.15),
    ('hc','c3','c3','hc', 3, 0.15), ('c3','c3','c3','o', 3, 0.15),
    ('c3','c3','c3','n3', 3, 0.15), ('c3','c3','c3','ca', 3, 0.15),
    ('c3','ca','ca','ca', 2, 3.5), ('c3','ca','ca','hc', 2, 3.5),
    ('hc','ca','ca','hc', 2, 3.5), ('ca','ca','ca','ca', 2, 3.5),
    ('ca','ca','ca','hc', 2, 3.5), ('ca','ca','n3','c3', 2, 3.5),
    ('c3','n3','c3','c3', 1, 0.5), ('c3','n3','c3','hc', 1, 0.5),
    ('c3','c3','n3','c3', 1, 0.5), ('c3','c3','ca','ca', 2, 1.0),
    ('c3','c3','ca','o', 2, 0.5), ('ca','ca','c3','o', 2, 0.5),
    ('ca','ca','c3','hc', 2, 0.5),
]
for t1, t2, t3, t4, per, k_kcal in torsion_params:
    k_kj = k_kcal * 4.184
    mfx_ffxml += f'      <Torsion type1="MFX-{t1}" type2="MFX-{t2}" type3="MFX-{t3}" type4="MFX-{t4}" periodicity="{per}" phase="0.0" k="{k_kj:.3f}"/>\n'

mfx_ffxml += '''  </Torsions>
  <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
    <UseAttributeFromResidue name="charge"/>
'''
for atype_name in types_seen:
    suffix = atype_name.replace('MFX-', '')
    sigma, eps = gaff_lj.get(suffix, (0.34, 0.45773))
    mfx_ffxml += f'      <Atom type="{atype_name}" sigma="{sigma:.4f}" epsilon="{eps:.6f}"/>\n'

mfx_ffxml += '''  </NonbondedForce>
</ForceField>
'''

print(f"Generated FFXML ({len(mfx_ffxml)} chars)")

# ========== 4. Create MFX-only topology to get forces ==========
# We need a topology with only MFX to generate the force field terms
# Then we'll copy these terms into the protein system

mfx_top = app.Topology()
mfx_chain = mfx_top.addChain('X')
mfx_res = mfx_top.addResidue('MFX', mfx_chain)
for i, atom in enumerate(mol.GetAtoms()):
    elem = atom.GetSymbol()
    aname = f'{elem}{i}'
    at = app.Element.getBySymbol(elem)
    mfx_top.addAtom(aname, at, mfx_res)

# Add bonds to MFX topology
for bond in mol.GetBonds():
    b1, b2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    atoms = list(mfx_res.atoms())
    mfx_top.addBond(atoms[b1], atoms[b2])

# Save and load the MFX force field
with open('data/pdb/gaff_MFX_v2.xml', 'w') as f:
    f.write(mfx_ffxml)
mfx_ff = app.ForceField('data/pdb/gaff_MFX_v2.xml')
mfx_system = mfx_ff.createSystem(mfx_top, nonbondedMethod=app.NoCutoff, constraints=None)

print(f"MFX system: {mfx_system.getNumParticles()} particles, {mfx_system.getNumForces()} forces")
for i in range(mfx_system.getNumForces()):
    f = mfx_system.getForce(i)
    nb = f.getNumBonds() if hasattr(f, 'getNumBonds') else 'N/A'
    na = f.getNumAngles() if hasattr(f, 'getNumAngles') else 'N/A'
    nt = f.getNumTorsions() if hasattr(f, 'getNumTorsions') else 'N/A'
    print(f"  Force {i}: {type(f).__name__} bonds={nb} angles={na} torsions={nt}")

# Save the MFX ffxml for debugging
with open('data/pdb/gaff_MFX_v2.xml', 'w') as f:
    f.write(mfx_ffxml)
print("Saved gaff_MFX_v2.xml")
