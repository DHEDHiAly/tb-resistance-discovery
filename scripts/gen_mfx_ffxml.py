from rdkit import Chem
from rdkit.Chem import AllChem
import xml.dom.minidom as minidom

mfx_smiles = 'CC1CN(CCN1C2=C(C=C3C(=C2OC)N(C(=O)C(=C3F)C(=O)O)CC4CC4)F)C'
mol = Chem.MolFromSmiles(mfx_smiles)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, AllChem.ETKDG())
AllChem.MMFFOptimizeMolecule(mol)
props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94')
if props is None:
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')

conf = mol.GetConformer(0)
charges = [props.GetMMFFPartialCharge(i) for i in range(mol.GetNumAtoms())]

# Adjust charges to sum to 0
charge_sum = sum(charges)
if abs(charge_sum) > 0.01:
    charges = [q - charge_sum/len(charges) for q in charges]

def get_atom_name(atom, idx):
    ai = atom.GetMonomerInfo()
    if ai and ai.GetName().strip():
        return ai.GetName().strip()
    return f'{atom.GetSymbol()}{idx}'

# GAFF-like atom classification
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

gaff_lj = {
    'c3': (3.3997, 0.1094), 'c2': (3.3997, 0.1094), 'ca': (3.3997, 0.1094),
    'n3': (3.2500, 0.1700), 'na': (3.2500, 0.1700), 'n2': (3.2500, 0.1700),
    'o': (2.9599, 0.2100), 'oh': (3.0664, 0.2104),
    'f': (3.1181, 0.0665), 'hc': (2.5997, 0.0157),
}

elem_mass = {'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'H': 1.008}

atom_info = []
for i, atom in enumerate(mol.GetAtoms()):
    atype = get_gaff_type(atom)
    name = get_atom_name(atom, i)
    symb = atom.GetSymbol()
    sigma, eps = gaff_lj.get(atype, (3.40, 0.109))
    atom_info.append({
        'idx': i, 'name': name, 'type': atype, 'element': symb,
        'charge': charges[i], 'sigma': sigma, 'epsilon': eps,
        'mass': elem_mass.get(symb, 12.0)
    })

unique_types = []
seen = set()
for a in atom_info:
    if a['type'] not in seen:
        seen.add(a['type'])
        unique_types.append(a)

print(f'Atoms: {len(atom_info)}, unique types: {len(unique_types)}')
print(f'Types: {[t["type"] for t in unique_types]}')

# Build XML
doc = minidom.Document()
ff = doc.createElement('ForceField')
doc.appendChild(ff)

info_el = doc.createElement('Info')
info_el.appendChild(doc.createElement('DateGenerated'))
info_el.firstChild.appendChild(doc.createTextNode('2025'))
ff.appendChild(info_el)

atypes_el = doc.createElement('AtomTypes')
for t in unique_types:
    te = doc.createElement('Type')
    te.setAttribute('name', f'MFX-{t["type"]}')
    te.setAttribute('class', t['type'])
    te.setAttribute('element', t['element'])
    te.setAttribute('mass', f'{t["mass"]:.3f}')
    atypes_el.appendChild(te)
ff.appendChild(atypes_el)

res_el = doc.createElement('Residues')
re = doc.createElement('Residue')
re.setAttribute('name', 'MFX')
for a in atom_info:
    ae = doc.createElement('Atom')
    ae.setAttribute('name', a['name'])
    ae.setAttribute('type', f'MFX-{a["type"]}')
    ae.setAttribute('charge', f'{a["charge"]:.6f}')
    re.appendChild(ae)
for bond in mol.GetBonds():
    b1, b2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    be = doc.createElement('Bond')
    be.setAttribute('from', str(b1))
    be.setAttribute('to', str(b2))
    re.appendChild(be)
res_el.appendChild(re)
ff.appendChild(res_el)

# Estimate bond parameters from MMFF or use generic values
bonds_el = doc.createElement('HarmonicBondForce')
bond_types_done = set()
for bond in mol.GetBonds():
    b1, b2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    t1, t2 = atom_info[b1]['type'], atom_info[b2]['type']
    key = tuple(sorted([t1, t2]))
    if key in bond_types_done:
        continue
    bond_types_done.add(key)
    # Get MMFF parameters
    try:
        sp = props.GetMMFFBondStretchParams(mol, b1, b2)
        if sp:
            _, k_mdyn, r0_a = sp
            k_kcal = k_mdyn * 143.88  # mdyn/A to kcal/(mol*A^2)
        else:
            k_kcal, r0_a = 300.0, 1.5
    except:
        k_kcal, r0_a = 300.0, 1.5
    be = doc.createElement('Bond')
    be.setAttribute('type1', f'MFX-{t1}')
    be.setAttribute('type2', f'MFX-{t2}')
    be.setAttribute('length', f'{r0_a/10:.4f}')
    be.setAttribute('k', f'{k_kcal*4.184*100:.1f}')
    bonds_el.appendChild(be)
ff.appendChild(bonds_el)

# Generic angle parameters
angles_el = doc.createElement('HarmonicAngleForce')
angle_types_done = set()
angle_template = [
    ('c3', 'c3', 'c3'), ('c3', 'c3', 'hc'), ('hc', 'c3', 'hc'),
    ('c3', 'c3', 'o'), ('c3', 'c3', 'n3'), ('c3', 'c3', 'ca'),
    ('c3', 'ca', 'ca'), ('ca', 'ca', 'ca'), ('ca', 'ca', 'hc'),
    ('ca', 'ca', 'n3'), ('c3', 'n3', 'c3'), ('c3', 'n3', 'hc'),
    ('c3', 'o', 'hc'), ('c3', 'c3', 'f'), ('c3', 'c3', 'na'),
    ('ca', 'ca', 'na'), ('ca', 'ca', 'o'), ('c3', 'n3', 'n3'),
    ('c3', 'c2', 'o'), ('c2', 'c2', 'o'), ('c2', 'c3', 'n3'),
    ('c3', 'n3', 'c3'),
]
for t1, t2, t3 in angle_template:
    key = (t1, t2, t3)
    if key in angle_types_done: continue
    angle_types_done.add(key)
    ae = doc.createElement('Angle')
    ae.setAttribute('type1', f'MFX-{t1}')
    ae.setAttribute('type2', f'MFX-{t2}')
    ae.setAttribute('type3', f'MFX-{t3}')
    ae.setAttribute('angle', f'{109.5*3.14159/180:.4f}')
    ae.setAttribute('k', f'{60.0*4.184:.1f}')
    angles_el.appendChild(ae)
ff.appendChild(angles_el)

# Generic torsion parameters  
torsions_el = doc.createElement('PeriodicTorsionForce')
tor_types_done = set()
for t1, t2, t3, t4, period, phase_kcal in [
    ('c3', 'c3', 'c3', 'c3', 3, 0.15),
    ('c3', 'c3', 'c3', 'hc', 3, 0.15),
    ('hc', 'c3', 'c3', 'hc', 3, 0.15),
    ('c3', 'c3', 'c3', 'o', 3, 0.15),
    ('c3', 'c3', 'c3', 'n3', 3, 0.15),
    ('c3', 'c3', 'c3', 'ca', 3, 0.15),
    ('c3', 'ca', 'ca', 'ca', 2, 3.5),
    ('c3', 'ca', 'ca', 'hc', 2, 3.5),
    ('hc', 'ca', 'ca', 'hc', 2, 3.5),
    ('ca', 'ca', 'ca', 'ca', 2, 3.5),
    ('ca', 'ca', 'ca', 'hc', 2, 3.5),
    ('ca', 'ca', 'n3', 'c3', 2, 3.5),
    ('c3', 'n3', 'c3', 'c3', 1, 0.5),
    ('c3', 'n3', 'c3', 'hc', 1, 0.5),
    ('c3', 'c3', 'n3', 'c3', 1, 0.5),
    ('c3', 'c3', 'ca', 'ca', 2, 1.0),
    ('c3', 'c3', 'ca', 'o', 2, 0.5),
    ('ca', 'ca', 'c3', 'o', 2, 0.5),
    ('ca', 'ca', 'c3', 'hc', 2, 0.5),
]:
    key = (t1, t2, t3, t4, period)
    if key in tor_types_done: continue
    tor_types_done.add(key)
    te = doc.createElement('Torsion')
    te.setAttribute('type1', f'MFX-{t1}')
    te.setAttribute('type2', f'MFX-{t2}')
    te.setAttribute('type3', f'MFX-{t3}')
    te.setAttribute('type4', f'MFX-{t4}')
    te.setAttribute('periodicity', str(period))
    te.setAttribute('phase', '0.0')
    te.setAttribute('k', f'{phase_kcal*4.184}')
    torsions_el.appendChild(te)
ff.appendChild(torsions_el)

nb = doc.createElement('NonbondedForce')
nb.setAttribute('coulomb14scale', '0.833333')
nb.setAttribute('lj14scale', '0.5')
uar = doc.createElement('UseAttributeFromResidue')
uar.setAttribute('name', 'charge')
nb.appendChild(uar)
for t in unique_types:
    ne = doc.createElement('Atom')
    ne.setAttribute('type', f'MFX-{t["type"]}')
    ne.setAttribute('sigma', f'{t["sigma"]:.4f}')
    ne.setAttribute('epsilon', f'{t["epsilon"]:.6f}')
    nb.appendChild(ne)
ff.appendChild(nb)

with open('data/pdb/gaff_MFX.xml', 'w') as f:
    f.write(doc.toprettyxml(indent='  '))
print('Done! gaff_MFX.xml written')
