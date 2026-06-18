"""
Sidechain building for residue mutations using standard geometry.
Builds complete sidechain heavy atoms given backbone coordinates.

Standard bond lengths, angles, and dihedral preferences from:
- Engh & Huber (2001) Acta Cryst.
- Dunbrack Rotamer Library (penultimate)
- Default χ angles: extended (180°)
"""
import numpy as np
from math import radians, cos, sin, sqrt

# Standard bond lengths (Å)
BOND = {
    ('N','CA'): 1.458, ('CA','C'): 1.525, ('C','O'): 1.231, ('C','OXT'): 1.26,
    ('CA','CB'): 1.531, ('CB','CG'): 1.530, ('CG','CD'): 1.530, ('CD','CE'): 1.530,
    ('CE','NZ'): 1.490, ('CE','CZ'): 1.490, ('CZ','NH1'): 1.340, ('CZ','NH2'): 1.340,
    ('CG','CD1'): 1.530, ('CG','CD2'): 1.530, ('CD','OE1'): 1.240, ('CD','NE2'): 1.340,
    ('CG','OD1'): 1.240, ('CG','ND2'): 1.340, ('CD','O'): 1.240, ('CD','N'): 1.340,
    ('CD','NE'): 1.470, ('CG','CB'): 1.530, ('CB','OG'): 1.430, ('CB','SG'): 1.820,
    ('CG','SD'): 1.820, ('SD','CE'): 1.820,
    ('CB','CG1'): 1.530, ('CB','CG2'): 1.530, ('CG1','CD1'): 1.530,
    ('CB','CD'): 1.530,
}

# Standard bond angles (degrees)
ANGLE = {
    ('N','CA','C'): 111.0, ('N','CA','CB'): 110.4, ('C','CA','CB'): 110.7,
    ('CA','C','O'): 120.8, ('CA','C','OXT'): 116.0, ('O','C','OXT'): 122.5,
    ('CA','CB','CG'): 113.0, ('CA','CB','OG'): 110.0, ('CA','CB','SG'): 113.0,
    ('CB','CG','CD'): 113.0, ('CB','CG','SD'): 113.0,
    ('CB','CG','CD1'): 110.0, ('CB','CG','CD2'): 110.0, ('CD1','CG','CD2'): 110.0,
    ('CG','CD','NE'): 112.0, ('CG','CD','OE1'): 121.0, ('CG','CD','NE2'): 116.0,
    ('OE1','CD','NE2'): 123.0, ('CG','CD','N'): 116.0, ('CG','CD','O'): 121.0,
    ('O','CD','N'): 123.0, ('CG','CD','CE'): 113.0,
    ('CD','CE','NZ'): 112.0, ('CD','CE','CZ'): 112.0,
    ('CE','CZ','NH1'): 120.0, ('CE','CZ','NH2'): 120.0, ('NH1','CZ','NH2'): 120.0,
    ('CG','CD1','CE1'): 120.0, ('CG','CD2','CE2'): 120.0,
    ('CA','CB','CG1'): 110.0, ('CA','CB','CG2'): 110.0, ('CG1','CB','CG2'): 110.0,
    ('CB','CG1','CD1'): 110.0, ('CG','CE1','ND1'): 106.0, ('CG','ND1','CE1'): 109.0,
    ('CB','CG','ND1'): 125.0, ('CB','CG','CD2'): 127.0, ('ND1','CG','CD2'): 108.0,
}

def norm(v):
    return sqrt(v[0]**2 + v[1]**2 + v[2]**2)

def unit(v):
    n = norm(v)
    return v/n if n > 1e-10 else np.array([0.0, 0.0, 0.0])

def build_atom(prev, center, next_p, bond_len, bond_angle, dihedral):
    """
    Build an atom position given:
    - prev: previous atom position
    - center: central atom position (where new atom connects)
    - next_p: the atom before center (for dihedral reference)
    - bond_len: distance center->new atom
    - bond_angle: angle next_p-center-new (degrees)
    - dihedral: dihedral angle next_p-center-prev-new (degrees)
    """
    r1 = center - next_p
    r2 = center - prev
    # Bond angle
    angle_rad = radians(180.0 - bond_angle)
    # Vector perpendicular to the plane
    n_vec = np.cross(r1, r2)
    n_unit = unit(n_vec)
    # Rotation axis
    axis = unit(r1)
    # Rotate around axis by dihedral
    dih_rad = radians(dihedral)
    cos_d = cos(dih_rad)
    sin_d = sin(dih_rad)
    # Start from the bond angle direction
    # Use Rodrigues' rotation formula
    v = -unit(r1)  # direction from center to prev
    # Rotate v by bond_angle towards n_vec direction
    v_rot = v * cos(angle_rad) + n_unit * sin(angle_rad)
    # Now rotate around r1 by dihedral
    v_final = v_rot * cos_d + np.cross(axis, v_rot) * sin_d + axis * np.dot(axis, v_rot) * (1 - cos_d)
    return center + v_final * bond_len

def get_bond_len(atoms, i, j):
    key = (i,j) if (i,j) in BOND else (j,i)
    return BOND.get(key, 1.53)

def get_angle(atoms, i, j, k):
    key = (i,j,k)
    if key in ANGLE: return ANGLE[key]
    key = (k,j,i)
    if key in ANGLE: return ANGLE[key]
    return 110.0

# ============================================================
# Residue topology definitions
# ============================================================
# Format for each AA:
# {
#   'atoms': [atom names in order],  # backbone atoms first
#   'parents': {atom: parent_idx},  # atom index it bonds to
#   'bonds': [(from_idx, to_idx, ), ...],
#   'build_order': [(new_atom, parent, prev, angle_key, dihedral_angle), ...]
# }

RESIDUE_BUILD = {
    'ALA': {
        'sc_atoms': ['CB'],
        'bonds': [('CA', 'CB')],
        'build': [
            ('CB', 'CA', 'N', 'C', ('N','CA','CB'), 180.0),
        ]
    },
    'ARG': {
        'sc_atoms': ['CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
        'bonds': [('CA','CB'), ('CB','CG'), ('CG','CD'), ('CD','NE'), ('NE','CZ'), ('CZ','NH1'), ('CZ','NH2')],
        'build': [
            ('CB', 'CA', 'N', 'C', ('N','CA','CB'), 180.0),
            ('CG', 'CB', 'CA', 'N', ('CA','CB','CG'), 180.0),
            ('CD', 'CG', 'CB', 'CA', ('CB','CG','CD'), 180.0),
            ('NE', 'CD', 'CG', 'CB', ('CG','CD','NE'), 180.0),
            ('CZ', 'NE', 'CD', 'CG', ('CD','NE','CZ'), 180.0),
            ('NH1', 'CZ', 'NE', 'CD', ('NE','CZ','NH1'), 180.0),
            ('NH2', 'CZ', 'NE', 'CD', ('NE','CZ','NH2'), 0.0),
        ]
    },
    'ASN': {
        'sc_atoms': ['CB', 'CG', 'OD1', 'ND2'],
        'bonds': [('CA','CB'), ('CB','CG'), ('CG','OD1'), ('CG','ND2')],
        'build': [
            ('CB', 'CA', 'N', 'C', ('N','CA','CB'), 180.0),
            ('CG', 'CB', 'CA', 'N', ('CA','CB','CG'), 180.0),
            ('OD1', 'CG', 'CB', 'CA', ('CB','CG','OD1'), 0.0),
            ('ND2', 'CG', 'CB', 'CA', ('CB','CG','ND2'), 180.0),
        ]
    },
    'ASP': {
        'sc_atoms': ['CB', 'CG', 'OD1', 'OD2'],
        'bonds': [('CA','CB'), ('CB','CG'), ('CG','OD1'), ('CG','OD2')],
        'build': [
            ('CB', 'CA', 'N', 'C', ('N','CA','CB'), 180.0),
            ('CG', 'CB', 'CA', 'N', ('CA','CB','CG'), 180.0),
            ('OD1', 'CG', 'CB', 'CA', ('CB','CG','OD1'), 0.0),
            ('OD2', 'CG', 'CB', 'CA', ('CB','CG','OD2'), 180.0),
        ]
    },
    'CYS': {
        'sc_atoms': ['CB', 'SG'],
        'bonds': [('CA','CB'), ('CB','SG')],
        'build': [
            ('CB', 'CA', 'N', 'C', ('N','CA','CB'), 180.0),
            ('SG', 'CB', 'CA', 'N', ('CA','CB','SG'), 180.0),
        ]
    },
    'GLN': {
        'sc_atoms': ['CB', 'CG', 'CD', 'OE1', 'NE2'],
        'bonds': [('CA','CB'), ('CB','CG'), ('CG','CD'), ('CD','OE1'), ('CD','NE2')],
        'build': [
            ('CB', 'CA', 'N', 'C', ('N','CA','CB'), 180.0),
            ('CG', 'CB', 'CA', 'N', ('CA','CB','CG'), 180.0),
            ('CD', 'CG', 'CB', 'CA', ('CB','CG','CD'), 180.0),
            ('OE1', 'CD', 'CG', 'CB', ('CG','CD','OE1'), 0.0),
            ('NE2', 'CD', 'CG', 'CB', ('CG','CD','NE2'), 180.0),
        ]
    },
    'GLU': {
        'sc_atoms': ['CB', 'CG', 'CD', 'OE1', 'OE2'],
        'bonds': [('CA','CB'), ('CB','CG'), ('CG','CD'), ('CD','OE1'), ('CD','OE2')],
        'build': [
            ('CB', 'CA', 'N', 'C', ('N','CA','CB'), 180.0),
            ('CG', 'CB', 'CA', 'N', ('CA','CB','CG'), 180.0),
            ('CD', 'CG', 'CB', 'CA', ('CB','CG','CD'), 180.0),
            ('OE1', 'CD', 'CG', 'CB', ('CG','CD','OE1'), 0.0),
            ('OE2', 'CD', 'CG', 'CB', ('CG','CD','OE2'), 180.0),
        ]
    },
    'GLY': {
        'sc_atoms': [],
        'bonds': [],
        'build': []
    },
    'HIS': {
        'sc_atoms': ['CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],
        'bonds': [('CA','CB'), ('CB','CG'), ('CG','ND1'), ('CG','CD2'), ('ND1','CE1'), ('CD2','NE2'), ('CE1','NE2')],
        'build': [
            ('CB', 'CA', 'N', 'C', ('N','CA','CB'), 180.0),
            ('CG', 'CB', 'CA', 'N', ('CA','CB','CG'), 180.0),
            ('ND1', 'CG', 'CB', 'CA', ('CB','CG','ND1'), 0.0),
            ('CD2', 'CG', 'CB', 'CA', ('CB','CG','CD2'), 180.0),
            ('CE1', 'ND1', 'CG', 'CB', ('CG','ND1','CE1'), 0.0),
            ('NE2', 'CD2', 'CG', 'CB', ('CG','CD2','NE2'), 0.0),
        ]
    },
    'ILE': {
        'sc_atoms': ['CB', 'CG1', 'CG2', 'CD1'],
        'bonds': [('CA','CB'), ('CB','CG1'), ('CB','CG2'), ('CG1','CD1')],
        'build': [
            ('CB', 'CA', 'N', 'C', ('N','CA','CB'), 180.0),
            ('CG1', 'CB', 'CA', 'N', ('CA','CB','CG1'), -60.0),
            ('CG2', 'CB', 'CA', 'N', ('CA','CB','CG2'), 60.0),
            ('CD1', 'CG1', 'CB', 'CA', ('CB','CG1','CD1'), 180.0),
        ]
    },
    'LEU': {
        'sc_atoms': ['CB', 'CG', 'CD1', 'CD2'],
        'bonds': [('CA','CB'), ('CB','CG'), ('CG','CD1'), ('CG','CD2')],
        'build': [
            ('CB', 'CA', 'N', 'C', ('N','CA','CB'), 180.0),
            ('CG', 'CB', 'CA', 'N', ('CA','CB','CG'), 180.0),
            ('CD1', 'CG', 'CB', 'CA', ('CB','CG','CD1'), 60.0),
            ('CD2', 'CG', 'CB', 'CA', ('CB','CG','CD2'), -60.0),
        ]
    },
    'LYS': {
        'sc_atoms': ['CB', 'CG', 'CD', 'CE', 'NZ'],
        'bonds': [('CA','CB'), ('CB','CG'), ('CG','CD'), ('CD','CE'), ('CE','NZ')],
        'build': [
            ('CB', 'CA', 'N', 'C', ('N','CA','CB'), 180.0),
            ('CG', 'CB', 'CA', 'N', ('CA','CB','CG'), 180.0),
            ('CD', 'CG', 'CB', 'CA', ('CB','CG','CD'), 180.0),
            ('CE', 'CD', 'CG', 'CB', ('CG','CD','CE'), 180.0),
            ('NZ', 'CE', 'CD', 'CG', ('CD','CE','NZ'), 180.0),
        ]
    },
    'MET': {
        'sc_atoms': ['CB', 'CG', 'SD', 'CE'],
        'bonds': [('CA','CB'), ('CB','CG'), ('CG','SD'), ('SD','CE')],
        'build': [
            ('CB', 'CA', 'N', 'C', ('N','CA','CB'), 180.0),
            ('CG', 'CB', 'CA', 'N', ('CA','CB','CG'), 180.0),
            ('SD', 'CG', 'CB', 'CA', ('CB','CG','SD'), 180.0),
            ('CE', 'SD', 'CG', 'CB', ('CG','SD','CE'), 180.0),
        ]
    },
    'PHE': {
        'sc_atoms': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
        'bonds': [('CA','CB'), ('CB','CG'), ('CG','CD1'), ('CG','CD2'), ('CD1','CE1'), ('CD2','CE2'), ('CE1','CZ'), ('CE2','CZ')],
        'build': [
            ('CB', 'CA', 'N', 'C', ('N','CA','CB'), 180.0),
            ('CG', 'CB', 'CA', 'N', ('CA','CB','CG'), 90.0),
            ('CD1', 'CG', 'CB', 'CA', ('CB','CG','CD1'), 0.0),
            ('CD2', 'CG', 'CB', 'CA', ('CB','CG','CD2'), 180.0),
            ('CE1', 'CD1', 'CG', 'CB', ('CG','CD1','CE1'), 0.0),
            ('CE2', 'CD2', 'CG', 'CB', ('CG','CD2','CE2'), 0.0),
            ('CZ', 'CE1', 'CD1', 'CG', ('CD1','CE1','CZ'), 0.0),
        ]
    },
    'PRO': {
        'sc_atoms': ['CB', 'CG', 'CD'],
        'bonds': [('CA','CB'), ('CB','CG'), ('CG','CD'), ('CD','N')],
        'build': [
            ('CB', 'CA', 'N', 'C', ('N','CA','CB'), 180.0),
            ('CG', 'CB', 'CA', 'N', ('CA','CB','CG'), 180.0),
            ('CD', 'CG', 'CB', 'CA', ('CB','CG','CD'), 180.0),
        ]
    },
    'SER': {
        'sc_atoms': ['CB', 'OG'],
        'bonds': [('CA','CB'), ('CB','OG')],
        'build': [
            ('CB', 'CA', 'N', 'C', ('N','CA','CB'), 180.0),
            ('OG', 'CB', 'CA', 'N', ('CA','CB','OG'), 180.0),
        ]
    },
    'THR': {
        'sc_atoms': ['CB', 'OG1', 'CG2'],
        'bonds': [('CA','CB'), ('CB','OG1'), ('CB','CG2')],
        'build': [
            ('CB', 'CA', 'N', 'C', ('N','CA','CB'), 180.0),
            ('OG1', 'CB', 'CA', 'N', ('CA','CB','OG1'), -60.0),
            ('CG2', 'CB', 'CA', 'N', ('CA','CB','CG2'), 60.0),
        ]
    },
    'TRP': {
        'sc_atoms': ['CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
        'bonds': [('CA','CB'), ('CB','CG'), ('CG','CD1'), ('CG','CD2'), ('CD2','CE2'), ('CD2','CE3'), ('NE1','CE2'), ('CE2','CZ2'), ('CE3','CZ3'), ('CZ2','CH2'), ('CZ3','CH2')],
        'build': [
            ('CB', 'CA', 'N', 'C', ('N','CA','CB'), 180.0),
            ('CG', 'CB', 'CA', 'N', ('CA','CB','CG'), 90.0),
            ('CD1', 'CG', 'CB', 'CA', ('CB','CG','CD1'), 0.0),
            ('CD2', 'CG', 'CB', 'CA', ('CB','CG','CD2'), 180.0),
            ('NE1', 'CD1', 'CG', 'CB', ('CG','CD1','NE1'), 0.0),
            ('CE2', 'CD2', 'CG', 'CB', ('CG','CD2','CE2'), 0.0),
            ('CE3', 'CD2', 'CG', 'CB', ('CG','CD2','CE3'), 180.0),
            ('CZ2', 'CE2', 'CD2', 'CG', ('CD2','CE2','CZ2'), 0.0),
            ('CZ3', 'CE3', 'CD2', 'CG', ('CD2','CE3','CZ3'), 0.0),
            ('CH2', 'CZ2', 'CE2', 'CD2', ('CE2','CZ2','CH2'), 0.0),
        ]
    },
    'TYR': {
        'sc_atoms': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],
        'bonds': [('CA','CB'), ('CB','CG'), ('CG','CD1'), ('CG','CD2'), ('CD1','CE1'), ('CD2','CE2'), ('CE1','CZ'), ('CE2','CZ'), ('CZ','OH')],
        'build': [
            ('CB', 'CA', 'N', 'C', ('N','CA','CB'), 180.0),
            ('CG', 'CB', 'CA', 'N', ('CA','CB','CG'), 90.0),
            ('CD1', 'CG', 'CB', 'CA', ('CB','CG','CD1'), 0.0),
            ('CD2', 'CG', 'CB', 'CA', ('CB','CG','CD2'), 180.0),
            ('CE1', 'CD1', 'CG', 'CB', ('CG','CD1','CE1'), 0.0),
            ('CE2', 'CD2', 'CG', 'CB', ('CG','CD2','CE2'), 0.0),
            ('CZ', 'CE1', 'CD1', 'CG', ('CD1','CE1','CZ'), 0.0),
            ('OH', 'CZ', 'CE1', 'CD1', ('CE1','CZ','OH'), 0.0),
        ]
    },
    'VAL': {
        'sc_atoms': ['CB', 'CG1', 'CG2'],
        'bonds': [('CA','CB'), ('CB','CG1'), ('CB','CG2')],
        'build': [
            ('CB', 'CA', 'N', 'C', ('N','CA','CB'), 180.0),
            ('CG1', 'CB', 'CA', 'N', ('CA','CB','CG1'), -60.0),
            ('CG2', 'CB', 'CA', 'N', ('CA','CB','CG2'), 60.0),
        ]
    },
}

def build_sidechain(bb_coords, target_aa):
    """
    Build sidechain atoms for a given amino acid.
    
    bb_coords: dict with keys 'N', 'CA', 'C', 'O' as numpy arrays
    target_aa: three-letter code (e.g., 'ARG')
    
    Returns: dict of {atom_name: np.array} for all heavy atoms
    """
    if target_aa not in RESIDUE_BUILD:
        return {}
    
    info = RESIDUE_BUILD[target_aa]
    coords = {}
    coords['N'] = bb_coords['N']
    coords['CA'] = bb_coords['CA']
    coords['C'] = bb_coords['C']
    if 'O' in bb_coords:
        coords['O'] = bb_coords['O']
    
    for new_atom, center_atom, prev_atom, dih_ref, angle_key, dih_angle in info['build']:
        if center_atom not in coords or prev_atom not in coords:
            continue
        if dih_ref not in coords:
            dih_ref = prev_atom
        if dih_ref not in coords:
            continue
        
        c_pos = coords[center_atom]
        p_pos = coords[prev_atom]
        dr_pos = coords[dih_ref]
        
        # Get bond length
        bl = get_bond_len({}, center_atom, new_atom)
        ba = get_angle({}, prev_atom, center_atom, new_atom)
        
        pos = build_atom(p_pos, c_pos, dr_pos, bl, ba, dih_angle)
        coords[new_atom] = pos
    
    # Filter to only sidechain atoms
    sc = {k: v for k, v in coords.items() if k in info['sc_atoms']}
    return sc


def mutate_residue_heavy(pdb_in, pdb_out, chain_id, resnum, new_resname_3, new_aa_1):
    """
    Mutate a residue in PDB by replacing its heavy atoms.
    Keeps backbone atoms (N, CA, C, O) and rebuilds sidechain.
    """
    from Bio.PDB import PDBParser, PDBIO, Select
    
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure('x', pdb_in)
    model = struct[0]
    chain = model[chain_id]
    
    try:
        residue = chain[(' ', resnum, ' ')]
    except KeyError:
        print(f"  ERROR: Residue {resnum} not found in chain {chain_id}")
        return False
    
    old_resname = residue.get_resname()
    
    # Get backbone coordinates
    bb_coords = {}
    for atom_name in ['N', 'CA', 'C', 'O']:
        try:
            bb_coords[atom_name] = residue[atom_name].get_vector().get_array()
        except KeyError:
            pass
    
    if 'N' not in bb_coords or 'CA' not in bb_coords:
        print(f"  ERROR: Missing backbone atoms for residue {resnum}")
        return False
    
    # Build new sidechain
    new_sc = build_sidechain(bb_coords, new_resname_3)
    
    if not new_sc:
        print(f"  No sidechain atoms for {new_resname_3} (probably GLY)")
    
    # Remove all old atoms from residue
    atom_ids = [a.get_id() for a in residue]
    for aid in atom_ids:
        if aid not in ['N', 'CA', 'C', 'O']:
            residue.detach_child(aid)
    
    # Rename residue
    # Need to change PDBIO output to use new name - do it via writing to string
    residue.resname = new_resname_3
    
    # Add new sidechain atoms as regular atoms (not hetero)
    from Bio.PDB.Atom import Atom
    
    # These elements match the atom names
    element_map = {
        'CB': 'C', 'CG': 'C', 'CD': 'C', 'CE': 'C', 'CZ': 'C',
        'CD1': 'C', 'CD2': 'C', 'CE1': 'C', 'CE2': 'C', 'CZ2': 'C', 'CZ3': 'C',
        'CH2': 'C', 'CG1': 'C', 'CG2': 'C', 'CD1': 'C',
        'ND1': 'N', 'ND2': 'N', 'NE': 'N', 'NE1': 'N', 'NE2': 'N',
        'NH1': 'N', 'NH2': 'N', 'NZ': 'N', 'N': 'N',
        'OD1': 'O', 'OD2': 'O', 'OE1': 'O', 'OE2': 'O', 'OG': 'O', 'OG1': 'O', 'OH': 'O',
        'SD': 'S', 'SG': 'S',
    }
    
    # Get bfactor from CA atom if possible
    bfactor = 0.0
    try:
        bfactor = residue['CA'].get_bfactor()
    except:
        pass
    
    import numpy as np
    coord_template = residue['CA'].get_vector()
    bf = residue['CA'].get_bfactor()
    
    # Use PDBIO's internal atom serial for residue? No, just append atoms
    # We need to create Atom objects
    for atom_name, pos in new_sc.items():
        # Use CA as template for occupancy and element
        element = element_map.get(atom_name, 'C')
        fullname = ' ' + atom_name.ljust(3)[:3] if len(atom_name) < 4 else atom_name
        vec = pos  # already numpy array
        
        from Bio.PDB.vectors import Vector
        new_atom = Atom(
            atom_name,
            Vector(vec[0], vec[1], vec[2]),
            bfactor,
            1.0,  # occupancy
            ' ',  # altloc
            fullname,
            None,  # serial_number
            element=element,
        )
        residue.add(new_atom)
    
    # Write the mutated structure
    io = PDBIO()
    io.set_structure(struct)
    
    # Custom select to only keep the mutated chain
    class MutSelect(Select):
        def accept_chain(self, c):
            return c.get_id() == chain_id
    
    io.save(pdb_out, MutSelect())
    return True


def prepare_pdbqt(pdb_in, pdbqt_out):
    """Convert PDB to rigid PDBQT via OpenBabel."""
    import subprocess, os
    
    flex_pdbqt = pdbqt_out + '.flex'
    result = subprocess.run(
        ['obabel', pdb_in, '-o', 'pdbqt', '-O', flex_pdbqt, '-xr'],
        capture_output=True, text=True, timeout=180
    )
    
    # Extract rigid atoms
    with open(flex_pdbqt) as f:
        lines = f.readlines()
    atoms = [l for l in lines if l.startswith(('ATOM', 'HETATM'))]
    with open(pdbqt_out, 'w') as f:
        for a in atoms:
            f.write(a)
        f.write('END\n')
    
    if os.path.exists(flex_pdbqt):
        os.remove(flex_pdbqt)
    
    return len(atoms)


if __name__ == '__main__':
    # Test: mutate rpoB PDB 451 HIS -> LEU
    import sys
    pdb_in = sys.argv[1] if len(sys.argv) > 1 else 'data/pdb/rpoB_receptor.pdb'
    chain = sys.argv[2] if len(sys.argv) > 2 else 'C'
    res = int(sys.argv[3]) if len(sys.argv) > 3 else 451
    new_aa = sys.argv[4] if len(sys.argv) > 4 else 'LEU'
    out = sys.argv[5] if len(sys.argv) > 5 else f'test_mut_{res}_{new_aa}.pdb'
    
    print(f"Testing: mutate {pdb_in} chain {chain} res {res} -> {new_aa}")
    ok = mutate_residue_heavy(pdb_in, out, chain, res, new_aa, 'L')
    print(f"  Result: {ok}")
    if ok:
        atoms = prepare_pdbqt(out, out.replace('.pdb', '.pdbqt'))
        print(f"  PDBQT atoms: {atoms}")
