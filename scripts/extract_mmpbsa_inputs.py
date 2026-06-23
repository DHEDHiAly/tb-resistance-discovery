"""
Extract MMPBSA-compatible PDB files from an OpenMM simulation.

Handles the common case where the ligand was added programmatically
(not present in the Modeller topology) by accepting a separate
ligand Topology + positions.

Produces three PDB files for MMPBSA.py:
  - complex.pdb  : protein (from Modeller topology) + ligand (from ligand_top)
  - protein.pdb  : protein only (solvent stripped)
  - ligand.pdb   : ligand only (with CONECT records)

Usage:
    write_mmpbsa_pdbs(positions, prot_topology,
                      ligand_topology, ligand_positions,
                      out_dir='analysis/results')
"""
import numpy as np
import openmm.app as app
import openmm.unit as u
import os


def _strip_solvent(topology, positions, solvent_names=None):
    """
    Remove solvent residues from topology/positions.

    Returns (new_top, new_positions, atom_map).
    """
    if solvent_names is None:
        solvent_names = {'HOH', 'WAT', 'NA', 'CL', 'K', 'NA+', 'CL-',
                         'K+', 'Mg', 'Mg2+', 'Ca', 'Ca2+', 'Na+', 'Cl-'}

    from openmm.app import Topology
    new_top = Topology()
    atom_map = {}

    keep = []
    for chain in topology.chains():
        for res in chain.residues():
            if res.name.strip() not in solvent_names:
                keep.append(res)
    keep.sort(key=lambda r: (r.chain.index, r.id))

    chain_map = {}
    for res in keep:
        cid = res.chain.id if res.chain.id else 'A'
        if cid not in chain_map:
            chain_map[cid] = new_top.addChain(cid)
        new_chain = chain_map[cid]
        new_res = new_top.addResidue(res.name, new_chain, res.id)
        for atom in res.atoms():
            new_atom = new_top.addAtom(atom.name, atom.element, new_res)
            atom_map[atom.index] = new_atom.index

    new_pos = [positions[i] for i in sorted(atom_map.keys())]
    return new_top, new_pos, atom_map


def _copy_bonds(source_top, dest_top, atom_map):
    """Transfer bonds from source_top to dest_top via atom_map."""
    atoms_list = list(dest_top.atoms())
    for bond in source_top.bonds():
        a1, a2 = bond
        if a1.index in atom_map and a2.index in atom_map:
            dest_top.addBond(atoms_list[atom_map[a1.index]],
                             atoms_list[atom_map[a2.index]])


def _pdb_write(topology, positions, path):
    """Write a PDB file handling Vec3 unit conversion."""
    arr = np.array([[p[0]._value, p[1]._value, p[2]._value]
                    if hasattr(p[0], '_value') else [p[0], p[1], p[2]]
                    for p in positions]) * u.nanometer
    with open(path, 'w') as f:
        app.PDBFile.writeFile(topology, arr, f)


def _ensure_chain_id(topology, chain_id='L'):
    """If topology has no chains, create one with the given id."""
    chains = list(topology.chains())
    if not chains:
        from openmm.app import Topology
        new_top = Topology()
        for res in topology.residues():
            c = new_top.addChain(chain_id)
            r = new_top.addResidue(res.name, c, res.id)
            for atom in res.atoms():
                a = new_top.addAtom(atom.name, atom.element, r)
        for bond in topology.bonds():
            a1 = list(new_top.atoms())[list(topology.atoms()).index(bond[0])]
            a2 = list(new_top.atoms())[list(topology.atoms()).index(bond[1])]
            new_top.addBond(a1, a2)
        return new_top
    return topology


def write_mmpbsa_pdbs(positions,
                       prot_topology,
                       ligand_topology,
                       ligand_positions,
                       out_dir='analysis/results'):
    """
    Write complex.pdb, protein.pdb, ligand.pdb for MMPBSA.py.

    Parameters
    ----------
    positions : list of openmm.Vec3
        Combined protein + ligand positions (protein first, then ligand).
    prot_topology : openmm.app.Topology
        Solvated protein topology (from Modeller).
    ligand_topology : openmm.app.Topology
        Ligand topology (small molecule, 1 residue, with bonds).
    ligand_positions : list of openmm.Vec3
        Ligand atomic positions (must match ligand_topology).
    out_dir : str
        Output directory.
    """
    os.makedirs(out_dir, exist_ok=True)

    n_prot = prot_topology.getNumAtoms()
    n_lig = ligand_topology.getNumAtoms()

    # ---- 1. Strip solvent from protein topology ----
    prot_stripped, prot_stripped_pos, prot_map = _strip_solvent(
        prot_topology, positions[:n_prot])
    _copy_bonds(prot_topology, prot_stripped, prot_map)

    # ---- 2. Write protein-only PDB ----
    _pdb_write(prot_stripped, prot_stripped_pos,
               os.path.join(out_dir, 'protein.pdb'))
    print(f"  Wrote protein.pdb ({prot_stripped.getNumAtoms()} atoms)")

    # ---- 3. Write ligand-only PDB (ensure chain id) ----
    lig_top = _ensure_chain_id(ligand_topology, 'L')
    _pdb_write(lig_top, ligand_positions,
               os.path.join(out_dir, 'ligand.pdb'))
    print(f"  Wrote ligand.pdb ({lig_top.getNumAtoms()} atoms)")

    # ---- 4. Build combined complex PDB ----
    from openmm.app import Topology
    complex_top = Topology()
    atom_map_c = {}  # old (topology, index) -> new index
    offset = 0

    # Copy protein residues
    prot_chain = complex_top.addChain('A')
    for res in prot_stripped.residues():
        new_res = complex_top.addResidue(res.name, prot_chain, res.id)
        for atom in res.atoms():
            a = complex_top.addAtom(atom.name, atom.element, new_res)
            atom_map_c[('prot', atom.index)] = a.index

    # Copy ligand residues
    lig_chain = complex_top.addChain('B')
    for res in lig_top.residues():
        new_res = complex_top.addResidue(res.name, lig_chain, res.id)
        for atom in res.atoms():
            a = complex_top.addAtom(atom.name, atom.element, new_res)
            atom_map_c[('lig', atom.index)] = a.index

    # Copy bonds
    all_atoms = list(complex_top.atoms())
    for bond in prot_stripped.bonds():
        i1 = atom_map_c[('prot', bond[0].index)]
        i2 = atom_map_c[('prot', bond[1].index)]
        complex_top.addBond(all_atoms[i1], all_atoms[i2])
    for bond in lig_top.bonds():
        i1 = atom_map_c[('lig', bond[0].index)]
        i2 = atom_map_c[('lig', bond[1].index)]
        complex_top.addBond(all_atoms[i1], all_atoms[i2])

    complex_pos = list(prot_stripped_pos) + list(ligand_positions)
    _pdb_write(complex_top, complex_pos,
               os.path.join(out_dir, 'complex.pdb'))
    n_cpx = complex_top.getNumAtoms()
    print(f"  Wrote complex.pdb ({n_cpx} atoms)")

    # ---- 5. Validate ----
    n_p = prot_stripped.getNumAtoms()
    n_l = lig_top.getNumAtoms()
    ok = n_p + n_l == n_cpx
    print(f"\n  Validation: protein({n_p}) + ligand({n_l}) "
          f"= {n_p + n_l} vs complex({n_cpx}) {'OK' if ok else 'MISMATCH!'}")

    print(f"\nMMPBSA input files ready in {os.path.abspath(out_dir)}/")
    print("Run: MMPBSA.py -O -i mmpbsa.in \\")
    print(f"        -p {out_dir}/complex.pdb \\")
    print(f"        -y trajectory.dcd")


def write_mmpbsa_input(ligand_mask=':676', out_dir='analysis/results'):
    """Write a default mmpbsa.in control file."""
    os.makedirs(out_dir, exist_ok=True)
    content = f"""\
&general
  startframe=200, endframe=1000, interval=2,
  verbose=2,
  ligand_mask={ligand_mask},
/
&pb
  istrng=0.150, radiopt=0,
/
"""
    path = os.path.join(out_dir, 'mmpbsa.in')
    with open(path, 'w') as f:
        f.write(content.lstrip())
    print(f"  Wrote {path}")
    return path
