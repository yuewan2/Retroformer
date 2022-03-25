import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

try:
    # from cairosvg import svg2png
    from IPython.display import SVG, display
    from rdkit.Chem.Draw import IPythonConsole
except:
    pass


def draw_rxn(gt, size=(900, 300), highlight=False):
    rxn = AllChem.ReactionFromSmarts(gt)
    d = Draw.MolDraw2DSVG(size[0], size[1])
    colors = [(1, 0.6, 0.6), (0.4, 0.6, 1)]

    d.DrawReaction(rxn, highlightByReactant=highlight, highlightColorsReactants=colors)
    d.FinishDrawing()
    svg = d.GetDrawingText()
    svg2 = svg.replace('svg:', '')
    svg3 = SVG(svg2)
    display(svg3)


def get_pair(atoms):
    atom_pairs = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            atom_pairs.append((atoms[i], atoms[j]))
    return atom_pairs


def get_bonds(mol, atoms):
    atom_pairs = get_pair(atoms)
    bond_list = []
    for ap in atom_pairs:
        bond = mol.GetBondBetweenAtoms(*ap)
        if bond is not None:
            bond_list.append(bond.GetIdx())
    return list(set(bond_list))


def draw_mols(smiles_list, smarts_list=None, noise_smarts_list=None, highlight=False,
              size=(500, 500), save_path='', color=None):
    if color is None:
        color1 = matplotlib.colors.to_rgb('lightcoral')
    else:
        color1 = color
    color2 = matplotlib.colors.to_rgb('cornflowerblue')  # cornflowerblue; darkseagreen
    mol_list, matched_atom_list, noise_matched_atom_list = [], [], []
    matched_bond_list, noise_matched_bond_list = [], []

    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        mol_list.append(mol)
        if smarts_list is not None:
            patt = Chem.MolFromSmarts(smarts_list[i])
            atoms_matched = mol.GetSubstructMatch(patt)
            bonds_matched = get_bonds(mol, atoms_matched)
            matched_atom_list.append(atoms_matched)
            matched_bond_list.append(bonds_matched)

        if noise_smarts_list is not None:
            patt = Chem.MolFromSmarts(noise_smarts_list[i])
            atoms_matched = mol.GetSubstructMatch(patt)
            bonds_matched = get_bonds(mol, atoms_matched)
            noise_matched_atom_list.append(atoms_matched)
            noise_matched_bond_list.append(bonds_matched)

        if smarts_list is None and highlight:
            atoms_matched = []
            for atom in mol.GetAtoms():
                if atom.HasProp('molAtomMapNumber'):
                    atom.ClearProp('molAtomMapNumber')
                    atoms_matched.append(atom.GetIdx())
            bonds_matched = get_bonds(mol, atoms_matched)
            matched_atom_list.append(atoms_matched)
            matched_bond_list.append(bonds_matched)

    all_matched_atom_list, all_matched_bond_list = [], []
    all_matched_atom_color, all_matched_bond_color = [], []

    for i in range(len(matched_atom_list)):
        if len(noise_matched_atom_list):
            all_matched_atom_list.append(matched_atom_list[i] + noise_matched_atom_list[i])
            all_matched_bond_list.append(matched_bond_list[i] + noise_matched_bond_list[i])
            atom2color = {a: color1 for a in matched_atom_list[i]}
            atom2color.update({a: color2 for a in noise_matched_atom_list[i]})
            bond2color = {b: color1 for b in matched_bond_list[i]}
            bond2color.update({b: color2 for b in noise_matched_bond_list[i]})
        else:
            all_matched_atom_list.append(matched_atom_list[i])
            all_matched_bond_list.append(matched_bond_list[i])
            atom2color = {a: color1 for a in matched_atom_list[i]}
            bond2color = {b: color1 for b in matched_bond_list[i]}

        all_matched_atom_color.append(atom2color)
        all_matched_bond_color.append(bond2color)

    svg = Draw.MolsToGridImage(mol_list, subImgSize=size, useSVG=True, molsPerRow=2,
                               highlightAtomLists=all_matched_atom_list,
                               highlightAtomColors=all_matched_atom_color,
                               highlightBondLists=all_matched_bond_list,
                               highlightBondColors=all_matched_bond_color)
    # svg = SVG(svg)
    display(svg)
    if save_path:
        svg2png(bytestring=svg.data, write_to=save_path)


def draw_mol_by_reaction_center(smiles, atom_scores, bond_scores, size=(400, 400), color_name='Oranges'):
    cmap = plt.get_cmap(color_name)
    mol = Chem.MolFromSmiles(smiles)
    plot_atoms_colors, plot_bonds_colors = [], []
    plot_atoms_matched, plot_bonds_matched = [], []
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atom_score = atom_scores[int(atom.GetProp('molAtomMapNumber'))]
            plot_atoms_colors.append(cmap(atom_score))
            plot_atoms_matched.append(atom.GetIdx())

            for n_atom in atom.GetNeighbors():
                bond_score = bond_scores[int(atom.GetProp('molAtomMapNumber')),
                                         int(n_atom.GetProp('molAtomMapNumber'))]
                bond = mol.GetBondBetweenAtoms(atom.GetIdx(), n_atom.GetIdx())
                plot_bonds_colors.append(cmap(bond_score))
                plot_bonds_matched.append(bond.GetIdx())

    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atom.ClearProp('molAtomMapNumber')

    atom2color = {plot_atoms_matched[i]: plot_atoms_colors[i] for i in range(len(plot_atoms_colors))}
    bond2color = {plot_bonds_matched[i]: plot_bonds_colors[i] for i in range(len(plot_bonds_colors))}
    svg = Draw.MolsToGridImage([mol], subImgSize=size, useSVG=True, molsPerRow=2,
                               highlightAtomLists=[plot_atoms_matched],
                               highlightAtomColors=[atom2color],
                               highlightBondLists=[plot_bonds_matched],
                               highlightBondColors=[bond2color])
    display(svg)
