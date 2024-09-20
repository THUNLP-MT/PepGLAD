#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
from typing import List

import numpy as np

from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Structure import Structure as BStructure
from Bio.PDB.Model import Model as BModel
from Bio.PDB.Chain import Chain as BChain
from Bio.PDB.Residue import Residue as BResidue
from Bio.PDB.Atom import Atom as BAtom

from data.format import Block, Atom, VOCAB


def list_blocks_to_pdb(list_blocks: List[List[Block]], chain_names: List[str], out_path: str) -> None:
    '''
        Convert pdb file to a list of lists of blocks using Biopython.
        Each chain will be a list of blocks.
        
        Parameters:
            list_blocks: A list of lists of blocks. Each list of blocks will be parsed into one chain in the pdb
            chain_names: name of chains
            out_path: Path to the pdb file

    '''
    pdb_id = os.path.basename(os.path.splitext(out_path)[0])
    structure = BStructure(id=pdb_id)
    model = BModel(id=0)
    for blocks, chain_name in zip(list_blocks, chain_names):
        chain = BChain(id=chain_name)
        for i, block in enumerate(blocks):
            chain.add(_block_to_biopython(block, i))
        model.add(chain)
    structure.add(model)
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path)


def _block_to_biopython(block: Block, pos_code: int) -> BResidue:
    _id = (' ', pos_code, ' ')
    residue = BResidue(_id, block.abrv, '    ')
    for i, atom in enumerate(block):
        fullname = ' ' + atom.name
        while len(fullname) < 4:
            fullname += ' '
        bio_atom = BAtom(
            name=atom,
            coord=np.array(atom.coordinate, dtype=np.float32),
            bfactor=0,
            occupancy=1.0,
            altloc=' ',
            fullname=fullname,
            serial_number=i,
            element=atom.element
        )
        residue.add(bio_atom)
    return residue