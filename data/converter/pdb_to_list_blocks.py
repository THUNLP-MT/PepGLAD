#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Dict, List, Optional, Union

from Bio.PDB import PDBParser

from data.format import Block, Atom


def pdb_to_list_blocks(pdb: str, selected_chains: Optional[List[str]]=None, return_chain_ids: bool=False, dict_form: bool=False) -> Union[List[List[Block]], Dict[str, List[Block]]]:
    '''
        Convert pdb file to a list of lists of blocks using Biopython.
        Each chain will be a list of blocks.
        
        Parameters:
            pdb: Path to the pdb file
            selected_chains: List of selected chain ids. The returned list will be ordered
                according to the ordering of chain ids in this parameter. If not specified,
                all chains will be returned. e.g. ['A', 'B']
            return_chain_ids: Whether to return the ids of each chain
            dict_form: Whether to return chains in dict form (chain id as the key and blocks
                as the value)

        Returns:
            A list of lists of blocks. Each chain in the pdb file will be parsed into
            one list of blocks.
            example:
                [
                    [residueA1, residueA2, ...],  # chain A
                    [residueB1, residueB2, ...]   # chain B
                ],
                where each residue is instantiated by Block data class.
    '''

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('anonym', pdb)

    list_blocks, chain_ids, chains = [], {}, []
    
    for model in structure.get_models():  # use model 1 only
        structure = model
        break

    for chain in structure.get_chains():

        _id = chain.get_id()
        if (selected_chains is not None) and (_id not in selected_chains):
            continue

        residues, res_ids = [], {}

        for residue in chain:
            abrv = residue.get_resname()
            hetero_flag, res_number, insert_code = residue.get_id()
            res_id = f'{res_number}-{insert_code}'
            if hetero_flag == 'W':
                continue   # residue from glucose (WAT) or water (HOH)
            if hetero_flag.strip() != '' and res_id in res_ids:
                continue  # the solvent (e.g. H_EDO (EDO))
            if abrv in ['EDO', 'HOH', 'BME']:  # solvent or other molecules
                continue
            if abrv == 'MSE':
                abrv = 'MET'  # MET is usually transformed to MSE for structural analysis
                
            # filter Hs because not all data include them
            atoms = [ Atom(atom.get_id(), atom.get_coord().tolist(), atom.element) for atom in residue if atom.element != 'H' ]
            block = Block(abrv, atoms, id=(res_number, insert_code))
            if block.is_residue():
                residues.append(block)
                res_ids[res_id] = True
        
        if len(residues) == 0:  # not a chain
            continue

        chain_ids[_id] = len(list_blocks)
        list_blocks.append(residues)
        chains.append(_id)

    # reorder
    if selected_chains is not None:
        list_blocks = [list_blocks[chain_ids[chain_id]] for chain_id in selected_chains]
        chains = selected_chains

    if dict_form:
        return { chain: blocks for chain, blocks in zip(chains, list_blocks)}
    
    if return_chain_ids:
        return list_blocks, chains

    return list_blocks


if __name__ == '__main__':
    import sys
    list_blocks = pdb_to_list_blocks(sys.argv[1])
    print(f'{sys.argv[1]} parsed')
    print(f'number of chains: {len(list_blocks)}')
    for i, chain in enumerate(list_blocks):
        print(f'chain {i} lengths: {len(chain)}')