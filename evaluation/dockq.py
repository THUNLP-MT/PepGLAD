#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re

from globals import DOCKQ_DIR


def dockq(mod_pdb: str, native_pdb: str, pep_chain: str):
    p = os.popen(f'{os.path.join(DOCKQ_DIR, "DockQ.py")} {mod_pdb} {native_pdb} -model_chain1 {pep_chain} -native_chain1 {pep_chain} -no_needle')
    text = p.read()
    p.close()
    res = re.search(r'DockQ\s+([0-1]\.[0-9]+)', text)
    score = float(res.group(1))
    return score