#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
'''
Two parts:
1. basic variables
2. benchmark definitions and configs for data processing
'''
# 1. basic variables
PROJ_DIR = os.path.split(__file__)[0]
# cache directory
CACHE_DIR = os.path.join(PROJ_DIR, '__cache__')
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# DockQ 
# IMPORTANT: change it to your path to DockQ project)
DOCKQ_DIR = os.path.join(PROJ_DIR, 'evaluation', 'DockQ')
if not os.path.exists(DOCKQ_DIR):
    os.system(f'cd {os.path.join(PROJ_DIR, "evaluation")}; git clone --branch v1.0 --depth 1 https://github.com/bjornwallner/DockQ.git')
    os.system(f'cd {DOCKQ_DIR}; make')
