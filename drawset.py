# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 13:07:27 2020

@author: f.motoyama
"""

import os, itertools 
from copy import copy
import numpy as np
from graphviz import Digraph




def transition_diagram(transition_table, fname='transition_diagram'): 
    """transition_table[x] = [[x1,x2,...],[p1,p2,...]]"""
    
    G = Digraph(format='png',engine='dot')#dot twopi
    G.attr(rankdir='LR') #'TB'
    G.attr('graph',dpi='300')#,ratio='0.5')
    #G = Digraph(format='png', engine='circo')
    #G.attr('node', shape='circle', fixedsize='true')#, width='0.75', fontsize='20')
    for x,data in transition_table.items():
        for next_x,prob in zip(*data):
            if x == next_x:
                #G.node(x,shape='doublecircle',color='red')
                G.edge(x, next_x, label=str(round(prob,5)))
            else:
                G.edge(x, next_x, label=str(round(prob,5)))
    
           
    #print(G)
    G.render(f'./figure/{fname}')
    os.remove(f'./figure/{fname}')








