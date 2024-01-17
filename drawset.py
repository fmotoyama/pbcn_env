# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 13:07:27 2020

@author: f.motoyama
"""

import os, itertools 
from copy import copy
import numpy as np
from graphviz import Digraph




def transition_diagram(transition_list: dict, fname='transition_diagram'): 
    """transition_list[x] = [[x1,x2,...],[p1,p2,...]]"""
    
    G = Digraph(format='pdf',engine='dot')#dot twopi
    G.attr(rankdir='LR') #'TB'
    G.attr('graph',dpi='300')#,ratio='0.5')
    #G = Digraph(format='png', engine='circo')
    #G.attr('node', shape='circle', fixedsize='true')#, width='0.75', fontsize='20')
    for x,data in transition_list.items():
        for next_x,prob in zip(*data):
            if x == next_x:
                #G.node(x,shape='doublecircle',color='red')
                G.edge(x, next_x, label=str(round(prob,5)))
            else:
                G.edge(x, next_x, label=str(round(prob,5)))
    
           
    #print(G)
    G.render(f'./figure/{fname}')
    os.remove(f'./figure/{fname}')


if __name__ == '__main__':
    import pbcn
    
    #info = pbcn.load_pbcn_info()
    pbn_model = [
        [['x[0] and x[1] and x[2] or x[1] and not x[2] and not x[0] or x[2] and not x[0] and not x[1]', 'x[0] and x[1] and x[2] or not x[2] and not x[0] or not x[0] and not x[1]'], [0.3, 0.7]],
        [['x[0] or x[2]', 'x[0] or x[2] or not x[1]'], [0.3, 0.7]],
        [['x[2] and not x[0] or x[1]'], [1]]
        ]
    pbcn_model = [
        [['(x[0] and x[1] and x[2] or x[1] and not x[2] and not x[0] or x[2] and not x[0] and not x[1]) ^ (x[0] and x[2] or x[0] and not x[1])', '(x[0] and x[1] and x[2] or not x[2] and not x[0] or not x[0] and not x[1]) ^ (x[0] and x[2] or x[0] and not x[1])'], [0.3, 0.7]],
        [['x[0] or x[2]', 'x[0] or x[2] or not x[1]'], [0.3, 0.7]],
        [['x[2] and not x[0] or x[1]'], [1]]
        ]
    transition_diagram(pbcn.pbcn_model_to_transition_list(pbn_model), 'td1')





