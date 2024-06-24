# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 13:07:27 2020

@author: f.motoyama
"""

import os, itertools 
from copy import copy
import numpy as np
from graphviz import Digraph




def transition_diagram(transition_list: dict, fname='transition_diagram', format_='pdf'): 
    """transition_list[x] = [[x1,x2,...],[p1,p2,...]]"""
    
    G = Digraph(format=format_, engine='dot')#dot twopi
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


def wiring_diagram(parent: dict, fname='wiring_diagram', format_='svg'):
    G = Digraph(format=format_, engine='sfdp')
    G.attr('node', shape='circle')
    G.attr('graph', splines = 'curved', overlap = '0:')
    G.attr('edge', arrowsize = '0.5', color="#00000080")
    
    for key in parent:
        for p in parent[key]:
            G.edge(str(p),str(key))
    #図を保存
    G.render(f'./figure/{fname}')
    os.remove(f'./figure/{fname}')


if __name__ == '__main__':
    import pbcn
    #"""
    pbcn_model = pbcn.load_pbcn_info('pbcn_model_10')['pbcn_model']
    """
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
    #"""
    #transition_diagram(pbcn.pbcn_model_to_transition_list(pbn_model), 'td1')
    
    # 1始まりにする
    parent = pbcn.pbcn_model_to_parent(pbcn_model)
    parent = {i+1: [str(int(v)+1) for v in V_set] for i,V_set in parent.items()}
    wiring_diagram(parent, 'wd1')





