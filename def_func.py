# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 13:39:02 2023

@author: f.motoyama
"""
import itertools
import numpy as np

from pbcn import PBCN
from utils import QM

def def_f(mode, **kwargs):
    """ランダムにfuncを生成する"""
    if mode == 'import':
        #同じディレクトリにあるfile_nameからf_dictを得る
        name = f'data/{kwargs["name"]}.txt' if kwargs else 'data/pbcn_model.txt'
        with open(name, mode='r', encoding="utf-8") as f:
            l = f.readline()
        #return ast.literal_eval(l)
        info = eval(l)
    
    elif mode == 'random':
        # 状態遷移図をランダムなつなぎ方をする
        default = {
            'n': 20,
            'gamma': 3.0     # gammaを大きくすると疎になる
            }
        kwargs = {**default, **kwargs}
        
        n = kwargs['n']
        x_space = np.array(list(itertools.product([1,0], repeat=n)), dtype=np.bool_)
        
        # スケールフリーっぽく状態遷移図を作成
        # 入り次数は、i個の要素の変更により必ず2**i (i=0,1,...,n)
        p = np.array([1/(i**kwargs['gamma']) for i in range(1,n+2)])
        p /= np.sum(p)      # 逆べき乗則の確率分布
        v_change = [        # v_change[x_idx] = 分岐を起こす変数の集合
            np.random.choice(n, np.random.choice(range(n+1), p=p), replace=False)
            for _ in range(2**n)
            ]
        # 各状態遷移で分岐を追加する
        output = np.random.randint(0,2,(n,2**n), dtype=np.int8)
        for x_idx in range(2**n):
            output[v_change[x_idx],x_idx] = 2
        
        # 式を作成
        v_change = set(itertools.chain.from_iterable(v_change))    # v_changeを平坦化
        func_listss = [
            [[],[]] if v in v_change else [[]]
            for v in range(n)
            ]
        for x_idx in range(2**n):
            term = [(i+1)*[-1,1][int(val)] for i,val in enumerate(x_space[x_idx])]
            # vの全ての式に項を渡す
            for v in np.where(output[:,x_idx] == 1)[0]:
                for func_list in func_listss[v]:
                    func_list.append(term)
            # 分岐するとき、どちらかの分岐に項を渡す
            for v in np.where(output[:,x_idx] == 2)[0]:
                func_listss[v][np.random.randint(2)].append(term)
        # 簡単化
        func_listss = [
            [QM(func_list) for func_list in func_lists]
            for func_lists in func_listss
            ]
        # funcに変換
        pbcn_model = [
            [
                [
                    ' or '.join(
                        ' and '.join(f'x[{v-1}]' if 0<v else f'not x[{-v-1}]'for v in term)
                        for term in func_list
                        ) if isinstance(func_list,list) else str(bool(func_list))
                    for func_list in func_lists
                    ],
                [0.3,0.7] if len(func_lists)==2 else [1]
                ]
            for func_lists in func_listss
            ]
        
        info = {'pbcn_model': pbcn_model}
        
    
    
    
    return info


def make_scalefree(n, gamma):
    """"
    スケールフリーネットワークっぽく、各ノードの親ノードをランダムに決定してpnodeを返す
    n                  :ノード数
    p(k) = 1/k^gamma   :ノードが次数kをもつ確率の指数　gammaを大きくすると疎になる
    """
    # 逆べき乗則の確率分布
    p = np.array([1/(i**gamma) for i in range(1,n+1)])
    p /= np.sum(p)
    #ノードxの入り次数と親ノードの決定
    pnode = {}
    for x in range(n):
        dim = np.random.choice(range(1,n+1), p=p)
        pnode[x] = sorted(np.random.choice(range(n), dim, replace=False))
    return pnode


if __name__ == '__main__':
    import drawset
    #info = def_f('import', name='pbcn_model_28')
    info = def_f('random', n=3, gamma=2.5)
    info = {'pbcn_model': [[['x[0] and x[1] and x[2] or x[1] and not x[2] and not x[0] or x[2] and not x[0] and not x[1]', 'x[0] and x[1] and x[2] or not x[2] and not x[0] or not x[0] and not x[1]'], [0.3, 0.7]], [['x[0] or x[2]', 'x[0] or x[2] or not x[1]'], [0.3, 0.7]], [['x[2] and not x[0] or x[1]'], [1]]]}
    
    transition_list = PBCN.pbn_model_to_transition_list(info['pbcn_model'])
    drawset.transition_diagram(transition_list, 'temp')
    
















