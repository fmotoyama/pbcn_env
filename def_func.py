# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 13:39:02 2023

@author: f.motoyama
"""
import itertools, copy
import numpy as np

from .utils import QM

def def_f(mode, **kwargs):
    """ランダムにfuncを生成する"""
    
    def import_(name='pbcn_model.txt'):
        #dataディレクトリにあるfile_nameからf_dictを得る
        name = f'data/{name}.txt'
        with open(name, mode='r', encoding="utf-8") as f:
            l = f.readline()
        #return ast.literal_eval(l)
        return eval(l)
    
    
    def random1(n, gamma=3.0, reduce=True):
        # 状態遷移図をランダムなつなぎ方をする
        x_space = np.array(list(itertools.product([1,0], repeat=n)), dtype=np.bool_)
        
        # スケールフリーっぽく状態遷移図を作成
        # 入り次数は、i個の要素の変更により必ず2**i (i=0,1,...,n)
        p = np.array([1/(i**gamma) for i in range(1,n+2)])
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
        if reduce:
            func_listss = [
                [QM(func_list) for func_list in func_lists]
                for func_lists in func_listss
                ]
        # funcに変換
        pbn_model = [
            [
                [
                    ' or '.join(
                        ' and '.join(f'x[{v-1}]' if 0<v else f'not x[{-v-1}]'for v in term)
                        for term in func_list
                        ) if isinstance(func_list,list) else str(bool(func_list))
                    for func_list in func_lists
                    ],
                #[0.3,0.7] if len(func_lists)==2 else [1]
                ]
            for func_lists in func_listss
            ]
        # 確率を付与
        for transition_rule in pbn_model:
            if len(transition_rule[0])==2:
                rand = np.random.randint(1,10) / 10     # 0.1, 0.2,..., 0.9
                transition_rule.append([rand, 1-rand])
            else:
                transition_rule.append([1])
        
        return {'pbcn_model': pbn_model}
    
    
    def random2(n, n_div, gamma=3.0, max_func_length=None, reduce=True, avoid_constants=True):
        """
        初めに使う変数を決め、その変数を使ったランダムな式を生成する
        n_div: 分岐を起こす要素の個数
        max_func_length: 1つの要素がもつ変数の最大数
        avoid_constants: 定数となった式を作り直す
        """
        if max_func_length is None:
            max_func_length = n
        V = list(range(1,n+1))
        
        # 式を生成
        p = np.array([1/(i**gamma) for i in range(1,max_func_length+1)])    
        p /= np.sum(p)      # 逆べき乗則の確率分布
        func_lists = []
        for _ in range(n + n_div):
            V_subset = np.random.choice(
                V,
                size=np.random.choice(range(1,max_func_length+1), p=p),
                replace=False
                )
            func_lists.append(make_random_func_list(V_subset, reduce=reduce, avoid_constants=avoid_constants))
        # 文字列に変換
        funcs = [
            ' or '.join(
                ' and '.join(f'x[{v-1}]' if 0<v else f'not x[{-v-1}]'for v in term)
                for term in func_list
                ) if isinstance(func_list,list) else str(bool(func_list))
            for func_list in func_lists
            ]
        
        # 分岐をもつ要素を決定
        V_div = np.random.choice(V, size=n_div, replace=False)
        # モデルの完成
        pbn_model = []
        index = 0
        for v in V:
            if v in V_div:
                funcs_subset = [funcs[index],funcs[index+1]]
                index += 2
                rand = np.random.randint(1,10) / 10     # 0.1, 0.2,..., 0.9
                probs = [rand, 1-rand]
            else:
                funcs_subset = [funcs[index]]
                index += 1
                probs = [1]
            pbn_model.append([funcs_subset, probs])
        return {'pbcn_model': pbn_model}
    

    if mode == 'import':
        info = import_(**kwargs)
    elif mode == 'random1':
        info = random1(**kwargs)
    elif mode == 'random2':
        info = random2(**kwargs)
    else:
        raise Exception
    return info

def make_scalefree(n, gamma):
    """"
    スケールフリーネットワークっぽく、各ノードの親ノード(1~n個)をランダムに決定してpnodeを返す
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


def make_random_func_list(n_or_V, p=0.5, reduce=False, avoid_constants=False):
    """
    ランダムな真理値表から式を生成する
    1~nの全ての変数が使われるとは限らない
    pの確率で、入力に対する出力が1に(項が生成される)
    """
    while True:
        if isinstance(n_or_V,int):
            n = n_or_V
            V = list(range(1,n+1))
        else:
            n = len(n_or_V)
            V = n_or_V
        func_list = []
        for x in itertools.product([1,0], repeat=n):
            if np.random.rand() < p:
                func_list.append([-v if val==0 else v for v,val in zip(V,x)])
        
        if reduce:
            func_list = QM(func_list)
            if avoid_constants and isinstance(func_list,int):
                continue
        break
    
    return func_list
        
def add_pinning_node(pbn_model, V=None):
    pbcn_model = copy.deepcopy(pbn_model)
    if V is None:
        V = list(range(len(pbcn_model)))
    for v in V:
        funcs = [
            ''.join(['(', func, f') ^ u[{v}]'])
            for func in pbcn_model[v][0]
            ]
        pbcn_model[v][0] = funcs
    return pbcn_model
    
    


if __name__ == '__main__':
    import pbcn
    import drawset
    #info = def_f('import', name='pbcn_model_28')
    #info = def_f('random', n=3, gamma=2.5)
    info = def_f('random2', n=5, n_div=3, gamma=2, reduce=True)
    #info = {'pbcn_model': [[['x[0] and x[1] and x[2] or x[1] and not x[2] and not x[0] or x[2] and not x[0] and not x[1]', 'x[0] and x[1] and x[2] or not x[2] and not x[0] or not x[0] and not x[1]'], [0.3, 0.7]], [['x[0] or x[2]', 'x[0] or x[2] or not x[1]'], [0.3, 0.7]], [['x[2] and not x[0] or x[1]'], [1]]]}
    
    pbn_model = info['pbcn_model']
    #transition_list = PBCN.pbn_model_to_transition_list(info['pbcn_model'])
    #drawset.transition_diagram(transition_list, 'temp')
    pbcn_model = add_pinning_node(pbn_model)
    
    num = [len(transition_rule[0]) for transition_rule in pbn_model]
















