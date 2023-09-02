# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 20:20:34 2022

@author: motoyama
"""
import re, itertools, copy
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

import warnings
warnings.simplefilter('error')

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from utils import QM, BDD
import drawset


def get_unique_list(seq):
    seen = []
    return [x for x in seq if x not in seen and not seen.append(x)]


"""
変数名、制御入力名は0からカウントアップ
pbcn_model = [transition_rule1, transition_rule2, ...]
transition_rule = [funcs, probs]
funcs = [func0, func1, ...]
probs = [prob0, prob1, ...], sum(probs) = 1
x: np.array((N,), dtype=np.bool_)
u: np.array((M,), dtype=np.bool_)
"""


def state2idx(state):
    return sum((2**i)*v for i,v in enumerate(np.flip(~state)))
def idx2state(idx, l):
    return ~np.array([int(v) for v in format(idx,f'0{l}b')], dtype=np.bool_)

def calc(transition_rule: list, x: np.ndarray, u: np.ndarray=None):
    func = np.random.choice(a=transition_rule[0], size=1, p=transition_rule[1])[0]
    return eval(func)

def controller_to_func(n, controller: np.ndarray):
    """express the controller as functions"""
    m = int(np.log2(len(controller)))
    controller_funcs = [[] for _ in range(m)]
    for x_idx in range(2**n):
        u_idxs = np.where(idx2state(controller[x_idx], m))[0]   # controller_funcに項を追加するuのindex
        if len(u_idxs) != 0:
            # xを積で表す項をつくり、controller_funcに追加
            func = ' and '.join(
                f'x[{i}]' if val==True else f'not x[{i}]'
                for i,val in enumerate(idx2state(x_idx, n))
                )
            for u_idx in u_idxs:
                controller_funcs[u_idx].append(func)
    controller_funcs = [
        ' or '.join(terms) if terms != [] else 'False'
        for terms in controller_funcs
        ]
    return controller_funcs


def controller_to_func_minimum(n, controller):
    """QM法"""
    m = int(np.log2(len(controller)))
    func_lists = [[] for _ in range(m)]
    for x_idx in range(2**n):
        u_idxs = np.where(idx2state(controller[x_idx], m))[0]   # controller_funcに項を追加するuのindex
        if len(u_idxs) != 0:
            # xを積で表す項をつくり、controller_funcに追加
            func = [(i+1)*[-1,1][int(val)] for i,val in enumerate(idx2state(x_idx, n))]
            for u_idx in u_idxs:
                func_lists[u_idx].append(func)
    
    func_lists_minimum = [QM(func_list) for func_list in func_lists]
    
    # funcに直す
    controller_funcs = [
        ' or '.join(
            ' and '.join(
                f'x[{v-1}]' if 0<v else f'not x[{-v-1}]'
                for v in term)
            for term in func_list)
        if type(func_list) is list else str(bool(func_list)) for func_list in func_lists_minimum]
    
    return controller_funcs


def minimize_func(func: str):
    """入力をもたない式を簡単化する"""
    # 全ての状態に対する出力を見て、func_listを得る
    V = sorted(map(int,list(set(re.findall(r'x\[(\d+)\]', func)))))
    if len(V) == 0:
        return func
    xs = []
    for x_origin in itertools.product([1,0], repeat=len(V)):
        # 生成した入力側の変数名をVと合わせる
        x = {variable:value for variable,value in zip(V,x_origin)}
        if eval(func):
            xs.append(x_origin)
    
    func_list = [
        [
            variable+1 if value==1 else -(variable+1)
            for variable,value in zip(V,x)]
        for x in xs]
    func_list_minimum = QM(func_list)
    
    # func_listをfuncに戻す
    if func_list_minimum == 1:
        return 'True'
    func_minimum = ' or '.join(
        ' and '.join(
            f'x[{v-1}]' if 0<v else f'not x[{-v-1}]'
            for v in term)
        for term in func_list_minimum)
    
    return func_minimum


BDD = BDD()
def get_ver_from_controller(n, controller):
    # find the variables used by the controller, using BDD
    m = int(np.log2(len(controller)))
    # 積和標準形をつくる
    func_lists = [[] for _ in range(m)]
    for x_idx in range(2**n):
        u_idxs = np.where(idx2state(controller[x_idx], m))[0]   # controller_funcに項を追加するuのindex
        if len(u_idxs) != 0:
            # xを積で表す項をつくり、controller_funcに追加
            func = [(i+1)*[-1,1][int(val)] for i,val in enumerate(idx2state(x_idx, n))]
            for u_idx in u_idxs:
                func_lists[u_idx].append(func)
    
    # func_listsに従ってApply演算
    vers = set()
    for func_list in func_lists:
        # func_listのBDDを求める
        root_terms = []
        for term in func_list:
            if term == []:
                continue
            root = BDD.leaf1   # 葉ノード1
            for v in term:
                root2 = BDD.GetUnitNode(abs(v))
                if v<0:
                    root2 = BDD.NOT(root2)
                root = BDD.AND(root, root2)
            root_terms.append(root)
        root_sum = BDD.leaf0   # 葉ノード0
        for root_term in root_terms:
            root_sum = BDD.OR(root_sum, root_term)
        
        vers_term = BDD.GetV(root_sum)
        vers = vers | vers_term
    
    return vers
    

def embed_controller(pbcn_model, controller, minimum=False):
    """embed controller in pbcn_model"""
    n = len(pbcn_model)
    m = int(np.log2(len(controller)))
    controller_func = controller_to_func_minimum(n,controller) if minimum is True else controller_to_func(n, controller)
    pbn_model = copy.deepcopy(pbcn_model)
    for transition_rule in pbn_model:
        for func_idx in range(len(transition_rule[0])):
            for i in range(m):
                transition_rule[0][func_idx] = transition_rule[0][func_idx].replace(f'u[{i}]',f'({controller_func[i]})')
    return pbn_model


def pbcn_model_to_transition_list(pbcn_model, controller=None):
    """
    transition_list[x] = [[x1,x2,...],[p1,p2,...]]
    type(x) is str
    pbcn_model,controllerを与えた場合とpbn_modelを与えた場合で結果は同じはず
    """
    n = len(pbcn_model)
    if controller is not None:
        m = int(np.log2(len(controller)))
        u_space = np.array(list(itertools.product([1,0], repeat=m)), dtype=np.bool_)
    # 遷移パターンを列挙
    transition_patterns = list(
        zip(
            itertools.product(*[transition_rule[0] for transition_rule in pbcn_model]),
            map(np.prod, itertools.product(*[transition_rule[1] for transition_rule in pbcn_model]))
            )
        )
    assert abs(sum(transition_pattern[1] for transition_pattern in transition_patterns) - 1) < 1e-05    #  float値の一致判定
    
    # 遷移パターンごとの遷移を計算
    transition_list = dict()
    for x_idx,x in enumerate(itertools.product([1,0], repeat=n)):
        u = None if controller is None else u_space[controller[x_idx]]
        next_xs = np.array(
            [[eval(func,{'x':x,'u':u}) for func in funcs] for funcs,_ in transition_patterns],
            dtype=np.bool_
            )
        # 遷移先が同じものを統合する
        unique_next_xs = np.unique(next_xs, axis=0)
        probs = [
            sum(
                transition_patterns[idx][1]
                for idx in np.where((next_xs == unique_next_x).all(axis=1))[0]
                )
            for unique_next_x in unique_next_xs
            ]
        
        x = ''.join(str(int(val)) for val in x)
        unique_next_xs = [''.join(str(int(val)) for val in unique_next_x) for unique_next_x in unique_next_xs]
        transition_list[x] = [unique_next_xs, probs]
        #print(f'\rpbcn_model_to_transition_list: {x_idx+1}/{2**n}', end='')
        
    return transition_list
    



def get_NM(pbcn_model, check=False):
    content = ' '.join(' '.join(func for func in transition_rule[0]) for transition_rule in pbcn_model)
    if check is False:
        # 式に出現する変数の最も大きいものを探す
        N = max(map(int,set(re.findall(r'x\[(\d+)\]', content))), default=-1) + 1
        M = max(map(int,set(re.findall(r'u\[(\d+)\]', content))), default=-1) + 1
    else:
        # 出現する変数の番号が飛んでいないことを確認
        N = len(pbcn_model)
        x_list = set(re.findall(r'x\[(\d+)\]', content))
        u_list = set(re.findall(r'u\[(\d+)\]', content))
        x_list = np.sort([int(s) for s in set(x_list)])
        u_list = np.sort([int(s) for s in set(u_list)])
        assert N == len(x_list)
        M = len(u_list)
        assert np.all(np.arange(N) == x_list)
        assert np.all(np.arange(M) == u_list)
    return N,M


def save_pbcn_info(info: dict, path='pbcn_model.txt'):
    """txtファイルとして出力する"""
    #with open('data/pbcn_model.txt', mode='w') as f:
    with open(path, mode='w') as f:
        f.write(str(info))

def load_pbcn_info(name='pbcn_model'):
    with open(f'data/{name}.txt', mode='r', encoding="utf-8") as f:
        l = f.readline()
    #return ast.literal_eval(l)
    return eval(l)
    
    
def is_same_func(func1, func2):
    content = ' '.join([func1,func2])
    # 最も大きい数字を検知する
    N = max(map(int,set(re.findall(r'x\[(\d+)\]', content))), default=-1) + 1
    M = max(map(int,set(re.findall(r'u\[(\d+)\]', content))), default=-1) + 1
    x_space = np.array(list(itertools.product([1,0], repeat=N)), dtype=np.bool_)
    u_space = np.array(list(itertools.product([1,0], repeat=M)), dtype=np.bool_)
    # 全状態に対する出力が一致することを確認
    for x,u in itertools.product(x_space,u_space):
        assert eval(func1) == eval(func2)
    return True


def is_controlled(transition_list, target_x):
    """全ての状態がtarget_xのbasinであることを確認する"""
    # 自分の遷移前を表現する辞書を作成
    transition_list_inv = defaultdict(set)
    for x,temp in transition_list.items():
        for next_x in temp[0]:
            transition_list_inv[next_x].add(x)
    
    # 目標状態から遷移前をさかのぼる
    targets = {''.join(str(int(val)) for val in target_x)}  # 探索対象
    while targets:
        targets2 = set()
        for x in targets:
            xs_parent = transition_list_inv.pop(x, None)    # 一度探索した状態は表から削除
            if xs_parent:
                targets2 = targets2 | xs_parent
        targets = targets2
        
    return transition_list_inv

    
class gym_PBCN():
    def __init__(self, name_or_info):
        if isinstance(name_or_info,str):
            info = load_pbcn_info(name_or_info)
        else:
            info = name_or_info
        self.pbcn_model = info['pbcn_model']
        self.target_x = np.array(info['target_x'])
        
        self.n, self.m = get_NM(self.pbcn_model)
        
        self.observation_shape = (self.n,)
        self.action_shape = (self.m,)
        
        #self.count = 0      # 遷移回数
    
    def reset(self):
        self.count = 0
        self.x = np.random.randint(0, 2, self.n, dtype=np.bool_)
        return self.x
    
    def random_action(self):
        return np.random.randint(0, 2, self.m, dtype=np.bool_)
    
    def step(self, action):
        self.count += 1
        x = self.x
        u = np.array(action, dtype=np.bool_)
        next_x = np.array([
            calc(transition_rule, x, u) for transition_rule in self.pbcn_model
            ])
        
        if np.all(next_x == self.target_x):
            # 目標状態に到達したとき
            reward = 1
            done = 1
        elif np.all(next_x == self.x):
            # 前と状態が変わらないとき
            reward = -1
            done = 0
        else:
            reward = 0
            done = 0
        
        self.x = next_x
        return next_x, reward, done, None
    
        
    def step_with_controller(self, controller):
        x = self.x
        u = idx2state(controller[state2idx(self.x)],self.m)
        next_x = np.array([
            calc(transition_rule, x, u) for transition_rule in self.pbcn_model
            ])
        
        if np.all(next_x == self.target_x):
            # 目標状態に到達したとき
            done = 1
        else:
            done = 0
            
        self.x = next_x
        return next_x, done
    



if __name__ == '__main__':
    # txtファイルからインポート
    info = load_pbcn_info('pbcn_model_pinning_3 (2)')
    pbcn_model = info['pbcn_model']
    target_x = np.array(info['target_x'], dtype=np.bool_)
    controller = np.array(info['controller'], dtype=np.int32)
    
    env = gym_PBCN('pbcn_model_pinning_3 (2)')
    n = len(pbcn_model)
    m = int(np.log2(len(controller)))
    x_space = np.array(list(itertools.product([1,0], repeat=n)), dtype=np.bool_)
    u_space = np.array(list(itertools.product([1,0], repeat=m)), dtype=np.bool_)
        
    # コントローラが用いる変数を取得
    ver = get_ver_from_controller(n, controller)
    
    # コントローラを式にしてpbcnに埋め込む
    controller_funcs = controller_to_func(n, controller)
    controller_funcs_minimum = controller_to_func_minimum(n, controller)
    controller_funcs_minimum2 = [minimize_func(controller_func) for controller_func in controller_funcs]
    [is_same_func(controller_func,controller_func_minimum) for controller_func,controller_func_minimum in zip(controller_funcs,controller_funcs_minimum)]
    pbn_model = embed_controller(pbcn_model, controller, minimum=True)
    transition_list = pbcn_model_to_transition_list(pbn_model)
    transition_list2 = pbcn_model_to_transition_list(pbcn_model,controller)
    is_controlled(transition_list, target_x)
    drawset.transition_diagram(transition_list)
    
    
    # コントローラで状態遷移を行う
    num_try = 100
    time = 150
    data = np.empty((num_try,time, n), dtype=np.bool_)
    for n in range(num_try):
        env.reset()
        #env.state = 2
        for t in range(time):
            next_x, done = env.step_with_controller(controller)
            data[n,t] = next_x
    # 各試行の平均をとる
    data_mean = np.mean(data, axis=0)
    
    for i in range(len(data_mean[0])):
        plt.plot(data_mean[:,i], label=f'x{i}')
    plt.legend()
    plt.show()
    
    save_pbcn_info(
        {
            'pbcn_model': pbcn_model,
            'target_x': target_x.tolist(),
            'controller': controller
            }
        )


    a = state2idx(np.array([True,False,False]))
    b = np.flip(~np.array([True,False,False]))
    c = [(2**i)*v for i,v in enumerate(b)]











