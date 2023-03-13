# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 20:20:34 2022

@author: motoyama
"""
import re, itertools, copy, ast
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.simplefilter('error')

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from utils import QM, BDD
import drawset


def get_unique_list(seq):
    seen = []
    return [x for x in seq if x not in seen and not seen.append(x)]

class PBCN:
    def __init__(self, pbcn_model):
        """
        変数名、制御入力名は0からカウントアップ
        pbcn_model = [transition_rule1, transition_rule2, ...]
        transition_rule = [funcs, probs]
        funcs = [func0, func1, ...]
        probs = [prob0, prob1, ...], sum(prob) = 1
        x: np.array((N,), dtype=np.bool_)
        u: np.array((M,), dtype=np.bool_)
        (x = self.x_space[x_idx])
        (x_idx = int(''.join(str(int(val)) for val in reversed(x)),2))
        """
        self.pbcn_model = pbcn_model
        self.N, self.M = self.get_NM(self.pbcn_model)
        self.x_space = np.array(list(itertools.product([1,0], repeat=self.N)), dtype=np.bool_)
        self.u_space = np.array(list(itertools.product([1,0], repeat=self.N)), dtype=np.bool_)
        
        self.BDD = BDD()
    
    
    def convert(self,x_):
        # xのインデックス表示/値表示を変換する
        if isinstance(x_, np.ndarray):
            return sum([(2**i)*v for i,v in enumerate(np.flip(~x_))])
        else:
            # int型、np.int64型を想定
            return self.x_space[x_]
    
    @staticmethod
    def calc(transition_rule: list, x: np.ndarray, u: np.ndarray=None):
        func = np.random.choice(a=transition_rule[0], size=1, p=transition_rule[1])[0]
        return eval(func)
    
    
    def controller_to_func(self, controller: np.ndarray):
        """express the controller as functions"""
        controller_funcs = [[] for _ in range(self.M)]
        for x_idx in range(2**self.N):
            u_idxs = np.where(self.u_space[controller[x_idx]])[0]   # controller_funcに項を追加するuのindex
            if len(u_idxs) != 0:
                # xを積で表す項をつくり、controller_funcに追加
                func = ' and '.join(
                    f'x[{i}]' if val==True else f'not x[{i}]'
                    for i,val in enumerate(self.x_space[x_idx])
                    )
                for u_idx in u_idxs:
                    controller_funcs[u_idx].append(func)
        controller_funcs = [
            ' or '.join(terms) if terms != [] else 'False'
            for terms in controller_funcs
            ]
        return controller_funcs
    
    
    def get_ver_from_controller(self, controller):
        # find the variables used by the controller, using BDD
        
        # 積和標準形をつくる
        func_lists = [[] for _ in range(self.M)]
        for x_idx in range(2**self.N):
            u_idxs = np.where(self.u_space[controller[x_idx]])[0]   # controller_funcに項を追加するuのindex
            if len(u_idxs) != 0:
                # xを積で表す項をつくり、controller_funcに追加
                func = [(i+1)*[-1,1][int(val)] for i,val in enumerate(self.x_space[x_idx])]
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
                root = self.BDD.GetLeafNode(False)  # 葉ノード1
                for v in term:
                    root2 = self.BDD.GetUnitNode(abs(v), v<0)
                    root = self.BDD.AP('AND', root, root2)
                root_terms.append(root)
            root_sum = self.BDD.GetLeafNode(True)   # 葉ノード0
            for root_term in root_terms:
                root_sum = self.BDD.AP('OR', root_sum, root_term)
            
            vers_term = self.BDD.GetV(root_sum)
            vers = vers | vers_term
        
        return vers
        
    
    def embed_controller(self, controller, minimum=False):
        """embed controller in pbcn_model"""
        controller_func = self.controller_to_func_minimum(controller) if minimum == True else self.controller_to_func(controller)
        pbn_model = copy.deepcopy(self.pbcn_model)
        for transition_rule in pbn_model:
            for func_idx in range(len(transition_rule[0])):
                for i in range(self.M):
                    transition_rule[0][func_idx] = transition_rule[0][func_idx].replace(f'u[{i}]',f'({controller_func[i]})')
        return pbn_model
    
    @staticmethod
    def pbn_model_to_transition_list(pbn_model):
        """
        transition_table[x] = [[x1,x2,...],[p1,p2,...]]
        type(x) is str
        """
        N,M = PBCN.get_NM(pbn_model)
        assert M==0
        x_space = np.fliplr(np.array(list(itertools.product([0,1], repeat=N)), dtype=np.bool_))
        # 遷移パターンを列挙
        transition_patterns = list(
            zip(
                itertools.product(*[transition_rule[0] for transition_rule in pbn_model]),
                map(np.prod, itertools.product(*[transition_rule[1] for transition_rule in pbn_model]))
                )
            )
        assert sum(transition_pattern[1] for transition_pattern in transition_patterns) == 1
        
        # 遷移パターンごとの遷移を計算
        transition_table = dict()
        for x in x_space:
            next_xs = np.array(
                [[eval(func,{'x':x}) for func in funcs] for funcs,_ in transition_patterns],
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
            transition_table[x] = [unique_next_xs, probs]
            
        return transition_table
        
        
    @staticmethod
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
    
    
    @staticmethod
    def save_pbcn_info(**kwargs):
        """txtファイルとして出力する"""
        with open('pbcn_model.txt', mode='w') as f:
            f.write(str(kwargs))
    
    @staticmethod
    def load_pbcn_info(name = 'pbcn_model'):
        with open(name+'.txt', mode='r', encoding="utf-8") as f:
            l = f.readline()
        return ast.literal_eval(l)
    
    
    def controller_to_func_minimum(self, controller):
        """QM法"""
        func_lists = [[] for _ in range(self.M)]
        for x_idx in range(2**self.N):
            u_idxs = np.where(self.u_space[controller[x_idx]])[0]   # controller_funcに項を追加するuのindex
            if len(u_idxs) != 0:
                # xを積で表す項をつくり、controller_funcに追加
                func = [(i+1)*[-1,1][int(val)] for i,val in enumerate(self.x_space[x_idx])]
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
    
    
    @staticmethod
    def is_same_func(func1, func2):
        content = ' '.join([func1,func2])
        # 最も大きい数字を検知する
        N = max(map(int,set(re.findall(r'x\[(\d+)\]', content))), default=-1) + 1
        M = max(map(int,set(re.findall(r'u\[(\d+)\]', content))), default=-1) + 1
        x_space = np.fliplr(np.array(list(itertools.product([0,1], repeat=N)), dtype=np.bool_))
        u_space = np.fliplr(np.array(list(itertools.product([0,1], repeat=M)), dtype=np.bool_))
        
        for x,u in itertools.product(x_space,u_space):
            assert eval(func1) == eval(func2)
        return True


    
class gym_PBCN(PBCN):
    def __init__(self, pbcn_model, target_x):
        super().__init__(pbcn_model)
        self.target_x = target_x
        #self.target_state = int(''.join(str(int(val)) for val in reversed(target_x)),2)
        self.target_state = self.convert(target_x)
        self.count = 0      # 遷移回数
    
    def reset(self):
        self.count = 0
        self.state = np.random.randint(2**self.N)
        return self.state
    
    def step(self, action):
        self.count += 1
        x = self.x_space[self.state]
        u = self.u_space[action]
        next_x = np.array([
            self.calc(transition_rule, x, u) for transition_rule in self.pbcn_model
            ])
        next_state = int(''.join(str(int(val)) for val in reversed(next_x)),2)
        
        if next_state == self.target_state:
            # 目標状態に到達したとき
            reward = 1
            done = 1
        elif next_state == self.state:
            # 前と状態が変わらないとき
            reward = -1
            done = 0
        else:
            reward = 0
            done = 0
        
        self.state = next_state
        return next_state, reward, done
    
        
    def step_with_controller(self, controller):
        x = self.x_space[self.state]
        u = self.u_space[controller[self.state]]
        next_x = np.array([
            self.calc(transition_rule, x, u) for transition_rule in self.pbcn_model
            ])
        next_state = int(''.join(str(int(val)) for val in reversed(next_x)),2)
        
        if np.all(next_x == self.target_x):
            # 目標状態に到達したとき
            done = 1
        else:
            done = 0
            
        self.state = next_state
        return next_x, done



if __name__ == '__main__':
    # txtファイルからインポート
    info = PBCN.load_pbcn_info('pbcn_model_pinning_3')
    pbcn_model = info['pbcn_model']
    target_x = np.array(info['target_x'], dtype=np.bool_)
    controller = np.array(info['controller'], dtype=np.int32)
    #controller = np.array([1,0,1,1,0,0,0,0], dtype=np.int32)
    #controller = np.array([0,0,0,0,0,0,0,0], dtype=np.int32)
    #controller = np.array([1,1,0,0,0,0,0,0], dtype=np.int32)
    
    env = gym_PBCN(pbcn_model, target_x)
    
    # コントローラが用いる変数を取得
    ver = env.get_ver_from_controller(controller)
    
    # コントローラを式にしてpbcnに埋め込む
    controller_funcs = env.controller_to_func(controller)
    controller_funcs_minimum = env.controller_to_func_minimum(controller)
    [env.is_same_func(controller_func,controller_func_minimum) for controller_func,controller_func_minimum in zip(controller_funcs,controller_funcs_minimum)]
    pbn_model = env.embed_controller(controller)
    transition_list = env.pbn_model_to_transition_list(pbn_model)
    drawset.transition_diagram(transition_list)
    
    
    
    # コントローラで状態遷移を行う
    num_try = 100; time = 150
    data = np.empty((num_try,time,env.N), dtype=np.bool_)
    for n in range(num_try):
        env.reset()
        #env.state = 2
        for t in range(time):
            next_state, done = env.step_with_controller(controller)
            data[n,t] = next_state
    # 各試行の平均をとる
    data_mean = np.mean(data, axis=0)
    
    for i in range(len(data_mean[0])):
        plt.plot(data_mean[:,i], label=f'x{i}')
    plt.legend()
    plt.show()
    
    env.save_pbcn_info(pbcn_model=pbcn_model, target_x=target_x, controller=controller)



    a = env.convert(np.array([True,False,False]))
    b = np.flip(~np.array([True,False,False]))
    c = [(2**i)*v for i,v in enumerate(b)]











