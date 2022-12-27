# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 20:20:34 2022

@author: motoyama
"""
import re, itertools, copy
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.simplefilter('error')

#import os, sys
#sys.path.append(os.path.join(os.path.dirname(__file__), '...'))
#from QuineMcCluskey import QM
#import drawset

"""
pbcn_model = [
    [['x[5] and x[12]'],[1]],
    [['x[24]'],[0.7,0.3]],
    [['x[1]'],[0.8,0.2]],
    [['x[27]'],[1]],
    [['x[20]'],[1]],
    [['x[4]'],[1]],
    [['x[14] and u[1] or x[25] and u[1]'],[1]],
    [['x[13]'],[1]],
    [['x[17]'],[1]],
    [['x[24] and x[27]'],[1]],
    [['not x[8]'],[1]],
    [['x[23]'],[1]],
    [['x[11]'],[1]],
    [['x[27]'],[1]],
    [['not x[19] and u[0] and u[1]'],[1]],
    [['x[2]'],[1]],
    [['not x[10]'],[1]],
    [['x[1]'],[1]],
    [['x[9] and x[10] and x[24] and x[27] or x[10] and x[22] and x[24] and x[27]'],[1]],
    [['x[6] and not x[25]'],[1]]
    [['x[10] and x[21]'],[1]]
    [['x[1] and x[17]'],[1]]
    [['x[14]'],[1]]
    [['x[17]'],[1]]
    [['x[7]'],[1]]
    [['not x[3] and u[2]','x[25]'],[0.5,0.5]]
    [['x[6] or x[14] and x[25]'],[1]]
    [['not x[3] and x[14] and x[23]'],[1]]
    ]
target_x = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,0, 1, 1, 0], dtype=np.bool_)
"""
"""
pbcn_model = [
    [['not x[1] and u[0]','u[0]'],[0.6,0.4]],
    [['not x[0] and x[2]','x[1]'],[0.7,0.3]],
    [['x[1] or u[0]','x[2]'],[0.8,0.2]],
    ]
target_x = np.array([0,1,1], dtype=np.bool_)
"""
pbcn_model = [
    [['x[1] and x[15]'],[1]],
    [['not (x[4] or x[2] or x[15])'],[1]],
    [['(x[1] or x[2]) and not x[15]'],[1]],
    [['x[14] and not x[15]'],[1]],
    [['x[3] and not x[15]'],[1]],
    [['not (x[6] or x[15])'],[1]],
    [['x[14] and not x[15] and u[0]'],[1]],
    [['x[5] and not (x[14] or x[15]) or u[1]'],[1]],
    [['(x[7] or x[5] and not x[10]) and not x[15]'],[1]],
    [['(x[11] and not x[12] or x[8]) and not x[15]'],[1]],
    [['not (x[8] or x[15])'],[1]],
    [['not (x[13] or x[15])'],[1]],
    [['not (x[11] or x[15])'],[1]],
    [['not (x[8] or x[15]) and u[2]','x[13]'],[0.5,0.5]],
    [['not (x[7] or x[15])'],[1]],
    [['x[9] or x[15]'],[1]],
    ]
target_x = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.bool_)


class PBCN:
    def __init__(self):
        """
        変数名、制御入力名は0からカウントアップ
        pbcn_model = [p_funcs1, p_funcs2, ...]
        p_funcs = [funcs, probs]
        funcs = [func1, func2, ...]
        probs = [prob1, prob2, ...] sum is 1
        x: (N,)             x = self.x_space[state]
        u: (M,)
        state: 0~2**N-1     state = int(''.join(str(int(val)) for val in reversed(x)),2)
        action: 0~2**M-1
        """
        self.pbcn_model = pbcn_model
        self.N, self.M = self.get_NM(self.pbcn_model)
        self.target_x = target_x
        
        self.x_space = np.fliplr(np.array(list(itertools.product([0,1], repeat=self.N)), dtype=np.bool_))
        self.u_space = np.fliplr(np.array(list(itertools.product([0,1], repeat=self.M)), dtype=np.bool_))
        self.target_state = int(''.join(str(int(val)) for val in reversed(self.target_x)),2)
        
        self.count = 0
    
    
    def reset(self):
        self.count = 0
        self.state = np.random.randint(2**self.N)
        return self.state
    
    
    def step(self, action=None):
        self.count += 1
        x = self.x_space[self.state]
        u = self.u_space[action] if action is not None else None
        next_x = np.array([
            self.calc(p_funcs, x, u) for p_funcs in self.pbcn_model
            ])
        next_state = int(''.join(str(int(val)) for val in reversed(next_x)),2)
        
        
        if next_state == self.target_state:
            # 目標状態に到達したとき
            reward = 1
        elif np.all(next_state == self.state):
            # 前と状態が変わらないとき
            reward = -1
            #reward = 0
        else:
            reward = 0
        done = 1 if self.count == 1000 else 0
        
        self.state = next_state
        return next_state, reward, done
    
    
    def step_with_controller(self, controller):
        x = self.x_space[self.state]
        u = self.u_space[controller[self.state]]
        next_x = np.array([
            self.calc(p_funcs, x, u) for p_funcs in self.pbcn_model
            ])
        next_state = int(''.join(str(int(val)) for val in reversed(next_x)),2)
        
        if np.all(next_x == self.target_x):
            # 目標状態に到達したとき
            done = 1
        else:
            done = 0
            
        self.state = next_state
        return next_x, done
    
    
    @staticmethod
    def calc(p_funcs: list, x: np.bool_, u: np.bool_=None):
        func = np.random.choice(a=p_funcs[0], size=1, p=p_funcs[1])[0]
        return eval(func)
    
    
    def get_controller_func(self, controller: np.int32):
        """express the controller as functions"""
        controller_func = [[] for _ in range(self.M)]
        for state,action in enumerate(controller):
            func = ' and '.join(f'x[{i}]' if val==True else f'not x[{i}]' for i,val in enumerate(self.x_space[state]))
            for i in np.where(self.u_space[action])[0]:
                controller_func[i].append(func)
        
        controller_func = [' or '.join(terms) if terms != [] else False for terms in controller_func]
        return controller_func
    
    def embed_controller(self, controller):
        """embed controller in pbcn_model"""
        controller_func = self.get_controller_func(controller)
        pbn_model = copy.deepcopy(self.pbcn_model)
        for p_funcs in pbn_model:
            for func_idx in range(len(p_funcs[0])):
                for i in range(self.M):
                    p_funcs[0][func_idx] = p_funcs[0][func_idx].replace(f'u[{i}]',f'({controller_func[i]})')
        return pbn_model
    
    def get_transition_table(self, pbn_model):
        """
        transition_table[x] = [[x1,x2,...],[p1,p2,...]]
        type(x) is str
        """
        # 遷移パターンを列挙
        transition_patterns = list(
            zip(
                itertools.product(*[p_funcs[0] for p_funcs in pbn_model]),
                map(np.prod, itertools.product(*[p_funcs[1] for p_funcs in pbn_model]))
                )
            )
        assert sum(transition_pattern[1] for transition_pattern in transition_patterns) == 1
        
        # 遷移パターンごとの遷移を計算
        transition_table = dict()
        for x in self.x_space:
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
    def get_NM(pbcn_model):
        N = len(pbcn_model)
        content = ' '.join(' '.join(func for func in p_funcs[0]) for p_funcs in pbcn_model)
        x_list = set(re.findall(r'x\[(\d+)\]', content))
        u_list = set(re.findall(r'u\[(\d+)\]', content))
        x_list = np.sort([int(s) for s in set(x_list)])
        u_list = np.sort([int(s) for s in set(u_list)])
        M = len(u_list)
        #assert np.all(np.arange(N) == x_list)
        #assert np.all(np.arange(M) == u_list)
        return N,M
    
    
            
                
        



if __name__ == '__main__':
    
    pbcn_model = [
        [['not x[1] and u[0]','u[0]'],[0.6,0.4]],
        [['not x[0] and x[2]','x[1]'],[0.7,0.3]],
        [['x[1] or u[0]','x[2]'],[0.8,0.2]],
        ]
    
    env = PBCN()
    
    
    Q_table = QL(env, config)
    controller = np.argmax(Q_table, axis=1)
    #controller = np.array([1,0,1,1,0,0,0,0], dtype=np.int32)
    #controller = np.array([0,0,0,0,0,0,0,0], dtype=np.int32)
    #controller = np.array([1,1,0,0,0,0,0,0], dtype=np.int32)
    
    pbn_model = env.embed_controller(controller)
    transition_table = env.get_transition_table(pbn_model)
    drawset.transition_diagram(transition_table)
    
    
    # コントローラで状態遷移を行う
    num_try = 100
    time = 150
    data = np.empty((num_try,time,config.N), dtype=np.bool_)
    for n in range(num_try):
        #env.reset()
        env.state = 2
        for t in range(time):
            next_state, done = env.step_with_controller(controller)
            data[n,t] = next_state
    # 各試行の平均をとる
    data_mean = np.mean(data, axis=0)
    
    plt.plot(data_mean[:,0], label='x0')
    plt.plot(data_mean[:,1], label='x1')
    plt.plot(data_mean[:,2], label='x2')
    plt.legend()
    plt.show()















