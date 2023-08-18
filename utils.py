# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 15:14:39 2022

@author: f.motoyama
"""
import itertools
from copy import copy
from collections import defaultdict
from typing import NamedTuple, Union
import numpy as np




def get_unique_list(seq):
    seen = []
    return [x for x in seq if x not in seen and not seen.append(x)]

    
def QM(f):
    """
    fを最簡形にする
    f: 積和標準形を表す2次元list 変数名には0を用いてはいけない
    項の表現にset()を用いることで、行列で作業するより小さくなりやすい
    """
    if f in [0,1]:
        return f
    
    # fで用いられている変数を抽出
    V = set([abs(v) for v in itertools.chain.from_iterable(f)])
    l = len(V)          #変数の個数
    
    # fの各項をset型に変換
    ## 項内の同じ変数を削除
    f = [set(term) for term in f]
    ## 符号違いの変数をもつ項を削除(その項は0を意味する)
    f = [term for term in f if len(set(map(abs,term))) == len(term)]
    if f == []:
        return 0
    ## 重複する項を削除
    f = get_unique_list(f)
    
    
    
    #fを主加法標準展開
    terms_minimum = []
    for term in f:
        v_add = V - {abs(v) for v in term}
        # v_addの変数の正負を変えるすべてのパターンを生成し、それぞれにtermを連結する
        term_list = [
            term | set(term_add)
            for term_add in itertools.product(*list((v,-v) for v in v_add))
            ]
        terms_minimum.extend(term_list)
    terms_minimum = get_unique_list(terms_minimum)
    
    
    #圧縮
    terms_principal = []      #主項
    table = terms_minimum
    for _ in range(1,l+1):
        #圧縮回数は最大l-1回だが、l=1の場合を処理するためl回の繰り返し
        table_next = []
        compressed = np.zeros(len(table), dtype = 'bool')   #圧縮できたらTrue
        for i,j in itertools.combinations(range(len(table)),2):
            # 符号の違う1つの変数以外が共通するとき、圧縮する
            symmetric_difference = list(table[i] ^ table[j])
            if len(symmetric_difference) == 2:
                if abs(symmetric_difference[0]) == abs(symmetric_difference[1]):
                    # tableに{v},{-v}が存在するとき、式は1で決定する
                    if len(table[i]) == 1:
                        return 1
                    compressed[[i,j]] = True
                    table_next.append(table[i] & table[j])
        
        #圧縮されなかった項を主項に追加
        for i in np.where(~compressed)[0]:
            terms_principal.append(table[i])
        
        if table_next == []:
            break
        table_next = get_unique_list(table_next)
        table = copy(table_next)
    
    
    #主項の変数と値が一致する最小項を探し、主項図に印をつける
    table_principal = np.zeros((len(terms_principal), len(terms_minimum)), dtype = 'bool')    #主項図
    for row,term_p in enumerate(terms_principal):
        for col,term_m in enumerate(terms_minimum):
            if term_m >= term_p:    #term_mがterm_pの部分集合のとき
                table_principal[row,col] = True
    
    
    #必須項を求める
    idx_m = np.where(np.count_nonzero(table_principal, axis=0) == 1)[0]     #単独の主項がカバーする最小項のidx
    idx_p = list(set(np.where(table_principal[:,idx_m])[0]))                #idx_mの最小項をカバーする主項のidx
    terms_essential = [terms_principal[i] for i in idx_p]
    #使われていない主項とカバーされていない主項図の領域を求める
    terms_principal2 = [terms_principal[i] for i in range(len(terms_principal)) if i not in idx_p]
    idx_m_noncover = np.where(np.sum(table_principal[idx_p], axis=0) != 1)[0]
    table_principal2 = table_principal[:,idx_m_noncover]        #カバーされていない最小項の列を抽出
    table_principal2 = np.delete(table_principal2, idx_p, 0)    #必須項の行を削除
    
    if len(table_principal2):
        #ぺトリック法 table_principal2の最小項をカバーするterms_principal2の主項を求める
        f_petric_ORAND = [np.where(column)[0].tolist() for column in table_principal2.T]  #ぺトリック方程式（和積系）
        #和積系を積和系に変換
        f_petric_ANDOR = [set()]
        for term_OR in f_petric_ORAND:
            f_petric_ANDOR2 = []
            for v in term_OR:
                f_petric_ANDOR2 += [term_AND | {v} for term_AND in f_petric_ANDOR]
            f_petric_ANDOR = get_unique_list(f_petric_ANDOR2)
        #積和系で最短の項を求める
        idx_principal2 = list()
        temp = len(terms_principal2)
        for term_AND in f_petric_ANDOR:
            if len(term_AND) < temp:
                idx_principal2 = term_AND
                temp = len(term_AND)
        
        terms_essential += [terms_principal2[i] for i in idx_principal2]
    
    terms_essential = [list(term) for term in terms_essential]
    return terms_essential




class Node(NamedTuple):
    v: int
    n0: 'Node'
    n1: 'Node'
    hash: int
class Leaf(NamedTuple):
    value: int
    hash: int

class BDD:
    """
    n,node  : Nodeオブジェクト 同じ意味のオブジェクトは1つのみ
    h,hash  : 各ノードに割り当てられたハッシュ
    """
    def __init__(self):
        self.table = dict()
        self.table_AP = defaultdict(dict)   # table_AP[op][(nA,nB)] = 演算結果  ※(nA,nB)の順序を気を付ける
        self.leaf0 = Leaf(0,hash((0,)))
        self.leaf1 = Leaf(1,hash((1,)))
        self.table[self.leaf0.hash] = self.leaf0
        self.table[self.leaf1.hash] = self.leaf1
    
    
    def GetNode(self, v, n0, n1):
        # 削除規則
        if n0 == n1:
            return n0
        # 共有規則
        h = hash((v,n0.hash,n1.hash))
        n = Node(v, n0, n1, h)
        n_found = self.table.get(h)
        if n_found:
            return n_found
        else:
            self.table[h] = n
            return n
    
    
    def GetUnitNode(self, v):
        return self.GetNode(v, self.leaf0, self.leaf1)
    
    
    def OR(self,nA,nB):
        # 掘り進む必要がないケース
        if nA == self.leaf1 or nB == self.leaf1:
            return self.leaf1
        if nA == nB or nB == self.leaf0:
            return nA
        if nA == self.leaf0:
            return nB
        
        # 過去の計算結果を用いる
        key = hash(frozenset({nA.hash,nB.hash}))  # 順序を気にしない
        result_found = self.table_AP['OR'].get(key)
        if result_found:
            return result_found
        
        # 再帰的にシャノン展開
        vA,nA0,nA1 = nA[:3]
        vB,nB0,nB1 = nB[:3]
        if vA<vB:
            r=self.GetNode(vA,self.OR(nA0,nB),self.OR(nA1,nB))
        elif vA>vB:
            r=self.GetNode(vB,self.OR(nA,nB0),self.OR(nA,nB1))
        elif vA==vB:
            r=self.GetNode(vA,self.OR(nA0,nB0),self.OR(nA1,nB1))
        
        self.table_AP['OR'][key] = r
        return r
        
        
    def AND(self,nA,nB):
        # 掘り進む必要がないケース
        if nA == self.leaf0 or nB == self.leaf0:
            return self.leaf0
        if nA == nB or nB == self.leaf1:
            return nA
        if nA == self.leaf1:
            return nB
        
        # 過去の計算結果を用いる
        key = hash(frozenset({nA.hash,nB.hash}))  # 順序を気にしない
        result_found = self.table_AP['AND'].get(key)
        if result_found:
            return result_found
        
        # 再帰的にシャノン展開
        vA,nA0,nA1 = nA[:3]
        vB,nB0,nB1 = nB[:3]
        if vA<vB:
            r=self.GetNode(vA,self.AND(nA0,nB),self.AND(nA1,nB))
        elif vA>vB:
            r=self.GetNode(vB,self.AND(nA,nB0),self.AND(nA,nB1))
        elif vA==vB:
            r=self.GetNode(vA,self.AND(nA0,nB0),self.AND(nA1,nB1))
        
        self.table_AP['AND'][key] = r
        return r
        
    def NOT(self,n):
        if isinstance(n,Leaf):
            if n == self.leaf0:
                return self.leaf1
            else:
                return self.leaf0
        # 過去の計算結果を用いる
        result_found = self.table_AP['NOT'].get(n.hash)
        if result_found:
            return result_found
        # 再帰的に計算
        r = self.GetNode(n.v,self.NOT(n.n0),self.NOT(n.n1))
        
        self.table_AP['NOT'][n.hash] = r
        return r
        
    
            
    ####################
    """
    #未点検
    @staticmethod
    def GetBDD(root):
        # not枝のないbddをlist型で返す !!!ノードの共有が表現されない
        def scan(n,sign):
            # ノードnのbddを返す
            if isinstance(n,Leaf):
                return int(sign)
            v,n0,n1 = n[:3]
            return [v, scan(n0, not sign if n.neg else sign), scan(n1, sign)]
        return scan(root.node, not root.neg)
    """
    
    #tableの仕様変更後、未点検
    @staticmethod
    def GetBDDdict(node):
        # bddをdict型で返す
        if isinstance(node,Leaf):
            return node.value
        bdd_dict = dict()   # hash: [v,hash0,hash1]
        def scan(node):
            if isinstance(node,Leaf):
                return
            n0_hash = node.n0.value if isinstance(node.n0,Leaf) else node.n0.hash
            n1_hash = node.n1.value if isinstance(node.n1,Leaf) else node.n1.hash
            bdd_dict[node.hash] = [node.v, n0_hash, n1_hash]
            scan(node.n0)
            scan(node.n1)
        scan(node)
        return bdd_dict
    
    
    
    def AssignConst(self,node,cdict):  #cdict[v]=True/False
        """各変数に定数を代入した結果のbddを返す"""
        if isinstance(node,Leaf):
            return node
        
        const = cdict.get(node.v)
        if const is None:
            r = self.GetNode(
                node.v,
                self.AssignConst(node.n0,cdict),
                self.AssignConst(node.n1,cdict)
                )
        else:
            assert const==0 or const==1
            r = self.AssignConst(node.n1,cdict) if const else self.AssignConst(node.n0,cdict)
        return r
    
    
    @staticmethod
    def AssignConstAll(node,cdict):
        """BDDのすべての変数の値が与えられているとき、BDDの出力を求める"""
        if isinstance(node,Leaf):
            return node.value
        while isinstance(node,Node):
            node = node.n1 if cdict[node.v] else node.n0
        return node.value


    @staticmethod
    def GetV(node):
        """bddで使われている変数を列挙する"""
        V = []
        def scan(node):
            if isinstance(node,Leaf):
                return
            V.append(node.v)
            scan(node.n0)
            scan(node.n1)
        scan(node)
        return set(V)


    def EnumPath(self, node, shorten=False, check=False):
        """1へ到達するパス（1を出力する状態）を全て求める"""
        if isinstance(node,Leaf):
            return
        V_ordered = sorted(self.GetV(node))
        l = len(V_ordered)
        
        def scan(node):
            # nodeから1へ到達するための入力の組み合わせを求める
            id_v = V_ordered.index(node.v)
            states = []
            for branch,node_c in enumerate([node.n0,node.n1]):
                if node_c == self.leaf1:
                    states_sub = np.full((1, l), -1, dtype = 'i1')
                    states_sub[:,id_v] = branch
                    states.append(states_sub)
                elif isinstance(node_c,Node):
                    states_sub = scan(node_c)
                    states_sub[:,id_v] = branch
                    states.append(states_sub)
            return np.concatenate(states)
        
        states = scan(node)
        
        if not shorten or check:
            # -1の部分を書き下す
            states2 = []
            for state in states:
                cols = np.where(state==-1)[0]
                states2_sub = np.tile(state, (2**len(cols),1))   # stateを縦に並べる
                states2_sub[:,cols] = list(itertools.product([1,0], repeat=len(cols)))
                states2.append(states2_sub)
            states2 = np.concatenate(states2)
                
            if check:
                # 求めたパスで1へたどり着くか検査
                for state in states2:
                    cdict = {v:value for v,value in zip(V_ordered,state)}
                    assert self.AssignConstAll(node,cdict)
            if not shorten:
                states = states2
        
        return states,V_ordered
    
    
    def CountPath(self, node, V_ordered=None):
        """
        1へ到達するパスの本数を数える
        V_orderedのうちbddに登場していない変数のパターンも数える
        V_ordered=Noneのときは単純にbddのパス数を数える
        """
        if V_ordered == None:
            def scan(node):
                if isinstance(node,Leaf):
                    return node.value
                return scan(node.n0) + scan(node.n1)
            return scan(node)
            
        else:
            l = len(V_ordered)
            def scan(node):
                if isinstance(node,Leaf):
                    return node.value, l
                # 子ノードとの間で消えているノード数に応じてcountを増やす
                id_v = V_ordered.index(node.v)
                count0, id_0 = scan(node.n0)
                count0 *= 2 ** (id_0 - id_v - 1)
                count1, id_1 = scan(node.n1)
                count1 *= 2 ** (id_1 - id_v - 1)
                return count0+count1, id_v
            
            count, id_ = scan(node)
            count *= 2 ** (id_)
            return count
    
    def PickPath(self,node):
        """ランダムにパスを1本示す"""
        if isinstance(node,Leaf):
            return
        path = dict()
        while node != self.leaf1:
            edge = np.random.randint(0,2)
            if node[1+edge] == self.leaf0:
                path[node.v] = 1-edge
                node = node[1+1-edge]
            else:
                path[node.v] = edge
                node = node[1+edge]
        return path



if __name__ == '__main__':
    
    #f = [[-1,-2,3,4],[2,3,4],[1,2,-3],[1,-2,3,4]]
    #f = [[2],[7,1,3],[1,3,7],[2]]
    #f_ = QM(f)
    
    f0 = [[1,-1]]
    f0_ = QM(f0)
    f1 = [[1,2],[-3],[2,1]]
    f1_ = QM(f1)
    #f1 = [[-1,-2],[-1,2],[1,-2],[1,2]]
    #f1_ = QM(f1)
    