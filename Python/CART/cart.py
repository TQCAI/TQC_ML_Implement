from collections import Counter,defaultdict
import numpy as np


class Node:
    def __init__(self,feat=-1,val=None,res=None,left=None,right=None):
        self.feat=feat
        self.val=val
        self.res=res
        self.left=left
        self.right=right

    def __repr__(self):
        if self.res is not None:
            return str(self.res)
        return '['+repr(self.left)+repr(self.right)+']'


class Cart:
    def __init__(self,epsilon,min_sample):
        self.epsilon=epsilon
        self.min_sample=min_sample
        self.tree=None

    @property
    def minSubSet(self):
        raise NotImplementedError

    def getVal(self,y_data:np.ndarray):
        raise NotImplementedError

    def getFeatVal(self,*setn):
        raise NotImplementedError

    def getMinVal(self,pre_val):
        return NotImplementedError

    def isValidSplit(self,pre_val,min_val):
        raise NotImplementedError

    def bestSplit(self,x_data,y_data,splits_set):
        '''
        回归树： 最小化二乘误差
        分类树：最大化信息增益
         返回所有切分点的基尼指数，以字典形式存储。
         键为split，是一个元组，第一个元素为最优切分特征，
         第二个为该特征对应的最优切分值
        '''
        pre_val=self.getVal(y_data)
        subdata_inds=defaultdict(list)   # 切分点以及相应的样本点的索引
        for split in splits_set:
            for ind,sample in enumerate(x_data):
                if sample[split[0]]<=split[1]:
                    subdata_inds[split].append(ind)
        min_val=self.getMinVal(pre_val)
        best_split=None
        best_set=None
        length=y_data.shape[0]
        for split,data_ind in subdata_inds.items():
            set1=y_data[data_ind]
            set2_inds=list(
                set(range(length))-set(data_ind)
            )
            set2=y_data[set2_inds]
            if set1.shape[0]<self.minSubSet or set2.shape[0]<self.minSubSet:
                continue
            now_val=self.getFeatVal(set1,set2)
            if now_val<min_val:
                min_val=now_val
                best_split=split
                best_set=(data_ind,set2_inds)
        if not self.isValidSplit(pre_val,min_val):
            best_split=None
        return best_split,best_set,min_val

    def leaf(self,y_data:np.ndarray):
        raise NotImplementedError()

    def buildTree(self,x_data,y_data,splits_set):
        if y_data.shape[0]<self.min_sample:
            return Node(res=self.leaf(y_data))  # Counter(y_data).most_common(1)[0][0]
        best_split, best_set, min_gini=self.bestSplit(x_data,y_data,splits_set)
        if best_split is None:
            return Node(res=self.leaf(y_data))  # Counter(y_data).most_common(1)[0][0]
        splits_set.remove(best_split)
        nodes=[0]*2
        for i in range(2):
            nodes[i]=self.buildTree(x_data[best_set[i]],y_data[best_set[i]],splits_set)
        return Node(feat=best_split[0],val=best_split[1],left=nodes[0],right=nodes[1])

    def fit(self,x_data,y_data):
        splits_set=[]
        for feat in range(x_data.shape[1]):
            unique_vals=np.unique(x_data[:,feat])
            if unique_vals.shape[0]<2:
                continue
            elif unique_vals.shape[0]==2:
                splits_set.append((feat,unique_vals[0]))
            else:
                for val in unique_vals:
                    splits_set.append((feat,val))
        self.tree=self.buildTree(x_data,y_data,splits_set)

    def travel(self, feat, tree):
        if tree.res is not None:
            return tree.res
        else:
            if feat[tree.feat]<=tree.val:
                branch=tree.left
            else:
                branch=tree.right
            return self.travel(feat, branch)

    def predict(self,x):
        return np.array([
            self.travel(feat,self.tree) for feat in x
        ])








