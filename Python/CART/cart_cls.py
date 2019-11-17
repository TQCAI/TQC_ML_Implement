from cart import Cart, Node
import numpy as np
from collections import Counter

class CartCls(Cart):

    def __init__(self,epsilon=1e-3,min_sample=1):
        super(CartCls, self).__init__(epsilon,min_sample)

    def getGini(self,y_data:np.ndarray):
        counter=Counter(y_data)
        length=y_data.shape[0]
        return 1-sum([(v/length)**2 for v in counter.values() ])

    def getFeatGini(self,*setn):
        num=sum([seti.shape[0] for seti in setn])
        return sum([
            (seti.shape[0]/num)*self.getGini(seti) for seti in setn
        ])

    @property
    def minSubSet(self):
        return 1

    def getVal(self,y_data:np.ndarray):
        return self.getGini(y_data)

    def getFeatVal(self,*setn):
        return self.getFeatGini(*setn)

    def leaf(self,y_data:np.ndarray):
        return Counter(y_data).most_common(1)[0][0]

    def isValidSplit(self,pre_val,min_val):
        return abs(pre_val-min_val)>self.epsilon

    def getMinVal(self,pre_val):
        return 1
