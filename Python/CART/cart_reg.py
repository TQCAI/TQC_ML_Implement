from cart import Cart,Node
import numpy as np
from collections import defaultdict

class CartReg(Cart):
    def __init__(self,epsilon=0.1,min_sample=10):
        super(CartReg, self).__init__(epsilon,min_sample)

    def getErr(self,y_data:np.ndarray):
        return y_data.var()*y_data.shape[0]

    def getFeatErr(self,*setn):
        return sum([
            self.getErr(seti) for seti in setn
        ])

    @property
    def minSubSet(self):
        return 2

    def getVal(self,y_data:np.ndarray):
        return self.getErr(y_data)

    def getFeatVal(self,*setn):
        return self.getFeatErr(*setn)

    def leaf(self,y_data:np.ndarray):
        return y_data.mean()

    def isValidSplit(self,pre_val,min_val):
        return min_val>self.epsilon

    def getMinVal(self,pre_val):
        return pre_val

