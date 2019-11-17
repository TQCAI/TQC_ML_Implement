from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from cart_reg import CartReg

x=load_boston().data
y=load_boston().target
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=10)
model=CartReg()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(y_pred)
print(y_test)
r=pearsonr(y_test,y_pred)[0]
r2=r2_score(y_test,y_pred)
print(f'r  = {r:.4f}')
print(f'r2 = {r2:.4f}')
