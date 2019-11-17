from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from cart_cls import CartCls
x=load_digits().data
y=load_digits().target
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=10)
model=CartCls()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(y_pred)
print(y_test)
report=classification_report(y_test,y_pred)
print(report)
# print(model.tree)
