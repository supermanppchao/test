import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
#读取数据集
df=pd.read_csv('./penguins_raw.csv')
# print(df.head())

#选择特征
feature_list=['Culmen Length (mm)','Culmen Depth (mm)','Flipper Length (mm)','Body Mass (g)','Delta 15 N (o/oo)','Delta 13 C (o/oo)']
label_list=['Sex']
#缺失值处理
df.dropna(axis=0,how='any',inplace=True,subset = feature_list+label_list)
#制作数据集
X=df[feature_list]
y=df[label_list]
# print(y)
y=LabelEncoder().fit_transform(y)#女性是0，男性是1
# print(y)
#划分数据集
train_x,test_x,train_y,test_y=train_test_split(X,y,test_size=0.25, random_state=42)
#数据标准化
stdScale = StandardScaler()
train_x=stdScale.fit_transform(train_x)
test_x=stdScale.transform(test_x)

#逻辑回归
log_model=LogisticRegression()
log_model.fit(train_x,train_y)
print('逻辑回归得分：',log_model.score(test_x, test_y))

#决策树
tree_model=DecisionTreeClassifier(max_depth=3)
tree_model.fit(train_x,train_y)
print('决策树得分：',tree_model.score(test_x, test_y))

#svm
svc_model = SVC()
svc_model.fit(train_x,train_y)
print('svm得分：',svc_model.score(test_x, test_y))
