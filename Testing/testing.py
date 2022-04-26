import pickle
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, confusion_matrix

from DataExploration.analysis import data_explore, scatterPlot
from DataManipulation import getData
import seaborn as sns
import matplotlib.pyplot as plt

# New Dataset Testing
from Training.training import getFeaturesOutput, train, featureSelectionTest

encode_ht = LabelEncoder()
encode_at = LabelEncoder()
encode_htr = LabelEncoder()
encode_out = LabelEncoder()

ref20 = getData.df_r20
sea20 = getData.df_s20
team20 = getData.df_t20

ref18 = getData.df_r18
sea18 = getData.df_s18
team18 = getData.df_t18

ref19 = getData.df_r19
sea19 = getData.df_s19
team19 = getData.df_t19

ref17 = getData.df_r17
sea17 = getData.df_s17
team17 = getData.df_t17

ref16 = getData.df_r16
sea16 = getData.df_s16
team16 = getData.df_t16

ref15 = getData.df_r15
sea15 = getData.df_s15
team15 = getData.df_t15

ref14 = getData.df_r14
sea14 = getData.df_s14
team14 = getData.df_t14


X1, Y1 = getFeaturesOutput(ref=ref20, team=team20, season=sea20, no=20, encode_ht=encode_ht, encode_at=encode_at,
                           encode_htr=encode_htr, encode_out=encode_out)
X2, Y2 = getFeaturesOutput(ref=ref19, team=team19, season=sea19, no=19, encode_ht=encode_ht, encode_at=encode_at,
                           encode_htr=encode_htr, encode_out=encode_out)
X3, Y3 = getFeaturesOutput(ref=ref18, team=team18, season=sea18, no=18, encode_ht=encode_ht, encode_at=encode_at,
                           encode_htr=encode_htr, encode_out=encode_out)
X4, Y4 = getFeaturesOutput(ref=ref17, team=team17, season=sea17, no=17, encode_ht=encode_ht, encode_at=encode_at,
                           encode_htr=encode_htr, encode_out=encode_out)
X5, Y5 = getFeaturesOutput(ref=ref16, team=team16, season=sea16, no=16, encode_ht=encode_ht, encode_at=encode_at,
                           encode_htr=encode_htr, encode_out=encode_out)
X6, Y6 = getFeaturesOutput(ref=ref15, team=team15, season=sea15, no=15, encode_ht=encode_ht, encode_at=encode_at,
                           encode_htr=encode_htr, encode_out=encode_out)

features = [X1, X2, X3, X4, X5, X6]
preds = [Y1, Y2, Y3, Y4, Y5, Y6]
X1 = pd.DataFrame(X1)
X_train = np.concatenate((X1, X2, X3, X4, X5, X6))
Y_train = np.concatenate((Y1, Y2, Y3, Y4, Y5, Y6))
X_ablation = SelectKBest(f_classif, k=20).fit_transform(X_train, Y_train)
train(X_train, Y_train)


def exploratoryAnalysis():
    seasons = [getData.df_s13, getData.df_s14, getData.df_s15, getData.df_s16, getData.df_s17, getData.df_s18,
               getData.df_s19, getData.df_s20]
    data_explore(seasons)


# exploratoryAnalysis()
X, Y = getFeaturesOutput(ref=ref14, team=team14, season=sea14, no=14, encode_ht=encode_ht, encode_at=encode_at,
                         encode_htr=encode_htr, encode_out=encode_out)
#X_ablation_test = SelectKBest(f_classif, k=20).fit_transform(X, Y)
#X_ablation_test = pd.DataFrame(X_ablation_test)
# scatterPlot(X)

num_features = np.arange(len(X.columns))
fScore, fmScore, top10Best = featureSelectionTest(X, Y)

fig = plt.figure()
plt.xlabel('Features')
plt.ylabel('F mutual Score')
plt.legend()
plt.plot(num_features, fmScore)
fig.savefig('fmScore.png', dpi=fig.dpi)

with open('LogisticRegression.sav', 'rb') as p:
    LogisticModel = pickle.load(p)
with open('SVC.sav', 'rb') as p:
    SvmModel = pickle.load(p)
with open('DT.sav', 'rb') as p:
    DtModel = pickle.load(p)
with open('XGBoost.sav', 'rb') as p:
    XgbModel = pickle.load(p)

Y_pred_log = LogisticModel.predict(X)
accuracy_log = accuracy_score(Y, Y_pred_log)
f1_log = f1_score(Y, Y_pred_log, average='macro')
confusion_log = confusion_matrix(Y, Y_pred_log)

Y_pred_svm = SvmModel.predict(X)
accuracy_svm = accuracy_score(Y, Y_pred_svm)
f1_svm = f1_score(Y, Y_pred_svm, average='macro')
confusion_svm = confusion_matrix(Y, Y_pred_svm)

Y_pred_dt = DtModel.predict(X)
accuracy_dt = accuracy_score(Y, Y_pred_dt)
f1_dt = f1_score(Y, Y_pred_dt, average='macro')
confusion_dt = confusion_matrix(Y, Y_pred_dt)

Y_pred_xgb = XgbModel.predict(X)
accuracy_xgb = accuracy_score(Y, Y_pred_xgb)
f1_xgb = f1_score(Y, Y_pred_xgb, average='macro')
confusion_xgb = confusion_matrix(Y, Y_pred_xgb)

file2write = open('Results.txt', 'w')
file2write.write(str(accuracy_log))
file2write.write(str(accuracy_svm))
file2write.write(str(accuracy_dt))
file2write.write(str(accuracy_xgb))
file2write.write(str(f1_log))
file2write.write(str(f1_svm))
file2write.write(str(f1_dt))
file2write.write(str(f1_xgb))
file2write.write(str(confusion_log))
file2write.write(str(confusion_svm))
file2write.write(str(confusion_dt))
file2write.write(str(confusion_xgb))

fig1 = plt.figure()
ax = sns.heatmap(confusion_log / np.sum(confusion_log), annot=True, cmap='Blues')
ax.set_title('Confusion Matrix with labels')
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values ')
ax.xaxis.set_ticklabels(['HW', 'AW', 'D'])
ax.yaxis.set_ticklabels(['HW', 'AW', 'D'])
fig1.savefig('Log_Confusion.png', dpi=fig1.dpi)

fig2 = plt.figure()
ax = sns.heatmap(confusion_xgb / np.sum(confusion_svm), annot=True, cmap='Blues')
ax.set_title('Confusion Matrix with labels')
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values')
ax.xaxis.set_ticklabels(['HW', 'AW', 'D'])
ax.yaxis.set_ticklabels(['HW', 'AW', 'D'])
fig2.savefig('SVM_Confusion.png', dpi=fig2.dpi)

fig3 = plt.figure()
ax = sns.heatmap(confusion_log / np.sum(confusion_dt), annot=True, cmap='Blues')
ax.set_title('Confusion Matrix with labels')
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values ')
ax.xaxis.set_ticklabels(['HW', 'AW', 'D'])
ax.yaxis.set_ticklabels(['HW', 'AW', 'D'])
fig3.savefig('DT_Confusion.png', dpi=fig3.dpi)

fig4 = plt.figure()
ax = sns.heatmap(confusion_log / np.sum(confusion_xgb), annot=True, cmap='Blues')
ax.set_title('Confusion Matrix with labels')
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values ')
ax.xaxis.set_ticklabels(['HW', 'AW', 'D'])
ax.yaxis.set_ticklabels(['HW', 'AW', 'D'])
fig4.savefig('XgB_Confusion.png', dpi=fig4.dpi)
