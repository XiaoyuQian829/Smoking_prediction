import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import callbacks

#read beta
LBC36_all = pd.read_table("/home/uqxqian/90days/Prediction/ML_linear/beta/LBC36.txt", sep="\s+", header=0)
LBC21_all = pd.read_table("/home/uqxqian/90days/Prediction/ML_linear/beta/LBC21.txt", sep="\s+", header=0)

#fill na with mean
LBC36_all = LBC36_all.fillna(LBC36_all.mean())
LBC21_all = LBC21_all.fillna(LBC21_all.mean())

#input smoke
smoke = pd.read_table("/home/uqxqian/90days/Prediction/ML_linear/beta/smoke.pheno", sep="\s+", names=['FID','IID','smoke'])
smoke = smoke.dropna()


LBC36_smoke = pd.merge(LBC36_all, smoke, how='inner', on=['FID','IID'])
LBC21_smoke = pd.merge(LBC21_all, smoke, how='inner', on=['FID','IID'])

n= LBC36_smoke.shape[1]
columns = LBC36_smoke.columns[2:n-1]
LBC21_individuals = LBC21_smoke.iloc[:,0]

X_train = preprocessing.scale(LBC36_smoke.iloc[:,2:n-1].to_numpy())
y_train = LBC36_smoke.iloc[:,-1].to_numpy()

X_test = preprocessing.scale(LBC21_smoke.iloc[:,2:n-1].to_numpy())
y_test = LBC21_smoke.iloc[:,-1].to_numpy()

components = X_train.shape[1]
print("Dimension of training matrix: ",X_train.shape)
print("Dimension of training matrix: ",X_test.shape)


print("")
print("")
print("Ordinary Least Square:")
linr = LinearRegression()
linr.fit(X_train,y_train)
linr_y_pred = linr.predict(X_test)
linr_fpr, linr_tpr, linr_threshold = roc_curve(y_test, linr_y_pred)
linr_roc_auc = auc(linr_fpr, linr_tpr)
print("Prediction AUC: ",linr_roc_auc)

d = {'IID' : LBC21_individuals, 'Prediction': linr_y_pred}
df = pd.DataFrame(data=d)
df.to_csv('/home/uqxqian/90days/Prediction/ML_linear/smoke_final/prediction/OLS.csv', index = False, header=True)


print("")
print("")
print("Ridge Regression:")
ridge = Ridge()
ridge_parameters = {'alpha': [0.0005, 0.001, 0.01, 0.03, 0.05, 0.1, 1, 10, 100]}
ridge_grid = GridSearchCV(ridge,
                        ridge_parameters,
                        cv = 5,
                        scoring = 'roc_auc',
                        n_jobs = -2,
                        verbose = 0)
ridge_grid.fit(X_train,y_train)
# summarize results
print("Best: %f using %s" % (ridge_grid.best_score_, ridge_grid.best_params_))
means = ridge_grid.cv_results_['mean_test_score']
stds = ridge_grid.cv_results_['std_test_score']
params = ridge_grid.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

ridge_best_model = ridge_grid.best_estimator_
ridge_y_pred = ridge_best_model.predict(X_test)
ridge_fpr, ridge_tpr, ridge_threshold = roc_curve(y_test, ridge_y_pred)
ridge_roc_auc = auc(ridge_fpr, ridge_tpr)
print("prediction AUC: ",ridge_roc_auc)

d = {'IID' : LBC21_individuals, 'Prediction': ridge_y_pred}
df = pd.DataFrame(data=d)
df.to_csv('/home/uqxqian/90days/Prediction/ML_linear/smoke_final/prediction/Ridge.csv', index = False, header=True)


print("")
print("")
print("Lasso regression:")
lasso = Lasso()
lasso_parameters = {'alpha': [0.0005, 0.001, 0.01, 0.03, 0.05, 0.1, 1, 10, 100]}
lasso_grid = GridSearchCV(lasso,
                        lasso_parameters,
                        cv = 5,
                        scoring = 'roc_auc',
                        n_jobs = -2,
                        verbose = 0)
lasso_grid.fit(X_train,y_train)
# summarize results
print("Best: %f using %s" % (lasso_grid.best_score_, lasso_grid.best_params_))
means = lasso_grid.cv_results_['mean_test_score']
stds = lasso_grid.cv_results_['std_test_score']
params = lasso_grid.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

lasso_best_model = lasso_grid.best_estimator_
lasso_y_pred = lasso_best_model.predict(X_test)
lasso_fpr, lasso_tpr, lasso_threshold = roc_curve(y_test, lasso_y_pred)
lasso_roc_auc = auc(lasso_fpr, lasso_tpr)
print("prediction AUC: ",lasso_roc_auc)

d = {'IID' : LBC21_individuals, 'Prediction': lasso_y_pred}
df = pd.DataFrame(data=d)
df.to_csv('/home/uqxqian/90days/Prediction/ML_linear/smoke_final/prediction/Lasso.csv', index = False, header=True)

lasso_coef = pd.Series(lasso_best_model.coef_, index = columns)
lasso_coef.to_csv('/home/uqxqian/90days/Prediction/ML_linear/smoke_final/lasso_coef.csv')


print("")
print("")
print("ElasticNet:")
elnet = ElasticNet()
elnet_parameters = {'alpha': [0.0005, 0.001, 0.01, 0.03, 0.05, 0.1, 1, 10, 100],
		    'l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
elnet_grid = GridSearchCV(elnet,
                        elnet_parameters,
                        cv = 5,
                        scoring = 'roc_auc',
                        n_jobs = -2,
                        verbose = 0)
elnet_grid.fit(X_train,y_train)
# summarize results
print("Best: %f using %s" % (elnet_grid.best_score_, elnet_grid.best_params_))
means = elnet_grid.cv_results_['mean_test_score']
stds = elnet_grid.cv_results_['std_test_score']
params = elnet_grid.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

elnet_best_model = elnet_grid.best_estimator_
elnet_y_pred = elnet_best_model.predict(X_test)
elnet_fpr, elnet_tpr, elnet_threshold = roc_curve(y_test, elnet_y_pred)
elnet_roc_auc = auc(elnet_fpr, elnet_tpr)
print("prediction AUC: ",elnet_roc_auc)

d = {'IID' : LBC21_individuals, 'Prediction': elnet_y_pred}
df = pd.DataFrame(data=d)
df.to_csv('/home/uqxqian/90days/Prediction/ML_linear/smoke_final/prediction/ElasticNet.csv', index = False, header=True)

elnet_coef = pd.Series(elnet_best_model.coef_, index = columns)
elnet_coef.to_csv('/home/uqxqian/90days/Prediction/ML_linear/smoke_final/elnet_coef.csv')


print("")
print("")
print("Support Vector Machine:")
svc = SVC()
svc_parameters = {'C': [1, 10, 100],
              'gamma': [0.1, 0.01, 0.001],
              'kernel': ['rbf', 'poly', 'sigmoid']}
svc_grid = GridSearchCV(svc,
                        svc_parameters,
                        cv = 5,
                        scoring = 'roc_auc',
                        n_jobs = -2,
                        verbose = 0)
svc_grid.fit(X_train,y_train)
# summarize results
print("Best: %f using %s" % (svc_grid.best_score_, svc_grid.best_params_))
means = svc_grid.cv_results_['mean_test_score']
stds = svc_grid.cv_results_['std_test_score']
params = svc_grid.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

svc_best_model = svc_grid.best_estimator_
svc_y_pred = svc_best_model.predict(X_test)
svc_fpr, svc_tpr, svc_threshold = roc_curve(y_test, svc_y_pred)
svc_roc_auc = auc(svc_fpr, svc_tpr)
print("prediction AUC: ",svc_roc_auc)

d = {'IID' : LBC21_individuals, 'Prediction': svc_y_pred}
df = pd.DataFrame(data=d)
df.to_csv('/home/uqxqian/90days/Prediction/ML_linear/smoke_final/prediction/SVM.csv', index = False, header=True)


print("")
print("")
print("Random Forest:")
rf = RandomForestClassifier()
rf_parameters = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [5, 10, 20],
    'max_features': ['auto', 'sqrt', 'log2']
}
rf_grid = GridSearchCV(rf,
                       rf_parameters,
                       cv = 5,
                       scoring = 'roc_auc',
                       verbose = 0,
                       n_jobs = -2)
rf_grid.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (rf_grid.best_score_, rf_grid.best_params_))
means = rf_grid.cv_results_['mean_test_score']
stds = rf_grid.cv_results_['std_test_score']
params = rf_grid.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

rf_best_model = rf_grid.best_estimator_
rf_y_pred = rf_best_model.predict(X_test)
rf_fpr, rf_tpr, rf_threshold = roc_curve(y_test, rf_y_pred)
rf_roc_auc = auc(rf_fpr, rf_tpr)
print("prediction AUC: ",rf_roc_auc)

d = {'IID' : LBC21_individuals, 'Prediction': rf_y_pred}
df = pd.DataFrame(data=d)
df.to_csv('/home/uqxqian/90days/Prediction/ML_linear/smoke_final/prediction/RF.csv', index = False, header=True)


print("")
print("")
print("Gradient Boosting Classifier:")
gbm = GradientBoostingClassifier()
gbm_parameters = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2 ,0.3],
    'n_estimators': [100, 500, 1000],
    'max_depth': [5, 10, 20],
    }
#passing the scoring function in the GridSearchCV
gbm_grid = GridSearchCV(gbm,
                       gbm_parameters,
                       cv = 5,
                       scoring = 'roc_auc',
                       verbose = 0,
                       n_jobs = -2)
gbm_grid.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (gbm_grid.best_score_, gbm_grid.best_params_))
means = gbm_grid.cv_results_['mean_test_score']
stds = gbm_grid.cv_results_['std_test_score']
params = gbm_grid.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

gbm_best_model = gbm_grid.best_estimator_
gbm_y_pred = gbm_best_model.predict(X_test)
gbm_fpr, gbm_tpr, gbm_threshold = roc_curve(y_test, gbm_y_pred)
gbm_roc_auc = auc(gbm_fpr, gbm_tpr)
print("prediction AUC: ",gbm_roc_auc)

d = {'IID' : LBC21_individuals, 'Prediction': gbm_y_pred}
df = pd.DataFrame(data=d)
df.to_csv('/home/uqxqian/90days/Prediction/ML_linear/smoke_final/prediction/GBM.csv', index = False, header=True)


print("")
print("")
print("MLPs:")
def create_model(dense_layer_sizes=2, activation = 'relu', optimizer="adam", dropout=0.0, nbr_features=components, dense_nparams=16):
    model = Sequential()
    model.add(Dense(dense_nparams, activation=activation, input_shape=(nbr_features,)))
    model.add(Dropout(dropout), )
    for layer_size in range(dense_layer_sizes):
        model.add(Dense(dense_nparams, activation=activation))
        model.add(Dropout(dropout), )
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    return model

mlp  = KerasClassifier(build_fn=create_model, verbose = 0, batch_size = 32, epochs=100)

mlp_parameters = {
    'activation':['relu', 'tanh', 'sigmoid'],
    'dropout': [0.0, 0.1, 0.2, 0.3, 0.4],
    'dense_nparams': [16, 32, 64 ,128],
    'optimizer': ['SGD','RMSprop','Adam'],
    'dense_layer_sizes': [2, 3, 4]
}

mlp_grid = GridSearchCV(mlp,
                       mlp_parameters,
                       cv = 5,
                       scoring = 'roc_auc',
                       verbose = 0,
                       n_jobs = -2)
early_stop = callbacks.EarlyStopping(monitor='accuracy',patience=10)
mlp_grid.fit(X_train, y_train, verbose=0, callbacks = [early_stop])

# summarize results
print("Best: %f using %s" % (mlp_grid.best_score_, mlp_grid.best_params_))
means = mlp_grid.cv_results_['mean_test_score']
stds = mlp_grid.cv_results_['std_test_score']
params = mlp_grid.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

mlp_best_model = mlp_grid.best_estimator_
mlp_y_pred = mlp_best_model.predict(X_test)
mlp_fpr, mlp_tpr, mlp_threshold = roc_curve(y_test, mlp_y_pred)
mlp_roc_auc = auc(mlp_fpr, mlp_tpr)
print("prediction AUC: ",mlp_roc_auc)

d = {'IID' : LBC21_individuals, 'Prediction': mlp_y_pred.tolist()}
df = pd.DataFrame(data=d)
df.to_csv('/home/uqxqian/90days/Prediction/ML_linear/smoke_final/prediction/MLP.csv', index = False, header=True)

lw = 1
plt.figure(figsize=[10,10], dpi=300)
plt.title('ROC curve')
plt.plot(linr_fpr, linr_tpr, color = 'black', lw = lw, label = 'OLS (AUC = %0.3f)' % linr_roc_auc)
plt.plot(ridge_fpr, ridge_tpr, color = 'b', lw = lw, label = 'Ridge (AUC = %0.3f)' % ridge_roc_auc)
plt.plot(lasso_fpr, lasso_tpr, color = 'g', lw = lw, label = 'Lasso (AUC = %0.3f)' % lasso_roc_auc)
plt.plot(elnet_fpr, elnet_tpr, color = 'r', lw = lw, label = 'ElasticNet (AUC = %0.3f)' % elnet_roc_auc)
plt.plot(svc_fpr, svc_tpr, color = 'c', lw = lw, label = 'SVM (AUC = %0.3f)' % svc_roc_auc)
plt.plot(rf_fpr, rf_tpr, color = 'm', lw = lw, label = 'RF (AUC = %0.3f)' % rf_roc_auc)
plt.plot(gbm_fpr, gbm_tpr, color = 'y', lw = lw, label = 'GBM (AUC = %0.3f)' % gbm_roc_auc)
plt.plot(mlp_fpr, mlp_tpr, color = 'darkorange', lw = lw, label = 'MLP (AUC = %0.3f)' % mlp_roc_auc)
plt.plot([0, 1], [0, 1],'r--', color = 'grey')
plt.legend(loc = 'lower right')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.savefig("/home/uqxqian/90days/Prediction/ML_linear/smoke_final/plot/all_ROC.png")


'''
#PBS -N all
#PBS -q normal
#PBS -A sf
#PBS -S /bin/bash
#PBS -r n
#PBS -l select=1:ncpus=8:mem=128GB
#PBS -l walltime=99:00:00

cd /home/uqxqian/90days/Prediction/ML_linear/smoke_final/

python all.py -> all.log
'''
