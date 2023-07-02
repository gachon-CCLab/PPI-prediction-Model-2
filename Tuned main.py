import numpy as np
import pandas as pd
import os
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

working_directory = os.getcwd()
print(working_directory)
path = 'total_tavr_v1.xlsx'
df = pd.read_excel(path)

df['DeltaPR'] = df['DeltaPR'].replace("#N/A", np.nan)
df['DeltaPR'] = pd.to_numeric(df['DeltaPR'])
df['DeltaPR'] = df['DeltaPR'].fillna(df['DeltaPR'].mean())
df['DeltaQRS'] = df['DeltaQRS'].replace("#N/A", np.nan)
df['DeltaQRS'] = pd.to_numeric(df['DeltaQRS'])
df['DeltaQRS'] = df['DeltaQRS'].fillna(df['DeltaQRS'].mean())
df['FirstdegreeAVblock'] = df['FirstdegreeAVblock'].fillna(df['FirstdegreeAVblock'].mean())
df['DiastolicBP'] = df['DiastolicBP'].fillna(df['DiastolicBP'].mean())
df['SystolicBP'] = df['SystolicBP'].fillna(df['SystolicBP'].mean())
df['LVEF'] = df['LVEF'].fillna(df['LVEF'].mean())
df['PR'] = df['PR'].replace("#N/A", np.nan)
df['PR'] = pd.to_numeric(df['PR'])
df['PR'] = df['PR'].fillna(df['PR'].mean())
df['LVOT'] = df['LVOT'].fillna(df['LVOT'].mean())
df['BSA'] = df['BSA'].fillna(df['BSA'].mean())

X = df.drop("New_Onset_LBBB", axis=1)
y = df['New_Onset_LBBB']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

k = 10
selector = SelectKBest(f_classif, k=k)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test_selected)

logistic_regression = LogisticRegression(random_state=42)
logistic_regression.fit(X_train_scaled, y_train_resampled)
y_pred_lr = logistic_regression.predict(X_test_scaled)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
lr_params = {'C': [0.01, 0.1, 1, 10]}
lr_grid = GridSearchCV(LogisticRegression(random_state=42), lr_params, cv=5)
lr_grid.fit(X_train_scaled, y_train_resampled)
best_lr = lr_grid.best_estimator_
y_pred_lr_tuned = best_lr.predict(X_test_scaled)
accuracy_lr_tuned = accuracy_score(y_test, y_pred_lr_tuned)
print("Accuracy (Tuned Logistic Regression): {:.2f}%".format(accuracy_lr_tuned * 100))

knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train_scaled, y_train_resampled)
y_pred_knn = knn_classifier.predict(X_test_scaled)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
knn_params = {'n_neighbors': [3, 5, 7]}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5)
knn_grid.fit(X_train_scaled, y_train_resampled)
best_knn = knn_grid.best_estimator_
y_pred_knn_tuned = best_knn.predict(X_test_scaled)
accuracy_knn_tuned = accuracy_score(y_test, y_pred_knn_tuned)
print("Accuracy (Tuned K Nearest Neighbors): {:.2f}%".format(accuracy_knn_tuned * 100))

svc_classifier = SVC()
svc_classifier.fit(X_train_scaled, y_train_resampled)
y_pred_svc = svc_classifier.predict(X_test_scaled)
accuracy_svc = accuracy_score(y_test, y_pred_svc)
svc_params = {'C': [0.01, 0.1, 1, 10], 'kernel': ['linear', 'rbf']}
svc_grid = GridSearchCV(SVC(random_state=42), svc_params, cv=5)
svc_grid.fit(X_train_scaled, y_train_resampled)
best_svc = svc_grid.best_estimator_
y_pred_svc_tuned = best_svc.predict(X_test_scaled)
accuracy_svc_tuned = accuracy_score(y_test, y_pred_svc_tuned)
print("Accuracy (Tuned C-support Vector Classification): {:.2f}%".format(accuracy_svc_tuned * 100))

nb_classifier = GaussianNB()
nb_classifier.fit(X_train_scaled, y_train_resampled)
y_pred_nb = nb_classifier.predict(X_test_scaled)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
nb_params = {}
nb_grid = GridSearchCV(GaussianNB(), nb_params, cv=5)
nb_grid.fit(X_train_scaled, y_train_resampled)
best_nb = nb_grid.best_estimator_
y_pred_nb_tuned = best_nb.predict(X_test_scaled)
accuracy_nb_tuned = accuracy_score(y_test, y_pred_nb_tuned)
print("Accuracy (Tuned Gaussian Naive-Bayes): {:.2f}%".format(accuracy_nb_tuned * 100))

sgd_classifier = SGDClassifier()
sgd_classifier.fit(X_train_scaled, y_train_resampled)
y_pred_sgd = sgd_classifier.predict(X_test_scaled)
accuracy_sgd = accuracy_score(y_test, y_pred_sgd)
sgd_params = {'alpha': [0.001, 0.01, 0.1], 'penalty': ['l2', 'l1']}
sgd_grid = GridSearchCV(SGDClassifier(random_state=42), sgd_params, cv=5)
sgd_grid.fit(X_train_scaled, y_train_resampled)
best_sgd = sgd_grid.best_estimator_
y_pred_sgd_tuned = best_sgd.predict(X_test_scaled)
accuracy_sgd_tuned = accuracy_score(y_test, y_pred_sgd_tuned)
print("Accuracy (Tuned SGD with SVM): {:.2f}%".format(accuracy_sgd_tuned * 100))

xgb_classifier = XGBClassifier(random_state=42)
xgb_classifier.fit(X_train_scaled, y_train_resampled)
y_pred_xgb = xgb_classifier.predict(X_test_scaled)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
xgb_params = {'n_estimators': [100, 200, 500], 'max_depth': [3, 5, 10], 'learning_rate': [0.1, 0.01, 0.001]}
xgb_grid = GridSearchCV(XGBClassifier(random_state=42), xgb_params, cv=5)
xgb_grid.fit(X_train_scaled, y_train_resampled)
best_xgb = xgb_grid.best_estimator_
y_pred_xgb_tuned = best_xgb.predict(X_test_scaled)
accuracy_xgb_tuned = accuracy_score(y_test, y_pred_xgb_tuned)
print("Accuracy (Tuned XGBoost): {:.2f}%".format(accuracy_xgb_tuned * 100))

dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train_scaled, y_train_resampled)
y_pred_dt = dt_classifier.predict(X_test_scaled)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
dt_params = {'max_depth': [None, 5, 10, 15], 'min_samples_split': [2, 5, 10]}
dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=5)
dt_grid.fit(X_train_scaled, y_train_resampled)
best_dt = dt_grid.best_estimator_
y_pred_dt_tuned = best_dt.predict(X_test_scaled)
accuracy_dt_tuned = accuracy_score(y_test, y_pred_dt_tuned)
print("Accuracy (Tuned Decision Tree): {:.2f}%".format(accuracy_dt_tuned * 100))

rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_scaled, y_train_resampled)
y_pred_rf = rf_classifier.predict(X_test_scaled)
rf_params = {'n_estimators': [100, 200, 500], 'max_depth': [None, 5, 10]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5)
rf_grid.fit(X_train_scaled, y_train_resampled)
best_rf = rf_grid.best_estimator_
y_pred_rf_tuned = best_rf.predict(X_test_scaled)
accuracy_rf_tuned = accuracy_score(y_test, y_pred_rf_tuned)
print("Accuracy (Tuned Random Forest): {:.2f}%".format(accuracy_rf_tuned * 100))

classifiers = [
    ('Gaussian Naive-Bayes', best_nb),
    ('Random Forest', best_rf),
    ('Decision Tree', best_dt),
    ('XGBoost', best_xgb),
    ('SGD with SVM', best_sgd),
    ('Logistic Regression', best_lr),
    ('C-Support Vector', best_svc),
    ('K-Nearest Neighbors', best_knn)
]

voting_classifier = VotingClassifier(classifiers, voting='hard')
voting_classifier.fit(X_train_scaled, y_train_resampled)
y_pred_combined = voting_classifier.predict(X_test_scaled)
accuracy_combined = accuracy_score(y_test, y_pred_combined)
print("Accuracy (Combined): {:.2f}%".format(accuracy_combined * 100))