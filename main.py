from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
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
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))

knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train_resampled)
y_pred_knn = knn.predict(X_test_scaled)
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))

svm = SVC(random_state=42)
svm.fit(X_train_scaled, y_train_resampled)
y_pred_svm = svm.predict(X_test_scaled)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))

gaussian_nb = GaussianNB()
gaussian_nb.fit(X_train_scaled, y_train_resampled)
y_pred_nb = gaussian_nb.predict(X_test_scaled)
print("GaussianNB Accuracy:", accuracy_score(y_test, y_pred_nb))

sgd = SGDClassifier(random_state=42)
sgd.fit(X_train_scaled, y_train_resampled)
y_pred_sgd = sgd.predict(X_test_scaled)
print("SGD Accuracy:", accuracy_score(y_test, y_pred_sgd))

xgb = XGBClassifier(random_state=42)
xgb.fit(X_train_scaled, y_train_resampled)
y_pred_xgb = xgb.predict(X_test_scaled)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))

decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train_scaled, y_train_resampled)
y_pred_dt = decision_tree.predict(X_test_scaled)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train_scaled, y_train_resampled)
y_pred_rf = random_forest.predict(X_test_scaled)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

voting_classifier = VotingClassifier(
    estimators=[
        ('lr', logistic_regression),
        ('knn', knn),
        ('svm', svm),
        ('nb', gaussian_nb),
        ('sgd', sgd),
        ('xgb', xgb),
        ('dt', decision_tree),
        ('rf', random_forest)
    ],
    voting='hard'
)
voting_classifier.fit(X_train_scaled, y_train_resampled)
y_pred_combined = voting_classifier.predict(X_test_scaled)
print("Voting Classifier Accuracy:", accuracy_score(y_test, y_pred_combined))

app = FastAPI()


class TAVR(BaseModel):
    Age: float
    Sex: int
    BSA: float
    BMI: float
    HTN: int
    CAD: int
    DM: int
    ACEi_ARB: int
    Beta_Blocker: int
    Aldosteroneantagonist: int
    CCB: int
    AntiPlateletotherthanASA: int
    ASA: int
    AntiplateletTherapy: int
    Diuretics: int
    LVEF: float
    SystolicBP: float
    DiastolicBP: float 
    LVOT: float
    ValveCode: int
    ValveSize: int
    BaselineRhythm: int
    PR: float
    QRS: int
    QRSmorethan120: int
    FirstdegreeAVblock: float
    Baseline_conduction_disorder: int
    BaselineRBBB: int
    DeltaPR: float
    DeltaQRS: int
    PacemakerImplantation: int

def main():
    return 'New-onset LBBB'

@app.post('/predict')
def predict(request: TAVR):
    input_data = request.dict()
    selected_features = selector.transform(pd.DataFrame([input_data]))
    scaled_input = scaler.transform(selected_features)
    prediction = int(voting_classifier.predict(scaled_input)[0])
    return {"prediction": prediction}

if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8000)
    