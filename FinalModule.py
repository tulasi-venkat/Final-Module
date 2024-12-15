import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.outliers_influence as vif
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

#Load Dataset
file_path = 'Final_Report_of_the_Asian_American_Quality_of_Life__AAQoL_.csv'
df = pd.read_csv(file_path)

#Data Cleaning
satisfaction_mapping = {
    "Strongly disagree": 1, "Disagree": 2, "Slightly disagree": 3,
    "Neither agree or disagree": 4, "Slightly agree": 5, "Agree": 6, "Strongly agree": 7
}

mental_health_mapping = {
    "Poor": 1, "Fair": 2, "Good": 3, "Very Good": 4, "Excellent": 5
}

ethnic_identity_mapping = {
    "Not at all": 1, "Not very close": 2, "Somewhat close": 3, "Very close": 4
}

df['Life_Satisfaction'] = df['Satisfied With Life 1'].map(satisfaction_mapping).fillna(0)
df['Mental_Health'] = df['Present Mental Health'].map(mental_health_mapping).fillna(0)
df['Ethnic_Identity'] = df['Identify Ethnically'].map(ethnic_identity_mapping).fillna(0)

def preprocess_income(income):
    if pd.isna(income):
        return np.nan
    elif '-' in income: 
        low, high = income.replace('$', '').replace(',', '').split(' - ')
        return (float(low) + float(high)) / 2
    elif 'over' in income: 
        return float(income.replace('$', '').replace(',', '').split(' ')[0]) + 1
    else:
        return np.nan

df['Income'] = df['Income'].apply(preprocess_income) / 1000

columns_to_clean = ['Life_Satisfaction', 'Mental_Health', 'Ethnic_Identity', 'Discrimination ', 'Income', 'Age']
df = df[columns_to_clean].dropna()

#Decision Tree Regression
X = df[['Discrimination ', 'Ethnic_Identity', 'Income', 'Age']]
Y = df['Mental_Health']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

dtree = DecisionTreeRegressor(max_depth=4, random_state=42)
dtree.fit(X_train, Y_train)

Y_pred = dtree.predict(X_test)
print("\nDecision Tree Performance:")
print("Mean Squared Error (MSE):", mean_squared_error(Y_test, Y_pred))
print("R-squared (R2):", r2_score(Y_test, Y_pred))

plt.figure(figsize=(12, 8))
plot_tree(dtree, feature_names=X.columns, filled=True, rounded=True)
plt.title("Decision Tree for Mental Health Prediction")
plt.show()


#Random Forest Regressor
rforest = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rforest.fit(X_train, Y_train)

Y_rf_pred = rforest.predict(X_test)
print("\nRandom Forest Performance:")
print("Mean Squared Error (MSE):", mean_squared_error(Y_test, Y_rf_pred))
print("R-squared (R2):", r2_score(Y_test, Y_rf_pred))

plt.figure(figsize=(8, 6))
plt.bar(X.columns, rforest.feature_importances_)
plt.title('Feature Importance in Random Forest')
plt.ylabel('Importance')
plt.xlabel('Variable')
plt.show()

#Subgroup Analysis
print("\nSubgroup Analysis by Ethnic Identity")
for group in df['Ethnic_Identity'].unique():
    subgroup = df[df['Ethnic_Identity'] == group]
    X_sub = subgroup[['Discrimination ', 'Income', 'Age']]
    Y_sub = subgroup['Mental_Health']

    if len(subgroup) > 50: 
        X_train_sub, X_test_sub, Y_train_sub, Y_test_sub = train_test_split(X_sub, Y_sub, test_size=0.2, random_state=42)
        rf_sub = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        rf_sub.fit(X_train_sub, Y_train_sub)
        Y_pred_sub = rf_sub.predict(X_test_sub)
        
        print(f"\nGroup {group} - Random Forest Performance:")
        print("Mean Squared Error (MSE):", mean_squared_error(Y_test_sub, Y_pred_sub))
        print("R-squared (R2):", r2_score(Y_test_sub, Y_pred_sub))
    else:
        print(f"\nGroup {group} - Not enough data for analysis")


#Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df[['Mental_Health', 'Discrimination ', 'Life_Satisfaction', 'Ethnic_Identity', 'Income', 'Age']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

