import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

titanicdata=pd.read_csv("Titanic-Dataset.csv", index_col="PassengerId")


# About Dataset

print("--- Dataset Information ---")
print(titanicdata.info())
print("--- Statistics ---")
print(titanicdata.describe())

# Data Cleaning

print("--- Missing values ---")
print(titanicdata.isnull().sum())

titanicdata=titanicdata.drop("Cabin", axis=1)
titanicdata=titanicdata.fillna({"Age":titanicdata["Age"].median(), "Embarked":titanicdata["Embarked"].mode()[0]})

print("After cleaning the data :")
print(titanicdata.isnull().sum())

# Survival Analysis

print("Survival Counts:")
print(titanicdata["Survived"].value_counts())

print("Survival Percentage:")
print(titanicdata["Survived"].value_counts(normalize=True) * 100)

sns.countplot(x="Survived", data=titanicdata)
plt.xticks([0, 1], ["Dead", "Alive"])
plt.title("Survival Count")
plt.show()

# Embarked Analysis (Embarked vs Survival)

print("Embarked vs Survival:")
print(pd.crosstab(titanicdata["Embarked"], titanicdata["Survived"]))

ax = sns.countplot(x="Embarked", hue="Survived", data=titanicdata)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ["Dead", "Alive"], title="Status")
plt.title("Embarked vs Survival")
plt.show()

# Correlation Analysis

titanic_corr = titanicdata.copy()

titanic_corr["Sex"] = titanic_corr["Sex"].map({
    "male": 0,
    "female": 1
})

titanic_corr["Embarked"] = titanic_corr["Embarked"].map({
    "S": 0,
    "C": 1,
    "Q": 2
})

corr = titanic_corr.corr(numeric_only=True)
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Conclusion
print()
print("---- CONCLUSION ----")
print("1. Overall survival rate:",end="")
print(round(titanicdata["Survived"].mean()*100,2), "%")

print("2. Survival by Gender:")
print(pd.crosstab(titanicdata["Sex"], titanicdata["Survived"], normalize="index")*100)

print("3. Survival by Passenger Class:")
print(pd.crosstab(titanicdata["Pclass"], titanicdata["Survived"], normalize="index")*100)

print("4. Average Fare by Survival:")
print(titanicdata.groupby("Survived")["Fare"].mean())

print("5. Average Age by Survival:")
print(titanicdata.groupby("Survived")["Age"].mean())