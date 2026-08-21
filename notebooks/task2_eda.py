import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
df = pd.read_csv("../data/titanic.csv")

print("Dataset loaded successfully!")
print(df.head())
# Basic information about the dataset

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns.tolist())

print("\nDATA TYPES")
print(df.dtypes)

print("\nDATASET INFORMATION")
df.info()

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

print("\nSTATISTICAL SUMMARY")
print(df.describe())
# Survival count visualization

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="Survived")

plt.title("Titanic Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")

plt.tight_layout()

plt.savefig("../visualizations/survival_count.png")

plt.show()
plt.savefig("../visualizations/survival_count.png")
# Survival by gender

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="Sex", hue="Survived")

plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")

plt.tight_layout()

plt.savefig("../visualizations/survival_by_gender.png")

plt.show()
# Survival by passenger class

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="Pclass", hue="Survived")

plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")

plt.tight_layout()

plt.savefig("../visualizations/survival_by_class.png")

plt.show()
# Survival by passenger class

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="Pclass", hue="Survived")

plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")

plt.tight_layout()

plt.savefig("../visualizations/survival_by_class.png")

plt.show()
# Age distribution

plt.figure(figsize=(7, 5))

sns.histplot(data=df, x="Age", bins=30, kde=True)

plt.title("Age Distribution of Titanic Passengers")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.tight_layout()

plt.savefig("../visualizations/age_distribution.png")

plt.show()
# Fare distribution

plt.figure(figsize=(7, 5))

sns.histplot(data=df, x="Fare", bins=30, kde=True)

plt.title("Fare Distribution of Titanic Passengers")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")

plt.tight_layout()

plt.savefig("../visualizations/fare_distribution.png")

plt.show()
# Survival rate analysis

print("\nSURVIVAL RATE BY GENDER")
print(df.groupby("Sex")["Survived"].mean() * 100)

print("\nSURVIVAL RATE BY PASSENGER CLASS")
print(df.groupby("Pclass")["Survived"].mean() * 100)
