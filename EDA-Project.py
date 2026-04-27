"""
Final Project
Please find my PowerPoint presentation in the folder
labeled  Data Analysis Process Project
Analysis on Kaggles Airbnb listings dataset in NYC
"""
import colorama
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import seaborn as sn
import sklearn
from colorama import Fore, Style, init
init()

dataset = pd.read_csv('AB_NYC_2019.csv')

# Initial inspection, determined what I needed and then commented out

# print(dataset.head().to_string())
# print()
# print(dataset.tail().to_string())
# print()
# print(dataset.describe().to_string())
# print()
# print(dataset.info())
# print()

# Creating a smaller dataset of just the variables I needed and inspecting the data again
df = dataset[['room_type', 'price', 'number_of_reviews', 'availability_365', 'neighbourhood_group']].copy()
print(Fore.BLUE + 'This is the reduced dataframe head and tail:\n' + Style.RESET_ALL)
print(df.head().to_string())
print()
print(df.tail().to_string())
print()
print(Fore.BLUE + 'This is the descriptive analysis:\n' + Style.RESET_ALL)
print(df.describe())
print()
print(Fore.BLUE + 'This is the info:\n' + Style.RESET_ALL)
print(df.info())
print()
print(Fore.BLUE + 'These are the null values:\n' + Style.RESET_ALL)
print(df.isnull().sum())
print()
print(Fore.BLUE + 'These are the unique value counts for room_type :\n' + Style.RESET_ALL)
print(df['room_type'].value_counts())
print()
print(Fore.BLUE + 'These are the unique value counts for neighbourhood_group :\n' + Style.RESET_ALL)
print(df['neighbourhood_group'].value_counts())
print()
numeric_columns = ['price', 'number_of_reviews', 'availability_365']
print(Fore.BLUE + 'These are the number of negative values:\n ' + Style.RESET_ALL)
print((df[numeric_columns] < 0).sum())
print()
print(Fore.BLUE + 'This is the skewness:\n' + Style.RESET_ALL)
print(df[numeric_columns].skew())
print()
print(Fore.BLUE + 'These are the number of duplicates:\n ' + Style.RESET_ALL)
print(df.duplicated().sum())
print(df.duplicated().mean() *100, '%')
print()



# Data Cleaning - only issue found was duplicated rows

df = df.drop_duplicates()
print(Fore.BLUE + 'These are the number of duplicates now:\n ' + Style.RESET_ALL)
print(df.duplicated().sum())
print()


# Exploratory analysis
#Exploring Price and Number of Reviews
fig, x = plt.subplots(2,2,figsize = (10,10))
fig.suptitle('Univariate Analysis (Part One)')

plt.subplot(2,2,1)
sns.histplot(df['price'], color='purple')
plt.title('Price Distribution')

plt.subplot(2,2,2)
sns.boxplot(x=df['price'], color='lightblue')
plt.title('Price')

plt.subplot(2,2,3)
sns.histplot(df['number_of_reviews'], color='purple')
plt.title('Number of Reviews Distribution')

plt.subplot(2,2,4)
sns.boxplot(x=df['number_of_reviews'], color='lightblue')
plt.title('Number of Reviews')

plt.tight_layout()
#plt.savefig('Univariate Analysis (Part One)')
plt.show()

#Exploring Availablability, Room Type and Neighborhood
fig, x = plt.subplots(2,2,figsize = (10,10))
fig.suptitle('Univariate Analysis (Part Two)')

plt.subplot(2,2,1)
sns.histplot(df['availability_365'], color='purple')
plt.title('Availability 365 Distribution')

plt.subplot(2,2,2)
sns.boxplot(x=df['availability_365'], color='lightblue')
plt.title('Availability 365')

plt.subplot(2,2,3)
sns.boxplot(x=df['room_type'], color='lightblue')
plt.title('Room Type')

plt.subplot(2,2,4)
sns.boxplot(x=df['neighbourhood_group'], color='purple')
plt.title('Neighborhood Group')

plt.tight_layout()
#plt.savefig('Univariate Analysis (Part Two)')
plt.show()

"""
From our analysis above I can see a heavy right tailed distribution for price
and number of reviews. Before I analyze further, I will use log transform for these
variables for better readability and visualize my analysis again.
"""
df['log_price'] = np.log1p(df['price'])
df['log_number_of_reviews'] = np.log1p(df['number_of_reviews'])

# Checking the skewness before and after log transform
skew_of_price = df['price'].skew()
skew_log_price = df['log_price'].skew()
skew_of_reviews= df['number_of_reviews'].skew()
skew_log_reviews = df['log_number_of_reviews'].skew()

skew_table = pd.DataFrame({
    'Variable':['Price','Number of Reviews'],
    'Skew (before)': [skew_of_price,skew_of_reviews],
    'Skew(after)': [skew_log_price,skew_log_reviews],})
skew_table = skew_table.round(4)

print(Fore.BLUE + 'This is the skewness comparison table after log transform:\n' + Style.RESET_ALL)
print(skew_table)
print()

# reviewing my distributions again

fig, x = plt.subplots(2,2,figsize =(10,10))
fig.suptitle('Univariate Analysis after Log Transform')

plt.subplot(2,2,1)
sns.histplot(df['log_price'], color='purple')
plt.title('Price Distribution')

plt.subplot(2,2,2)
sns.boxplot(x=df['log_price'], color='lightblue')
plt.title('Price')

plt.subplot(2,2,3)
sns.histplot(df['log_number_of_reviews'], color='purple')
plt.title('Number of Reviews Distribution')

plt.subplot(2,2,4)
sns.boxplot(x=df['log_number_of_reviews'], color='lightblue')
plt.title('Number of Reviews')

plt.tight_layout()
#plt.savefig('Univariate Analysis after Log Transform')
plt.show()


# Now I will plot my scatter plots

fig, ax = plt.subplots(2,2, figsize = (15,12))
ax[1,1].axis('off')
fig.suptitle('Correlation Analysis')

plt.subplot(2,2,1)
sns.regplot(data=df, x='log_number_of_reviews', y='log_price', scatter_kws={'color': 'red'}, line_kws={'color': 'blue'})
plt.title('Number of Reviews vs. Price')

plt.subplot(2,2,2)
sns.regplot(data=df, x='availability_365', y='log_price', scatter_kws={'color': 'red'}, line_kws={'color': 'blue'})
plt.title('Availabilty vs. Price')

plt.subplot(2,2,3)
sns.regplot(data=df, x='log_number_of_reviews', y='availability_365', scatter_kws={'color': 'red'}, line_kws={'color': 'blue'})
plt.title('Availability 365 vs. Number of Reviews')

#plt.savefig('Correlation Analysis')
plt.show()


# We will now explore number of reviews with room type and neighborhood
fig, ax = plt.subplots(1,2, figsize = (10,6))
fig.suptitle("Number of reviews in reference to Room Type and Neighborhood")

ordered_room_type_reviews = (df.groupby('room_type')['log_number_of_reviews'].mean().sort_values(ascending=False).index)
plt.subplot(1,2,1)
sns.barplot(data=df, x='room_type', y='log_number_of_reviews', order=ordered_room_type_reviews, color='crimson')
plt.xticks(rotation=45)
plt.title('Number of Reviews vs. Room Type')

ordered_neighbourhood_reviews = (df.groupby('neighbourhood_group')['log_number_of_reviews'].mean().sort_values(ascending=False).index)
plt.subplot(1,2,2)
sns.barplot(data=df, x='neighbourhood_group', y='log_number_of_reviews', order=ordered_neighbourhood_reviews, color='crimson')
plt.xticks(rotation=45)
plt.title('Number of Reviews vs. Neighborhood')

plt.tight_layout()
#plt.savefig('reviews')
plt.show()


#I will now explore further regarding room type,neighborhood and price

fig, ax = plt.subplots(1,2, figsize = (12,6))
fig.suptitle('Room Type and Neighborhood')

ordered_room_type = (df.groupby(['room_type'])['log_price'].mean().sort_values(ascending=False).index)
plt.subplot(1,2,1)
sns.barplot(data=df, x='room_type', y='log_price', order=ordered_room_type, color='#6a0dad')
plt.xticks(rotation=45)
plt.title('Room Type/Price')

ordered_neighbourhood = (df.groupby(['neighbourhood_group'])['log_price'].mean().sort_values(ascending=False).index)
plt.subplot(1,2,2)
sns.barplot(data=df, x='neighbourhood_group', y='log_price', order=ordered_neighbourhood, color='#6a0dad')
plt.xticks(rotation=45)
plt.title('Neighborhood/Price')

plt.tight_layout()
#plt.savefig('Bar Plots')
plt.show()

# Now I will use machine modeling to predict airbnb price based on room type and neighbourhood
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# define variables
X = df[['neighbourhood_group', 'log_price']]
y = df['room_type']

# encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train the models
classifier_model = DecisionTreeClassifier(random_state=42)
classifier_model.fit(X_train, y_train)

logistic_regression_model = LogisticRegression(max_iter=800)
logistic_regression_model.fit(X_train, y_train)

random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train, y_train)


# make predictions
y_pred_dtc = classifier_model.predict(X_test)
y_pred_lr = logistic_regression_model.predict(X_test)
y_pred_rfc = random_forest_model.predict(X_test)

# Evaluate performance and print results for readability
print(Fore.BLUE + "Decision Tree Classifier performance evaluation:\n" + Style.RESET_ALL)
accuracy_dtc = accuracy_score (y_test, y_pred_dtc)
confusion_matrix_dtc = confusion_matrix(y_test, y_pred_dtc)
classification_report_dtc = classification_report(y_test, y_pred_dtc)

print(f'The accuracy: {accuracy_dtc:.4f}')
print()
print(f'The confusion matrix:\n {confusion_matrix_dtc}')
print()
print(f'The classification report:\n {classification_report_dtc}')
print()

print(Fore.BLUE + 'Linear Regression performance evaluation:\n' + Style.RESET_ALL)
accuracy_lr = accuracy_score (y_test, y_pred_lr)
confusion_matrix_lr = confusion_matrix(y_test, y_pred_lr)
classification_report_lr = classification_report(y_test, y_pred_lr)

print(f'The accuracy: {accuracy_lr:.4f}')
print()
print(f'The confusion matrix:\n {confusion_matrix_lr}')
print()
print(f'The classification report:\n {classification_report_lr}')
print()

print(Fore.BLUE + 'Random Forest performance evaluation:\n' + Style.RESET_ALL)
accuracy_rfc = accuracy_score (y_test, y_pred_rfc)
confusion_matrix_rfc = confusion_matrix(y_test, y_pred_rfc)
classification_report_rfc = classification_report(y_test, y_pred_rfc)

print(f'The accuracy: {accuracy_rfc:.4f}')
print()
print(f'The confusion matrix: \n {confusion_matrix_rfc}')
print()
print(f'The classification report: \n {classification_report_rfc}')
print()








