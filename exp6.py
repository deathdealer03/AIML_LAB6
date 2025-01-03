import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# load the datasets
data=pd.read_csv('exp6.csv')
#  simulate missing value by randomly setting some values to nan for demonstration purpose
data_missing=data.copy()
data_missing.loc[0, 'Age']=np.nan #simulate a missing value a age
data_missing.loc[2, 'Salary']=np.nan #simulate a missing value a salary

# handling missing values
# filling missing numerical values wtih the median (avoiding chained assigned warning)
data_missing['Age']=data_missing['Age'].fillna(data_missing['Age'].median())
data_missing['Salary']=data_missing['Salary'].fillna(data_missing['Salary'].median())

#  detecting outlinear using IQR (INTERQUARTILE RANGE) method
Q1= data_missing[['Age','Salary']].quantile(0.25)
Q3= data_missing[['Age','Salary']].quantile(0.75)
IQR=Q3-Q1

#  identifying outlinear (1.5*iqr rule)

outlinears= (data_missing[['Age', 'Salary']] <(Q1-1.5*IQR))| ((data_missing[['Age', 'Salary']] <( Q3-1.5*IQR)))

# crapping outlinear for each column if outlinear replace with max/min allowed value

data_no_outliers=data_missing.copy()
for column in ['Age','Salary']:
    lower_bound=Q1[column]-1.5*IQR[column]
    upper_bound=Q3[column]+1.5*IQR[column]
    # relace value below the lower bound
    data_no_outliers[column]=np.where(data_no_outliers[column]<lower_bound,lower_bound,data_no_outliers[column])
    # replace value above the upper bound

    data_no_outliers[column]=np.where(data_no_outliers[column]>upper_bound,upper_bound,data_no_outliers[column])
    # comparing differences
    # finding changes agter missing value handling
    missing_value_changes=data_no_outliers.compare(data_missing)
# finding changes after outliners handling
outliner_value_changes=data_no_outliers.compare(data_missing)

    # displaying the difference
print("\n cahnges After handling Missing values:")
print(missing_value_changes)
print("\n changes after handling outlinears:")
print(outliner_value_changes)

# visualization the effect of outlier treatment
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
sns.boxplot(data=data_missing[['Age', 'Salary']])
plt.title('Before outlier handling')

plt.subplot(1,2,2)
sns.boxplot(data=data_no_outliers[['Age','Salary']])
plt.title('After outlier handling')
plt.tight_layout()
plt.show()