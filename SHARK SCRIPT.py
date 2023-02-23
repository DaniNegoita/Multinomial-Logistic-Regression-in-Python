import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import random
import os


os.getcwd()
dati = pd.read_excel("C:\\Users\\danin\\Desktop\\Datasets\\Australian Shark-Incident Database Public Version.xlsx")
dati
# ======================
#      PREPROCESSING
# ======================

#Columns' names
dati.columns


#Rename columns
dati.rename(columns ={'Victim.injury':'Injury'},inplace=True)
dati.rename(columns ={'Victim.activity':'Activity'},inplace=True)
dati.rename(columns ={'Shark.common.name':'Shark name'},inplace=True)
dati.rename(columns ={'Victim.gender':'Gender'},inplace=True)
dati.rename(columns ={'Victim.age':'Age'},inplace=True)
dati.rename(columns ={'Site.category':'Site'},inplace=True)

#Percentages of missing values
empty = (dati.isna().sum().sort_values()/len(dati)) 
empty


# The last 5 features have high percentages of missings 
# Month                    0.000000
# Year                     0.000000
# Date_mm/y                0.000000
# Injury                   0.000000
# State                    0.000000
# Site                     0.000000
# Provoked/unprovoked      0.003344
# Gender                   0.014214
# Activity                 0.020903
# Shark name               0.043478
# N.sharks                 0.074415
# Injury.severity          0.289298
# Age                      0.415552
# Victim.aware.of.shark    0.486622
# Time                     0.568562
# Distance.to.shore.m      0.702341 

dati.drop(columns = ["Injury.severity", 
                     "Age", 
                     "Victim.aware.of.shark", 
                     "Time", 
                     "Distance.to.shore.m"], inplace = True)


# I will also drop these two columns as they are irrelevant

dati.drop(columns = ["Date_mm/y","Year"], inplace = True)


# Check duplicates 
duplicati = len(dati) - len(dati.drop_duplicates())
#258 duplicates
dati = dati.drop_duplicates()


# For the other features, I will start filling the null values with simple values
# that do not invalidate columns' reliability

dati["N.sharks"] = dati["N.sharks"].fillna(1)

dati["Provoked/unprovoked"] = dati["Provoked/unprovoked"].fillna(0)

dati["Gender"] = dati["Gender"].fillna(0)


# Checking the most frequent activity
most_frequent_activity = dati["Activity"].groupby(dati["Activity"]).count()
most_frequent_activity.sort_values()

# Output
# Floating                 2
# Motorised Boating        2
# Standing in water       24
# Unmotorised Boating     29
# Fishing                 34
# Diving                  85
# Spearfishing           107
# Snorkelling            110
# Boarding               202
# Swimming               320

# Swimming is the most common activity: missing values and the least common
# activities will be imputed with Swimming

# For the missings
dati["Activity"] = dati["Activity"].fillna("Swimming")  


# For the least common activities
dati["Activity"] = dati["Activity"].replace("Motorised Boating", "Swimming")
dati["Activity"] = dati["Activity"].replace("Floating", "Swimming")

# Same procedure for the sharks' names
most_frequent_shark=dati["Shark name"].groupby(dati["Shark name"]).count()
most_frequent_shark.sort_values()

# Output

# Bull Shark                   135
# Wobbegong                    163
# Tiger Shark                  185
# White Shark                  256

random.seed(0) # nullify the variability of the data or random generates numbers

# For the missing values           
dati["Shark name"].fillna(random.choice(["White Shark", 
                                         "Wobbegong", 
                                         "Tiger Shark", 
                                         "Bull Shark"]), inplace = True )


less_frequent_name = most_frequent_shark[most_frequent_shark < 10]
less_frequent_name = dict(less_frequent_name).keys()
less_frequent_name

# Output
# dict_keys(['Blacktip Reef Shark', 'Blind Shark', 'Broadnose Sevengill Shark', 
#'Dogfish', 'Dusky Shark', 'Galapagos Shark', 'Grey Nourse Shark', 
#'Grey Reef Shark', 'Hammerhead Shark ', 'Lemon Shark', 'Port Jackson Shark', 
#'School Shark', 'Sevengill Shark', 'Shortfin Mako Shark', 'Silvertip Shark', 
#'Whitetip Reef Shark'])

# For the least frequent shark names
for el in less_frequent_name:
    el = el.strip()
    dati["Shark name"].replace(el, random.choice(["White Shark", 
                                         "Wobbegong", 
                                         "Tiger Shark", 
                                         "Bull Shark"]), inplace = True)


# ======================
#     ENCODING
# ======================

# Categorical features will be transformed into new numerical columns 
# to which values of 1 or 0 will be assigned.

from sklearn.preprocessing import OneHotEncoder

# Binary features
dati['Provoked/unprovoked'] = dati['Provoked/unprovoked'].replace("Provoked",1)
dati['Provoked/unprovoked'] = dati['Provoked/unprovoked'].replace("Unprovoked",0)

dati['Gender'] = dati['Gender'].replace("Female",1)
dati['Gender'] = dati['Gender'].replace("Male",0)


# Categorical features
dati.State
State = OneHotEncoder(sparse = False)
dati['New_South_Wales'],dati['Western_Australia'],dati['Tasmania'],dati['South_Australia'],dati['Queensland'],dati['Victoria'],dati['Northern_TerritorYes'] = State.fit_transform(dati[["State"]]).T
dati.drop(columns = "State", inplace = True)

Site = OneHotEncoder(sparse = False)
dati['Coastal'], dati['Estuary/Harbour'], dati['Island Open Ocean'], dati["River"], dati['Ocean/Pelagic'], dati["Other: Fish Farm"] = Site.fit_transform(dati[["Site"]]).T
dati.drop(columns = "Site", inplace = True)

Shark_name = OneHotEncoder(sparse = False)
dati['White Shark'], dati['Tiger Shark'], dati['Bull Shark'] , dati['Whaler Shark'], dati['Wobbegong'], dati['Bronze Whaler Shark'], dati['Hammerhead Shark'] = Shark_name.fit_transform(dati[["Shark name"]]).T
dati.drop(columns = "Shark name", inplace = True)


Activity = OneHotEncoder(sparse = False)
dati['Swimming'], dati['Fishing'], dati['Spearfishing'], dati['Unmotorised Boating'], dati['Snorkelling'], dati['Diving'], dati['Standing in water'], dati['Boarding'] = Activity.fit_transform(dati[["Activity"]]).T
dati.drop(columns = "Activity", inplace = True)

# Date columns

dati["Month"] = 2 * np.pi * (dati["Month"]/ dati["Month"].max())
dati["Cos Month"] = np.cos(dati["Month"])
dati["Sin Month"] = np.sin(dati["Month"])

dati.drop(columns = "Month", inplace = True)



#==========================
# MULTICOLLINEARITY CHECK
#==========================

from statsmodels.stats.outliers_influence import variance_inflation_factor

#Correlation matrix
corr = dati.corr()
corr 

corr_df = corr.stack().reset_index()

corr_df.columns = ["feature1", "feature2", "correlation"]  
corr_df.sort_values(by = "correlation", ascending = False, inplace = True)
corr_df = corr_df[corr_df["feature1"] != corr_df["feature2"]]

def vif_calc(x):
    vif = pd.DataFrame()
    vif["variable"] = x.columns
    vif["vif"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    vif.sort_values(by = ["vif"], ascending = False)

    return vif

# output:
#                 variable       vif
# 0    Provoked/unprovoked  2.156410
# 1               N.sharks  1.039648
# 2                 Gender  1.071477
# 3        New_South_Wales       inf
# 4      Western_Australia       inf
# 5               Tasmania       inf
# 6        South_Australia       inf
# 7             Queensland       inf
# 8               Victoria       inf
# 9   Northern_TerritorYes       inf
# 10               Coastal       inf
# 11       Estuary/Harbour       inf
# 12     Island Open Ocean       inf
# 13                 River       inf
# 14         Ocean/Pelagic       inf
# 15      Other: Fish Farm       inf
# 16           White Shark       inf
# 17           Tiger Shark       inf
# 18            Bull Shark       inf
# 19          Whaler Shark       inf
# 20             Wobbegong       inf
# 21   Bronze Whaler Shark       inf
# 22     Hammerhead Shark        inf
# 23              Swimming       inf
# 24               Fishing       inf
# 25          Spearfishing       inf
# 26   Unmotorised Boating       inf
# 27           Snorkelling       inf
# 28                Diving       inf
# 29     Standing in water       inf
# 30              Boarding       inf
# 31             Cos Month  1.057046
# 32             Sin Month  1.084114

# Only "Month, Provoked/unprovoked, gender, and n_sharks
# are not strongly correlated

# Save a copy of the updated dataframe to be used for the Multinomial Regression 
dati.to_excel("dati.xlsx", index = False)


#==================================
#  MULTINOMIAL LOGSITIC REGRESSION
#==================================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Split dataset in features and target variables
features = ["Provoked/unprovoked", "Gender", "N.sharks"]
x = dati[features]
y = dati.Injury 

#Replace the target variable with numeric values
dati["Injury"].unique()
dati["Injury"] = dati["Injury"].replace({'Uninjured':1, 'Injured':2, 'Fatal':3})


X_train, X_test,y_train,y_test=train_test_split(x,y,
                                                test_size=0.3, 
                                                random_state=0) 
# Instantiate the model
mlr=LogisticRegression(max_iter=120,multi_class="multinomial",random_state=0)
X_train
y_train

# Fitting the model
mlr.fit(X_train,y_train)

# Predict
pred_train = mlr.predict(X_train)
pred_test = mlr.predict(X_test)

pred_train
pred_test

# Models' performance
acc_train=accuracy_score(y_train, pred_train)
acc_test= accuracy_score(y_test, pred_test)

acc_train
# 58%

acc_test
# 65%

# Low accuracy for both sets but they are close meaning that the model does not
# atleast underfit or overfit


  # Model evaluation using Confusion Matrix
from sklearn import metrics
matrix = metrics.confusion_matrix(y_test, pred_test)
matrix

# Plot confusion matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class_names=[1,2,3] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual values')
plt.xlabel('Predicted values')

# Injured class was accurately predicted 184 times.  Fatal and Uninjured predicted 0 times.
# Non-diagonal outputs: 55 instances of False Negatives (Type 2 error) and 43
# instances of False Positives (Type 1 error).

# Evaluate model's accuracy, precision, and recall.
from sklearn.metrics import classification_report
cn = ["uninjured", "injured", "fatal"]
print(classification_report(y_test, pred_test, target_names=cn)) 


# Output: 
                 # precision    recall  f1-score   support

  # uninjured       0.00      0.00      0.00        43
   #  injured       0.65      1.00      0.79       184
     #  fatal       0.00      0.00      0.00        55

   # accuracy                           0.65       282
  # macro avg       0.22      0.33      0.26       282
#weighted avg       0.43      0.65      0.52       282

# Accuracy --> classification rate of 65%
# Precision --> the model predicted the likelihood of being injured 65% of the time
# Recall --> the model can identify injured cases 100% of the time

# The performance of the model is not the best, it carries limitations most likely
# due to model misspecification (more features could be added).
















