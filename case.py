#importing needed libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder

from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from sklearn.manifold import LocallyLinearEmbedding
from sklearn import manifold, datasets

from sklearn.preprocessing import binarize, LabelEncoder, MinMaxScaler
from sklearn import preprocessing

from sklearn.feature_selection\
    import VarianceThreshold
from sklearn.feature_selection import VarianceThreshold

from scipy.cluster.hierarchy import dendrogram, ward
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score



#reading from file
dataset = pd.read_csv('mental-heath-in-tech-2016_20161114.csv')

ndata = pd.DataFrame(dataset)



#for loop to iterate through every feature for visualization of data
def info_loop(ndata, a):
    for i in range(a):
        colname = ndata.columns[i] #get names of ith column
        values = ndata[colname].value_counts()  #group and count amount of value occurences
        print(f'Column {i}: \n', values)   #print grouped info
        total = sum(values)  #get total of valid input in ith feature
        Miss = 1433 - total  #amount of null input in ith feature
        print('Sum: ', total)  #print total valid input
        print('Missing: ', Miss)   #print amount of missing inputs
        print()  #new lines


#info_loop(ndata, len(ndata.columns)) #can be commented out to call the info loop function forr visualization

#Data Cleansing
ndata.drop(columns = [ "Why or why not?", "Why or why not?.1", "What US state or territory do you live in?",
                     "What US state or territory do you work in?"], inplace = True)


#dropping column with over fifty percent of null values
cols = [42]
ndata.drop(ndata.columns[cols],axis=1,inplace=True)


#Cleaning: Renaming of country feature
ndata.rename(columns = {'What country do you live in?':'Country'}, inplace = True)

#Merging three features together by filling the null values from one by features in the same row of a
#similar feature
(ndata["If maybe, what condition(s) do you believe you have?"]
.fillna(ndata["If yes, what condition(s) have you been diagnosed with?"], inplace=True))

(ndata["If so, what condition(s) were you diagnosed with?"]
.fillna(ndata["If maybe, what condition(s) do you believe you have?"], inplace=True))

#dropping of features
ndata.drop(columns = [ "If yes, what condition(s) have you been diagnosed with?", "What country do you work in?",
                     "If maybe, what condition(s) do you believe you have?"], inplace = True)

#Cleaning Data: gender
ndata.rename(columns = {"What is your gender?":"Gender"}, inplace=True)

trans =  ["M (cis)", "Agender", "Nonbinary", "M (trans, FtM)", "F or Multi-Gender Femme ", "Queer", "Human",
         "fm", "Unicorn", "mtf", "none of your business", "genderqueer woman", "Genderfluid (born female)",
         "Fluid", "Genderfluid", "F-bodied; no feelings about gender", "Transitioned, M2F", "Cis female",
         "nb masculine", "Other/Transfeminine", "cis male", "F (props for making this a freeform field, though)",
         "AFAB", "Cisgender Female", "Bigender", "cisdude", "Other", "Cis male", "M 9:1 female, roughly ",
         "Enby", "Genderqueer", "genderqueer", "cis man", "Cis Male", "M/genderqueer", "Transgender woman",
         "Genderflux demi-girl", "Cis-woman", "Androgynous", "F assigned at birth", "non-binary",
         "Male (cis)", "Female or Multi-Gender Femme", "male 9:1 female, roughly", "Male/genderqueer", "human",
         "Female (props for making this a freeform field, though)  ", "Cis female ", "Male (trans, FtM)    ",
         "I'm a man why didn't you make this a drop down question. You should of asked sex? And I would of answered yes please. Seriously how much text can this take?",
         ]

male = ["Male", "M", "m", "mail", "Mail", "dude", "Malr", "M.", "Sex is male", "M|", "MALE", "Man", "man", "Male ", "male ",
        "Male.", "Dude"]

female = ["Female", "female", "F", "f", "woman", "fm", "fem", " Female", "female ", "female/woman", "Female ", "Woman",
         "I identify as female.", "Female assigned at birth "]

n = ["Nan"]

for (row, col) in ndata.iterrows():

    if col.Gender in male:
        ndata["Gender"].replace(to_replace=col.Gender, value='male', inplace=True)

    if col.Gender in female:
        ndata["Gender"].replace(to_replace=col.Gender, value='female', inplace=True)
    if col.Gender in trans:
        ndata["Gender"].replace(to_replace=col.Gender, value='trans', inplace=True)


#Data Cleaning: Age
ndata["What is your age?"].fillna(ndata["What is your age?"].median(), inplace = True)

# Fill with media() values < 18 and > 120
n = pd.Series(ndata["What is your age?"])
n[n<18] = ndata["What is your age?"].median()
ndata["What is your age?"] = n
n = pd.Series(ndata["What is your age?"])
n[n>70] = ndata["What is your age?"].median()
ndata["What is your age?"] = n

#Creating Ranges of Age
ndata['age_range'] = pd.cut(ndata["What is your age?"], [0,30,45,60,70], labels=["15-30", "31-45", "45-60", "61-70"],
                               include_lowest=True)

ndata.drop(columns=["What is your age?"], inplace=True)

#Renaming the diagnosis featurre by merging first two strings
coname = 'If so, what condition(s) were you diagnosed with?'
first = ndata[coname].str.split().str.get(0)
second = ndata[coname].str.split().str.get(1)
illness = first + " " + second
ndata[coname] = illness

#filling null values with not sure
ndata["If so, what condition(s) were you diagnosed with?"].fillna("Not Sure", inplace=True)

#Renaming the values work position features by first string in text
job_name = 'Which of the following best describes your work position?'
first = ndata[job_name].str.split().str.get(0)
ndata[job_name] = first

#Data Cleansing: Replace 'Which of the following best describes your work position?' with Work Position
ndata["Work Position"] = ndata['Which of the following best describes your work position?'].str.split("|", expand=True)[0]
ndata.drop(columns=['Which of the following best describes your work position?'], inplace=True)

#drop columns with over 70 percent null values
colds = [16,17,18,19,20,21,22,23]
ndata.drop(ndata.columns[colds],axis=1,inplace=True)


ndata.drop(columns = ["Is your primary role within your company related to tech/IT?"], inplace = True)


ndata = ndata.apply(lambda x: x.fillna(x.value_counts().index[0]))

#info_loop(ndata, len(ndata.columns))

#Encoding
#Encoding and creating Datframe for country and illness before and after encoding
"""The aim of this section is to encode and print a list of
countries and their encoded values for understanding after
clustering is carried out"""
c_name = pd.DataFrame(ndata["Country"])  #creating a dataframe to store countries name before encoding

#creating a dataframe to store countries name before encoding
ill_name = pd.DataFrame(ndata["If so, what condition(s) were you diagnosed with?"])

#encoding using label encoder
for i in range(len(ndata.columns)):
    colname = ndata.columns[i] #get names of ith column
    ndata[colname] = LabelEncoder().fit_transform(ndata[colname])

#dataframe for list of encoded country
count_no = pd.DataFrame(ndata["Country"])
country_list = pd.concat([count_no, c_name], axis=1)
country_list.columns = ["Number", "Country"]

#dataframe for unique and sorted list of countries
unique_count = pd.DataFrame(pd.unique(country_list["Country"]))
unique_no = pd.DataFrame(pd.unique(country_list["Number"]))
unique_country_list = pd.concat([unique_no, unique_count], axis=1)
unique_country_list.columns = ["Number", "Country"]
list_of_countries = unique_country_list.sort_values(by = 'Number')  #sorting dataframe by number

#dataframe for list of encoded illnesses
ill_no = pd.DataFrame(ndata["If so, what condition(s) were you diagnosed with?"])
ill_list = pd.concat([ill_no, ill_name], axis=1)
ill_list.columns = ["Number", "Illness"]


#dataframe for unique and sorted list of illnesses
unique_ill = pd.DataFrame(pd.unique(ill_list["Illness"]))
unique_ilno = pd.DataFrame(pd.unique(ill_list["Number"]))
unique_ill_list = pd.concat([unique_ilno, unique_ill], axis=1)
unique_ill_list.columns = ["Number", "Ilness"]
list_of_illness = unique_ill_list.sort_values(by = 'Number')  #sorting dataframe by number




#Feature Variance which reduced our features to fifteen
X = ndata
#y = ndata.loc[:, ["Country"]].values
selector = VarianceThreshold(threshold=0.90) #using a threshold of 90 percent
Var = selector.fit_transform(X)

new_col = ndata[ndata.columns[selector.get_support(indices=True)]]



#Dendogram and figuring out number of clusters
scaled_col = new_col
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))
plt.title("Dendrograms")
dend = shc.dendrogram(shc.linkage(scaled_col, method='ward'))
plt.axhline(y=300, color='r', linestyle='--')
plt.savefig("Dendogram.jpg")


#agglomerative clustering and sihlouette score


cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
cluster.fit_predict(scaled_col)

lab = cluster.labels_

#add cluster information to the dataframe
scaled_col['lab']=lab

#score
S = silhouette_score(scaled_col, lab)


#clustering country and Illness
plt.figure(figsize=(10, 7))
plt.scatter(scaled_col['If so, what condition(s) were you diagnosed with?'], scaled_col['Country'], c=cluster.labels_)
plt.xlabel("Illness")
plt.ylabel("Country")
plt.title("Mental health by Country")
plt.savefig("Cluster.jpg")




print()
print("Sihlouette Score: ",S)
print(list_of_countries.to_string(index = False))
print(list_of_illness.to_string(index = False))


    
    