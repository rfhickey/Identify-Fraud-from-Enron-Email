#!/usr/bin/python

import sys
import pickle
import numpy
import matplotlib.pyplot as plt
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from collections import defaultdict

### Task 1: Select what features you'll use.
### used_features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
used_features_list = ['poi','salary','total_payments','bonus','total_stock_value',
                 'exercised_stock_options','long_term_incentive','to_messages',
                 'from_poi_to_this_person', 'from_messages',
                 'from_this_person_to_poi','shared_receipt_with_poi'] 

### Load the dictionary containing the dataset

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
# I want to get a sense of the data I am working with here
# Number of People in the Dataset: 146

number_of_people = len(data_dict)    
    
print number_of_people   

# Of These 146 People, I Wanted to See How Many are Persons of Interest. There
# are 18
poi_count = 0
for person, feature in data_dict.iteritems():
    if feature['poi']:
        poi_count += 1

print poi_count

# We Were Given a List of Names of People from a USA Today Article Who Were Seen
# As Persons of Interest. Do These Match Up with the POIs in the
# final_project_dataset.pkl file? The Answer Here Is No As There are 35 In That
# List

with open("poi_names.txt") as f:
    pois_total_in_list = len(f.readlines()[2:])
print(pois_total_in_list)
  
# Let's See How Many Features There Are for Each Person. To do this, I Printed
# The Length of the Dictionary Storing the Values for the first person in the 
# PDF Provided in the final_project folder. There are 21 features, which 
# Matches with the Number Provided in the Udacity Final Project Page 
print (len(data_dict['ALLEN PHILLIP K']))

### Task 2: Remove outliers

# Let's Do a Quick Test to See What Some of the Most Intuitively Easy to
# Understand Featues Look Like

features_outlier_test_1 = ['salary', 'total_stock_value']

features = featureFormat(data_dict, features_outlier_test_1) 

for i in features:
    salary = i[0]
    total_stock_value = i[1]
    plt.scatter(salary, total_stock_value)
   

plt.xlabel("salary")
plt.ylabel("total stock value")
plt.show()

# I remember this from the Udacity lessons that we had a "TOTAL" figure in here.
# Let's make sure we remove that and anything else we find when we do a manual
# scan of the data.

# I wanted to see if there were any odd keys in the dataset that needed to 
# be removed. I found "TOTAL" and something called "THE TRAVEL AGENCY IN THE
# PARK" (that has NaNs for all entries) which are not individuals like 
# the other entries, so I removed them
for key, value in data_dict.iteritems():
    print key

data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')

# I'm interested in seeing how many of the features in the data are missing 
# values and how many values they are missing

missing_data = defaultdict(float)
value_list = next(data_dict.itervalues()).keys()
for key in data_dict.itervalues():
    for value in value_list:
        if key[value] == "NaN":
            missing_data[value] += 1
            
print missing_data

# I also want to see what percentage of each feature is missing, so I have created
# the code below based on the cleaned data with 144 people with emails

missing_data_1 = defaultdict(float)

for key in missing_data.itervalues():
    for value in missing_data: 
        missing_data_1[value] = round(missing_data[value] / 144, 2)

print missing_data_1

# You can see from these results that there is a lot of missing data here. I would
# say that we would not want to include deferral_payments, restricted_stock_deferred,
# deferred_income, loan_advances, director_fees, anf long_term_incentive as
# features when training our machine learning algorithm.   

# Since There Seem to Be Decently Robust Data for salary and total_stock_value,
# let's visualize those data again after we have cleaned up the data. 

features_outlier_test_1 = ['salary', 'total_stock_value']

features = featureFormat(data_dict, features_outlier_test_1) 

for i in features:
    salary = i[0]
    total_stock_value = i[1]
    plt.scatter(salary, total_stock_value)

plt.xlabel("salary")
plt.ylabel("total stock value")
plt.show()

# Great, this is looking much better now. Let's see if we can remove the top 
# 10% of outliers from here as well.



salary_feature = ['salary']
total_stock_value_feature = ['total_stock_value']

salary_train = featureFormat(data_dict, salary_feature)
total_stock_value_train = featureFormat(data_dict, total_stock_value_feature)

salary_train = salary_train[:94]
total_stock_value_train = total_stock_value_train[:94]

from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(salary_train, total_stock_value_train)


features_outlier_test_1 = ['salary', 'total_stock_value']

features = featureFormat(data_dict, features_outlier_test_1) 

for i in features:
    salary = i[0]
    total_stock_value = i[1]
    plt.scatter(salary, total_stock_value)

plt.xlabel("salary")
plt.ylabel("total stock value")
plt.plot(salary_train, reg.predict(salary_train))
plt.show()

from outlier_cleaner import outlierCleaner

cleaned_data = []
try:
    predictions = reg.predict(salary_train)
    cleaned_data = outlierCleaner(predictions, salary_train, total_stock_value_train)
except NameError:
    print "your regression object doesn't exist, or isn't name reg"
    print "can't make predictions to use in identifying outliers"

# Select KBest Parameters    
    
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, used_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)