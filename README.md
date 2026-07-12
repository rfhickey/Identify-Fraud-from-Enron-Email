# Identify Fraud from Enron Email

This is the working-draft repository for a Udacity Data Analyst Nanodegree machine learning project: using the public Enron financial and email dataset to identify persons of interest (POIs) in the Enron fraud case. It contains the in-progress notebook and script iterations from that project.

This project was superseded by the finished, cleaned-up version at [Identify-Fraud-from-Enron-Email_Final](https://github.com/rfhickey/Identify-Fraud-from-Enron-Email_Final). That repository has the final feature selection, classifier, and write-up; this one is kept for history only.

## Repository structure

- `poi_id.py` and `poi_id.py(version_1)`, `poi_id.py(version2)`, `poi_id.py(version 3)`: successive drafts of the main analysis script (feature selection, outlier removal, classifier comparison and tuning). File names are left as originally submitted.
- `tester.py`: Udacity-provided evaluation script (stratified shuffle split validation of precision and recall).
- `poi_email_addresses.py`: helper for matching POI names to email addresses.
- `poi_names.txt`: list of persons of interest referenced during the project.
- `enron61702insiderpay.pdf`: source financial data (insider payments) for the Enron dataset.
- `final_project_dataset.pkl`, `final_project_dataset_modified.pkl`: the Enron dataset used for the project, in pickled dictionary form.
- `my_dataset.pkl`, `my_classifier.pkl`, `my_feature_list.pkl`: intermediate outputs dumped by the draft script for grading.

## How to run

Written and run against Python 2 with scikit-learn 0.17-era APIs (`sklearn.cross_validation`, `sklearn.grid_search`), consistent with the Nanodegree environment at the time (2018). Library versions are historical and are not pinned here; a modern scikit-learn will not run this code without changes.

For the current, working version of this analysis, see the final project repository linked above.
