# Multinomial-Logistic-Regression-in-Python

This project develops and predicts a three-class classification using a Python machine-learning technique. 

The project is divided into the following stages: 
1) Pre-processing: removal of columns with high shares of missing values, imputation using the mode or values that did not undermine data’s soundness, duplicate rows were dropped, the transformation of the categorical features with OneHotEncoder, multicollinearity check using the Variance Inflation Factor.
2) Model development and prediction: i) creation of a Logistic Regression classifier specifying the multinomial scheme over one-vs-rest ii) the fitting of  the model on the training set iii) predictions on the training and test sets (the algorithm does not overfit or underfit the data).
3) Model evaluation: Confusion Matrix to visualise class-wise correct and incorrect predictions, along with the metrics for precision, accuracy and recall.

Limitations: 

High shares of missing values 

Potential model misspecification 



Reference: 

Phoebe Meagher. (2022). Australian Shark Incident Database (4.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7608411
