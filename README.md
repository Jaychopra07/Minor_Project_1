# Minor_project_1
## Aim

The main aim of this project is to make a reliable model that can predict whether a passenger onboard Titanic survived the incidence or not. This model bases on given information and features.

## Dataset

The dataset for this project is imported from a CSV file, "titanic_train.csv". The dataset have information about passengers onboard the Titanic, which include their name, survival status, class (Pclass), sex (Gender), age (Age) cabin and embark .

## Libraries Used

The following important libraries were used for this project:

- numpy
- pandas
- matplotlib.pyplot
- seaborn
- sklearn.preprocessing.LabelEncoder
- sklearn.model_selection.train_test_split
- sklearn.linear_model.LogisticRegression
- sklearn.metrics.confusion_matrix
- sklearn.metrics.classification_report
  
## Data Exploration and Preprocessing

1. The dataset used here was fetched using pandas (pd) as a DataFrame, and its shape and a glimpse of the starting rows were displayed using `titanic.shape` and `titanic.head()`.
2. The count of passengers who survived and those who did not was visualized using `titanic['Survived'].value_counts()`.
4. The count of survivals was visualized with respect to the Pclass using `sns.countplot(x='Survived', hue='Pclass', data=titanic)`.
5. The count of survivals was visualized with respect to the gender using `sns.countplot(x='Survived', hue='Sex', data=titanic)`.
6. The onboard people were categorised with their spouse and siblings using `sns.countplot(x='SibSp',data=titanic)`.
7. The age group of passengers were categorised using `sns.displot(titanic['Age'].dropna())`.
8. The survival rate by gender was calculated and displayed using `sns.countplot(x='Survived', hue='Sex', data=titanic)`.
9. The 'Sex' column was converted from categorical to numerical values using LabelEncoder from `sklearn.preprocessing`.
10. After encoding the 'Sex' column, non-required columns like 'Age' and 'Cabin' were dropped from the DataFrame.

## Model Training

1. The feature matrix `X` and target vector `Y` were created using relevant columns from the DataFrame.
2. The dataset was split into training and testing sets using `train_test_split` from `sklearn.model_selection`.
3. A logistic regression model was initialized and trained on the training data using `LogisticRegression` from `sklearn.linear_model`.

## Model Prediction

1. The model was used to predict the survival status of passengers in the test set.
2. The predicted results were printed using `log.predict(X_test)`.
3. The actual target values in the test set were printed using `Y_test`.

## Accuracy 

 1. The accuracy score is checked with the help of `accuracy_score(Y_test, prediction)`.
 2. The model turns out to be 100% accurate.

## Evaluation

1. We  had checked the  prediction by using classificaqtion report `classification_report`.
2. We had used `print(classification_report(Y_test,prediction))` to measure the quality of prediction.
