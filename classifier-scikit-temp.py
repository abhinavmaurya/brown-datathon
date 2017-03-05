import argparse
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2,f_regression
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
#import pydot
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

# Purpose: Classifier class to use the data to train and create a prediction model.
class BookingPredictor():

    #Purpose: Intialize the class varaibles and read data from CSV

    def __init__(self,file_name):
        print("init")
        self.df = pd.read_csv(file_name)
        # To consider only first encounter of a patient : Not sure if rest encounters to be ignored
        #self.df = self.df.groupby('patient_nbr').first().reset_index()
        self.df_bucket = None
        self.df_oversampled=None
        self.nan_columns = []

    # Purpose: PreProcess phase
    # Convert label to required form
    # Drop irrelevant columns based on human expertize of the data set
    # Impute missing values
    # Feature Selection
    def preprocess(self):
        #self.preprocess_range_col('age')
        #self.preprocess_range_col('weight')
        #self.df.replace(to_replace={'readmitted':{'<30':'Yes','>30':'No'}},inplace=True)
        #self.df['readmitted'] = self.df['readmitted'].str.lower()
        self.preprocess_others()
        self.drop_irrelevant_cols()
        #self.normalize()
        #self.impute_most_frequent()
        self.select_k_best()

    #Purpose: Uses the Mode to replace all the missing values
    def impute_most_frequent(self):
        imputer = Imputer(missing_values='NA', strategy='most_frequent', axis=0)
        imputer.fit(self.df)
        self.df = pd.DataFrame.from_records(imputer.transform(self.df),columns=self.df.columns)

	#Purpose: Uses the Median to replace all the missing values
    def impute_missing(self):
        imputer = Imputer(strategy = "median")
        df_y = self.df['readmitted']
        df_imputed = imputer.fit_transform(self.df[self.df.columns[:-1]],df_y)
        self.df = pd.DataFrame.from_records(df_imputed,columns=self.df.columns[:-1])
        self.df['readmitted'] = df_y


    #Purpose: These features have more than 50% missing values and hence it was
    #decided to remove then.
    def drop_irrelevant_cols(self):
        for col in ['user_id','p_AddToCart','daysToCheckin', 'p_TotalPrice']:
            self.df.drop(col, axis=1, inplace=True)

    #Purpose: Feature selection is done using F_regression
    def select_k_best(self):
        df_dataset_y = self.df['BookingPurchase']
        df_dataset = self.df[self.df.columns[:-1]]
        sel_k = int(round(self.df.shape[1]*0.60))
        f_reg = SelectKBest(f_regression, k=sel_k)
        df_dataset = f_reg.fit_transform(df_dataset,df_dataset_y)
        selected_cols = np.asarray(self.df.columns[:-1])[f_reg.get_support()]
        print(selected_cols)
        selected_cols = np.append(selected_cols,'BookingPurchase')
        #self.df = self.df[selected_cols]

    #Purpose: Normalize the data using Z-Score method.
    def normalize(self):
        for col in self.df.columns[:-1]:
            if self.df[col].std(ddof=1) == 0:
                self.df[col] = (self.df[col] - self.df[col].mean())
            else:
                self.df[col] = (self.df[col] - self.df[col].mean())/self.df[col].std(ddof=1)

    #Purpose: Convert the string representation values into categorical buckets
    def replace_dayToPrevVisit(self,value):
        try:
            value = float(value)
            if (value >= 0 and value <= 2):
                return 0
            if (value >= 3 and value <= 21):
                return 1
            if (value >= 22 and value <= 43):
                return 2
            if (value >= 44):
                return 3
            if (value=="NA"):
                return 4
        except ValueError:
            return 4

    def replace_day(self,value):
        try:
            x = value.split("-")
            #print(x[1])
            return x[1]
        except ValueError:
            return '13'


    def replace_sessionDuration(self,value):
        try:
            value = float(value)
            if (value >= 0 and value <= 50):
                return 0
            if (value >= 50):
                return 1
            if (value=="NA"):
                return 2
        except ValueError:
            return 2

    def replace_pageViews(self,value):
        try:
            value = float(value)
            if (value == 0):
                return 0
            if (value == 1):
                return 1
            if (value >= 2 and value <= 3):
                return 3
            if (value >= 4 and value <= 5):
                return 4
            if (value >= 6 and value <= 9):
                return 5
            if (value >= 10 and value <= 17):
                return 6
            if (value >= 18 and value <= 29):
                return 7
            if (value >= 30 and value <= 52):
                return 8
            if (value >= 53):
                return 9
            if (value=="NA"):
                return 10
        except ValueError:
            return 10
 
    # Purpose: Make sure every column is formatted to a numerical meaningful value.
    # ? in label are re labled
    # Cateorical string values are num encoded.
    def preprocess_others(self):
        le = LabelEncoder()
        classes = {}
        string_cols = ['p_trafficChannel','osTypeName']
        for col in self.df.columns.values:
            if col == 'daysFromPreviousVisit':
                self.df[col] = self.df[col].apply(self.replace_dayToPrevVisit)
            elif col == 'p_pageViews':
                self.df[col] = self.df[col].apply(self.replace_pageViews)
            elif col == 'day':
                self.df[col] = self.df[col].apply(self.replace_day)
            elif col in string_cols:
                le.fit(self.df[col].values)
                classes[col] = list(le.classes_)
                self.df[col] = le.transform(self.df[col])

    #Purpose: Convert range columns to mean
    def preprocess_range_col(self,col):
        self.df[col] = self.df[col].str.extract('\[(\d+)\-(\d+)\)',expand=True).astype(int).mean(axis=1)

    #Purpose: Convert ? in to Nan which can be processed later.
    def set_nan_cols(self):
        for col in self.df.columns:
            temp = self.df[col].unique()
            if 'NA' in temp:
                self.nan_columns.append(col)

        for col in self.nan_columns:
            print("Percentage of NA in col {} = {}".format(col,(self.df.groupby(by=[col]).size()*100/len(self.df))['NA']))

    #Purpose: Create
    def create_df_bucket(self):
        df_grp = self.df.groupby(by=['BookingPurchase'])
        df_grp_yes = df_grp.get_group(1)
        df_grp_no = df_grp.get_group(0)
        df_grp_no = shuffle(df_grp_no)
        self.df_bucket = np.array_split(df_grp_no,4)
        for ind in range(0,len(self.df_bucket)):
            self.df_bucket[ind] = pd.concat([self.df_bucket[ind],df_grp_yes])

    #Purpose: Read data from disk
    def spool_df(self):
        self.df_oversampled.to_csv("converted_data.csv",index=False)

    #Purpose: Oversampling implemented to balance the data.
    def oversampling(self):
        purchased_yes = self.df['BookingPurchase'] == 1
        df_yes = self.df[purchased_yes]
        self.df_oversampled = self.df.append([df_yes] * 3, ignore_index=True)


    #Purpose: Implements the decision tree classifier
    def run_decision_tree(self):
        Y = self.df_oversampled["BookingPurchase"]
        X = self.df_oversampled[self.df_oversampled.columns[:-1]]
        dt = DecisionTreeClassifier(random_state=0)
        dt.fit(X, Y)
        scores = cross_val_score(dt, X, Y, cv=10)
        print("Decision Tree Score for bucket is: {}".format(scores))
        #self.visualize_tree(dt, self.df.columns[:-1])

    def run_svm(self):
        Y = self.df_oversampled["BookingPurchase"]
        X = self.df_oversampled[self.df_oversampled.columns[:-1]]
        dt = LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, loss='squared_hinge', max_iter=1000, multi_class='ovr',penalty='l1', random_state=None, tol=0.001, verbose=0)
        dt.fit(X, Y)
        scores = cross_val_score(dt, X, Y, cv=10)
        print("SVM Score for bucket is: {}".format(scores))
        #self.visualize_tree(dt, self.df.columns[:-1])

    #Purpose: Implement voting classifier
    def run_voting_classifier(self):
        clf1 = DecisionTreeClassifier(random_state=0)
        #clf2 = LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, loss='squared_hinge', max_iter=1000, multi_class='ovr',penalty='l2', random_state=None, tol=0.001, verbose=0)
        clf2 = SVC()
        clf3 = RandomForestClassifier(n_estimators = 25)
        eclf = VotingClassifier(estimators=[('dr', clf1), ('lsvm', clf2), ('rf', clf3)], voting='hard')

        train, test = train_test_split(self.df_oversampled, test_size = 0.2)
        yTrain = train["BookingPurchase"]
        xTrain = train[train.columns[:-1]]

        yTest = test["BookingPurchase"]
        xTest = test[train.columns[:-1]]

        #self.spool_df()

        # eclf.fit(xTrain,yTrain)
        #for clf, label in zip([clf1, clf2, clf3, eclf], ['Decision Tree', 'Linear SVM', 'Random Forest', 'Ensemble']):
        for clf, label in zip([clf1, clf2, eclf], ['Decision Tree', 'SVMrbf']):
            clf.fit(xTrain,yTrain)
            trainScores = cross_val_score(clf, xTrain, yTrain, cv=5, scoring='accuracy')
            y_trainPred = clf.predict(xTrain)
            trainAccuracy = accuracy_score(yTrain, y_trainPred)
            print("Train Accuracy: %0.2f [%s]" % (trainAccuracy, label))
            print("Train Accuracy (5 Fold CV): %0.2f (+/- %0.2f) [%s]" % (trainScores.mean(), trainScores.std(), label))
            y_pred = clf.predict(xTest)
            testScores = accuracy_score(yTest, y_pred)
            confusionMatrix = confusion_matrix(yTest, y_pred)
            print(confusionMatrix)
            joblib.dump(clf, label + '.pkl')
            # testScores = cross_val_score(clf, xTest, yTest, cv=5, scoring='accuracy')
            print("Test Accuracy: %0.2f [%s]" % (testScores, label))


    def run_voting_classifier_on_bucket(self):

        # eclf.fit(xTrain,yTrain)
        #for clf, label in zip([clf1, clf2, clf3, eclf], ['Decision Tree', 'Linear SVM', 'Random Forest', 'Ensemble']):
        for ind in range(0, len(self.df_bucket)):
            clf = DecisionTreeClassifier(random_state=0)
            train, test = train_test_split(self.df_bucket[ind], test_size = 0.2)
            yTrain = train["BookingPurchase"]
            xTrain = train[train.columns[:-1]]

            yTest = test["BookingPurchase"]
            xTest = test[train.columns[:-1]]
            clf.fit(xTrain,yTrain)
            trainScores = cross_val_score(clf, xTrain, yTrain, cv=5, scoring='accuracy')
            y_trainPred = clf.predict(xTrain)
            trainAccuracy = accuracy_score(yTrain, y_trainPred)
            print("Train Accuracy: %0.2f [%s]" % (trainAccuracy, ind))
            print("Train Accuracy (5 Fold CV): %0.2f (+/- %0.2f) [%s]" % (trainScores.mean(), trainScores.std(), ind))
            y_pred = clf.predict(xTest)
            testScores = accuracy_score(yTest, y_pred)
            # testScores = cross_val_score(clf, xTest, yTest, cv=5, scoring='accuracy')
            print("Test Accuracy: %0.2f [%s]" % (testScores, ind))




def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("-f","--file_name",type=str,help="Data file name")
    args=parser.parse_args()
    file_name = args.file_name
    predictor = BookingPredictor(file_name)
    predictor.set_nan_cols()
    predictor.preprocess()
    predictor.oversampling()

    #predictor.create_df_bucket()

    predictor.run_voting_classifier()
    #predictor.run_voting_classifier_on_bucket()
if __name__ == "__main__":
    main()