import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

# Purpose: Classifier class to use the data to train and create a prediction model.
class BookingPredictor():

    #Purpose: Intialize the class varaibles and read data from CSV

    def __init__(self,file_name):
        print("init")
        self.df = pd.read_csv(file_name)
        self.nan_columns = []
        self.clf = joblib.load("model.pkl")
        self.set_nan_cols()
        self.preprocess()


    # Purpose: PreProcess phase
    # Convert label to required form
    # Drop irrelevant columns based on human expertize of the data set
    # Impute missing values
    # Feature Selection
    def preprocess(self):
        self.preprocess_others()
        self.drop_irrelevant_cols()


    #Purpose: These features have more than 50% missing values and hence it was
    #decided to remove then.
    def drop_irrelevant_cols(self):
        for col in ['user_id','p_AddToCart','daysToCheckin', 'p_TotalPrice']:
            self.df.drop(col, axis=1, inplace=True)


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

    #Purpose: Convert ? in to Nan which can be processed later.
    def set_nan_cols(self):
        for col in self.df.columns:
            temp = self.df[col].unique()
            if 'NA' in temp:
                self.nan_columns.append(col)

        for col in self.nan_columns:
            print("Percentage of NA in col {} = {}".format(col,(self.df.groupby(by=[col]).size()*100/len(self.df))['NA']))


    #Purpose: Implement voting classifier
    def run_voting_classifier(self):
        test = self.df

        yTest = test["BookingPurchase"]
        xTest = test[test.columns[:-1]]

        y_pred = self.clf.predict(xTest)
        testScores = accuracy_score(yTest, y_pred)
        confusionMatrix = confusion_matrix(yTest, y_pred)
        print(confusionMatrix)
        return ("Test Accuracy: %0.2f" % (testScores))

