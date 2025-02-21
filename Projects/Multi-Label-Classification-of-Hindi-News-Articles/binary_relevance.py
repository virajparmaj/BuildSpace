import pandas as pd
from function import punctuations, score, load_stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression


# Load stopwords from a csv file and store them in a list
stopwords = load_stopwords("stopwords.csv")

# add punctuations in the list of stopwords
stopwords.extend(punctuations)

# get dataframe for the dataset
try:
    dataset = pd.read_excel("preprocessed_data_new.xlsx")
except FileNotFoundError as e:
    raise Exception(e)

# get column names(17 out of which latter 14 are class labels)
cols = dataset.columns.ravel()

# reading features file
label_features = []
for i in range(3, 17):
    sheet = cols[i]
    feature = pd.read_excel("output_new.xlsx", sheet_name=sheet)
    label_features.append(feature)

# alternate creation of dataframe(with 31 columns)
columns = []
for i in range(3, 17):
    col_name = "score_" + cols[i]
    columns.append(col_name)
dataset[columns] = dataset['desc'].apply(score, label_features = label_features, cols = cols, stopwords = stopwords).apply(pd.Series)
#print(dataset.columns.ravel())
#print(dataset.head())
df = dataset

# split dataset into training and testing modules
# Split the dataframe into two separate dataframes
label_df = df.iloc[:, :17]
tfidf_df = df.iloc[:, 17:]

# Split the data into a training set and a test set
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_class_counts = train_df.iloc[:, 3:17].sum()
test_class_counts = test_df.iloc[:, 3:17].sum()

print("Train class counts:\n", train_class_counts)
print("Test class counts:\n", test_class_counts)

# Convert the label columns into binary indicator variables
binarizer = MultiLabelBinarizer()
y_train = binarizer.fit_transform(train_df.iloc[:, 3:17].values)
y_test = binarizer.transform(test_df.iloc[:, 3:17].values)

# Use the pre-calculated tf-idf scores as input features for your model
X_train_tfidf = train_df.iloc[:, 17:].values
X_test_tfidf = test_df.iloc[:, 17:].values

# Apply binary relevance by training a separate binary classifier for each label column
classifiers = []
for i in range(14):
    clf = LogisticRegression()
    clf.fit(X_train_tfidf, y_train[:, i])
    classifiers.append(clf)

# Evaluate the performance of the model on the test set
y_pred = []
for clf in classifiers:
    y_pred.append(clf.predict(X_test_tfidf))
y_pred = np.array(y_pred).T
y_pred = binarizer.inverse_transform(y_pred)
y_true = binarizer.inverse_transform(y_test)

# Compute the accuracy of the model
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)