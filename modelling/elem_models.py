import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier


people = ['gdhande', 'Munesh', 'nivedita', 'Shweta', 'Swapnanil', 'Sharath']
dfs = []

for i,person in enumerate(people):
    df = pd.read_csv(f'data/s{i+1}.csv')
    df = df.rename(columns={'tweet': 'feature', f'annotations.{person}.supernarrative_1' : 'label'})
    df['label'] = df['label'].fillna('None')
    df['label'] = pd.factorize(df['label'])[0]
    dfs.append(df)

df = pd.concat(dfs,ignore_index=False)
# split the data
X_train = df['feature']
y_train = df['feature']


df_test = pd.read_csv('data/test.csv',encoding='latin1')
X_test = df_test['original_tweet']
y_test = df_test['group_annotator_supernarrative_1']


X_train, X_test, y_train, y_test = train_test_split(df['feature'], df['label'], test_size=0.2, random_state=23788)

# Define a dictionary of models
models = {
    'Multinomial Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Linear SVM': LinearSVC(dual=False),
    'Multilayer Perceptron': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000),
    'Gradient Boosting': GradientBoostingClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Bagging Classifier': BaggingClassifier(),
    'XGBoost': XGBClassifier(),
    'Voting Classifier': VotingClassifier(estimators=[('lr', LogisticRegression()), ('rf', RandomForestClassifier()), ('svc', LinearSVC())], voting='hard'),
    'Stacking Classifier': StackingClassifier(estimators=[('lr', LogisticRegression()), ('rf', RandomForestClassifier()), ('svc', LinearSVC())], final_estimator=LogisticRegression())
}

accuracy_list = []
f1_list = []
precision_list = []
recall_list = []

# Iterate over models
for name, model in models.items():

    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', model)
    ])
    
    # Fit the model
    text_clf.fit(X_train, y_train)
    
    # Predict labels
    predicted = text_clf.predict(X_test)
    
    # Calculate F1 score
    f1 = f1_score(y_test, predicted, average='weighted')
    f1_list.append(f1)

    # Calculate precision
    precision = precision_score(y_test, predicted, average='weighted')
    precision_list.append(precision)

    # Calculate recall
    recall = recall_score(y_test, predicted, average='weighted')
    recall_list.append(recall)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predicted)
    accuracy_list.append(accuracy)
    
    # Print model name and accuracy
    print(f'{name} Accuracy: {accuracy}')
    print(f'{name} F1: {f1}')
    print(f'{name} Precision: {precision}')
    print(f'{name} Recall: {recall}\n\n')