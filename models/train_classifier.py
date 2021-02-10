# import libraries
from sqlalchemy import create_engine
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import numpy as np
import nltk
import pickle
import sys
nltk.download(['punkt', 'stopwords', 'wordnet'])


def load_data(database_filepath):
    """
    load data from the sqlite database
    
    input - database_filepath: the path of sqlite file to load
    output - X: messages
             Y: categories
             category_names: the name of the categories
    """
    # Read the table as pandas dataframe
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages', con=engine)

    # Split the dataframe into x and y
    X = df['message']
    Y = df.drop(['message', 'original', 'genre', 'id'], axis=1)

    # Get the label names
    category_names = Y.columns

    return X, Y, category_names

def tokenize(text):
    """
    Tokenize and lemmatize each word in a given text
    
    input - text: text message to be tokenized
    output - clean_tokens: cleaned token
    """
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize to words
    tokens = word_tokenize(text)
    # Remove stop words
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        # Strip and Lemmatize
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    """
    Create a machine learning pipeline
    
    pipeline - machine learning pipeline created
    """
    # Create a pipeline consists of count vectorizer -> RandomForest()
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('mul', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    ## Find the optimal model using GridSearchCV
    parameters = {
        'vect__max_df': (0.5, 0.75),
        'mul__estimator__n_estimators': [50, 100],
    }

    pipeline = GridSearchCV(pipeline, param_grid=parameters, verbose=5, cv=2)
    
    return pipeline

def overal_accuracy(y_true, y_pred):
    """
    Display the accuracy of the test set
    
    input - y_true: y_test
            y_pred: predict with X_test
    """
    accuracy = ((y_true - y_pred) == 0).sum().sum() / (y_true.shape[0] * y_true.shape[1])
    print("The accuracy of the test set is: {}".format(accuracy))
    
    
def evaluate_model(Y_test, Y_pred, category_names):
    """
    Display the classification report for the given model
    
    input - model: the machine learning model
            Y_test: X test set
            Y_pred: Y test set
            category_names: names of the categories
    """

    # Predict the given X_test and create the report based on the Y_pred
    for i, y in enumerate(category_names):
        print(category_names[i])
        print(classification_report(np.array(Y_test)[i], Y_pred[i]))
        print()


def save_model(model, model_filepath):
    """Save the given model into pickle object"""

    # Save the model based on model_filepath given
    pkl_filename = '{}'.format(model_filepath)
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
        
        print('Pridicting test set...')
        Y_pred = model.predict(X_test)
        
        print('Evaluating model...')
        evaluate_model(Y_test, Y_pred, category_names)


    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()