import sys
# import libraries
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
import re
import nltk
from sklearn.metrics import confusion_matrix
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.stats import hmean
from scipy.stats.mstats import gmean
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report
from scipy.stats import hmean
from scipy.stats.mstats import gmean

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])



def load_data(database_filepath):
    '''
    Returns message dataset, category dataset and category list
    Arguments:
        database_filepath (str): database filepath
    Input:
        database_filepath : database name
    Output:
        X= Dataset containing messages
        Y= Dataset containing categories
        category_names= list of category names
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('df',engine)
    X = df['message']
    y = df.iloc[:,4:]
    category_names = list(df.columns[4:])
    return X, y, category_names

def tokenize(text):
    '''
    Cleans text
    Arguments:
        text (str): input text messages to clean
    Input:
        text= input text document
    Output:
        clean_tokens= cleaned text
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

#Class StartVerbExtractor added to use it as a feature in function build_model
class StartVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model():
    '''
    Builds the model by calling a pipeline which sequentially applies a list of transforms and a final estimator
    Arguments:
        none
    Input:
        none
    Output:
        pipeline= model pipeline
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('first_verb', StartVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    ''''
    Evaluates model's precision and prints to screen the results
    Arguments:
        none
    Input:
        none
    Output:
        none
    '''
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test.iloc[:, 1:].values, np.array([x[1:] for x in Y_pred]), target_names = category_names))


def save_model(model, model_filepath):
    '''
    Saves the model in a pickle file
    Arguments:
        model : the input model
        model_filepath: destination model file path
    Input:
        none
    Output:
        none
    '''
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    '''
    This is the main function which calls other functions in this code to Extract data from 
    Sql database, Trains the model, evaluates the output and finally saves it to a pickle file
    Arguments:
        none
    Input:
        none
    Output:
        none
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()