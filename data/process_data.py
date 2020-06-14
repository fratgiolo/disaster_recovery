import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load Data function
    Description: Loads and merges input datasets by id. Output is saved to a pandas DataFrame
    
    Inputs:
        messages_filepath : message file path
        categories_filepath -> categories file path
    Output:
        df : dataset containing merge of categories and messages by id
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')
    return df 


def clean_data(df):
    """
    Clean Data function
    Description: Cleaning input dataset
    
    Inputs:
        df: raw dataset
    Output:
        df : clean dataset
    """
        
    categories = df.categories.str.split(";",expand=True)
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x:x[:-2])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(np.int)
    df=df.drop("categories",axis=1)
    df = pd.concat([df,categories],axis=1)
    df=df.drop_duplicates()
    return df

def save_data(df, database_filename):
    """
    Save Data function
    Description: saves clean dataset to specific SQLite database
    
    Inputs:
        df: clean input dataset
         database_filename : destination database
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('df', engine, index=False)


def main():
    """
    Main Function
    Description: Takes care of the entire ETL pipeline by loading data from csv, cleaning and uploading to SQlite
    """
        
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()