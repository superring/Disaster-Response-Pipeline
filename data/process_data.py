from sqlalchemy import create_engine
import pandas as pd
import sys

def load_data(messages_filepath, categories_filepath):
    """
    Read messages and categories data and merge it into one dataframe
    
    input - message_filepath: the filepath of message file
            categories_filepath: the filepath of category file
    output - df: the merged data file
    """
    # Read files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge files
    df = pd.merge(messages, categories, how='left', on='id')
    
    return df


def clean_data(df):
    """
    Clean the merged dataframe to make it ready to analyze
    
    input - df: the dataframe to be cleaned
    output - df: the cleaned dataframe
    """
    # create a dataframe of the multiple individual category columns
    categories = df.categories.str.split(";",expand=True)
   
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    # rename the columns
    category_colnames = pd.Series(row).apply(lambda x: x[0:-2])
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split("-").str.get(1)
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    Take the input dataframe and save it into sqlite database
    input - df: the dataframe to save
            database_filename: the file path of the database file

    """
    # Create sqlite engine and save the dataframe with the name messages
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages', engine, index=False) 


def main():
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