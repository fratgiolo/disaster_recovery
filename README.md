# Disaster Response Pipeline Project

### Table of Contents

1. [Prerequisites and Installation](#Prerequisites)
2. [Motivation](#motivation)
3. [File Description](#files)
4. [Acknowledgments](#licensing)

## Prerequisites and Installation <a name="Prerequisites"></a>

Prerequisites:

The following libraries are required in order to run the project: Pandas, Nltk, Sqlalchemy, Flask, Plotly, NumPy, SciPy and Sciki-Learn

Installation:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Motivation <a name="motivation"></a>

This project was made for an an Udacity Nanodegree project. The study focused on data from real messages that were sent during disaster events. Our aim was to create a machine learning pipeline to categorize these events so that they can be sent to an appropriate disaster relief agency.

## File Description <a name="files"></a>

**data/process_data.py** : Python code containing the ETL process for this project </br>
**data/train_classifier.py** : Python code running the Machine Learning Pipeline </br>
**data/disaster_categories.csv**: CSV containing message categories </br>
**data/disaster_messages.csv**: CSV containing the disaster messages </br>

## Acknowledgements<a name="licensing"></a>
- Figure Eight for providing the input files used for this project
- Udacity for running the nanodegree course in Data Science
