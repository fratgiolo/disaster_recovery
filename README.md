# Disaster Response Pipeline Project

### Table of Contents

1. [Prerequisites and Installation](#Prerequisites)
2. [Motivation](#motivation)
3. [File Description](#files)
4. [Conclusion](#conclusion)
5. [Detailed Results](#results)
6. [Acknowledgments](#licensing)

## Prerequisites and Installation <a name="Prerequisites"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Motivation <a name="motivation"></a>

This project was made for an an Udacity Nanodegree project real messages that were sent during disaster events. You will be creating a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

## File Description <a name="files"></a>

**project.ipynb**: Python code containing my analysis. </br>
**input/2018/survey_results_public.zip**: Zip containing Stackoverflow's 2018 Annual Developer Survey data. </br>
**input/2019/survey_results_public.zip**: Zip containing Stackoverflow's 2019 Annual Developer Survey data. </br>

## Conclusion <a name="conclusion"></a>
In brief, the answers to the above question have been:
* America and India are the easiest countries to find a job as a developer
* Freelancer and part-time jobs are extremely unpopular in the IT sector and we are most likely to find a job as a full-time employee.
* While older programming languages like JavaScript, HTML and CSS are still dominating the market, relatively younger ones ​​like Python have been steadily emerging.

## Detailed Results <a name="results"></a>
If you would like to read more details around how I've reached those conclusions, the results of my analysis are discussed [here](https://medium.com/@marco.altamura88/what-are-the-countries-with-the-highest-percentage-of-employed-programmers-b65b29ed9be4)

## Acknowledgements<a name="licensing"></a>
This project makes use of the following external resources:

Stackoverflow’s 2018 and 2019 Developer Surveys which can be found [here](https://insights.stackoverflow.com/survey).
