# Predictive Health Analytics for Diabetic Risk Assessment and Personalized Reporting WebApp using Streamlit

## Table of Contents
- [Overview](#overview)
- [Aim](#aim)
- [Mission](#mission)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Run](#run)
- [Bug / Feature Request](#bug--feature-request)
- [Team](#team)
- [License](#license)
- [Credits](#credits)

## Overview
This project aims to predict the risk of diabetes in individuals based on various features such as pregnancies, insulin level, age, and BMI. The dataset used for this project is sourced from Kaggle, originally provided by the National Institute of Diabetes and Digestive and Kidney Diseases.

## Aim
To develop a predictive health analytics tool for assessing diabetic risk and providing personalized reports.

## Mission
To leverage machine learning for early detection of diabetes, enabling timely medical intervention and improving health outcomes.

## Demo
[Link to Demo Video/Website]

## Learning Objective
- Understand the end-to-end process of developing a machine learning model.
- Gain experience in deploying applications on cloud platforms like Heroku.
- Learn to build interactive web applications using Flask.

## Technical Aspect
- Training a machine learning model using scikit-learn.
- Building and hosting a Flask web app on Heroku.
- User input for features such as pregnancies, insulin level, age, BMI, etc., followed by a prediction display.

## Technologies Used
- Python
- scikit-learn
- Flask
- Heroku

## To Do
- Improve model accuracy
- Enhance the web application's user interface
- Add more features to the predictive model

## Installation
1. Clone this repository and unzip it.
2. Navigate into the project directory.
    ```bash
    cd predictive-health-analytics
    ```
3. Create a virtual environment with Python 3 and activate it.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
4. Install the required packages.
    ```bash
    pip install -r requirements.txt
    ```

## Run
Execute the following command to start the application:
```bash
python app.py
