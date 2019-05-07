# Taxi Fare Prediction with Pyspark
This is a Python Flask web application that can help you predict the total cost for your taxi trip, including tips, toll, and other surcharges. Type in your pickup location and your destination, our app can predict the total fare using history taxi and weather data.

Main body of the application is in the application folder and the pyskark mllib model is in the utils folder

This is a trial version with toy data, to get more accuracte predictions, use the original data instead.
## How to use: 
- start backend: python wsgi.py 
- website access: 
  - homepage: http://127.0.0.1:5400/home (get)
  - application page: homepage: http://127.0.0.1:5400/app
## Authors
Tiancheng Yin, Qiuyao Liu, and Guanjia Wang
## Requirements
- Python 3
- wsgi 
- flask
- pyspark
- geopy

