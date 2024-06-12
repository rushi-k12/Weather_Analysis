# Weather Analysis Project
This project involves collecting weather data for Indian cities, preprocessing the data, building a machine learning model to predict temperatures, and deploying the model on Hugging Face.

## Table of Contents
Overview
Data Collection
Data Cleaning
Model Building
Model Deployment
Usage
Requirements
Contributing
License
## Overview
This project aims to analyze weather data for various Indian cities and predict temperatures using a neural network model. The project workflow includes web scraping, data preprocessing, model building, and deployment.

## Data Collection
Weather data for Indian cities was collected from timeanddate.com using the Scrapy framework. The collected data is stored in a file named result.csv.

## Data Cleaning
The raw data in result.csv was cleaned and preprocessed to remove inconsistencies and prepare it for model training. The cleaned data is saved in a file named preprocessed_result.csv.

## Model Building
An Artificial Neural Network (ANN) was built using TensorFlow to predict temperatures based on the cleaned data. The trained model is saved as temperature_prediction_model.h5.

## Model Deployment
The trained model was deployed on Hugging Face and is accessible at the following link:
Hugging Face Deployment

## Usage
To use the deployed model, follow these steps:

Visit the Hugging Face space.
Enter the name of an Indian city.
Get the predicted temperature for the city.
## Requirements
The following Python packages are required to run the project:

pandas
numpy
tensorflow
scrapy
streamlit
Install the required packages using the following command:

sh
Copy code
pip install -r requirements.txt
## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License
This project is licensed under the MIT License.
