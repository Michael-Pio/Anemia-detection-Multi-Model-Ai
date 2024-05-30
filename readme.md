# Anemia Detection Project

This project aims to develop a multi-model approach for anemia detection using various data sources, including symptoms, eye conjunctiva images, palm images, and fingernail images.

## Project Structure

The project is organized into the following directories:

- **Model**: Contains the trained machine learning models for each data source.
- **web**: Contains the web application for user interaction and prediction.
- **data**: Contains the datasets used for training and testing the models.
- **utils**: Contains utility functions for data preprocessing and image manipulation.

## Web Application

The web application allows users to:

- Select symptoms from a list.
- Upload images of their eye conjunctiva, palm, and fingernail.
- Submit the data for prediction.
- View the prediction results and learn more about anemia.

## Models

The project uses four different machine learning models:

- **Symptoms Model**: A logistic regression model trained on a dataset of patient symptoms and anemia diagnoses.
- **Conjunctiva Model**: A convolutional neural network (CNN) trained on a dataset of eye conjunctiva images and anemia diagnoses.
- **Palm Model**: A CNN trained on a dataset of palm images and anemia diagnoses.
- **Fingernail Model**: A CNN trained on a dataset of fingernail images and anemia diagnoses.

for Model Download link : [Models Download Link](https://drive.google.com/drive/folders/1oCFiai2aM_tjYpk2_nmIpHbeB2Ud5p4y?usp=sharing)

## Prediction

The web application combines the predictions from all four models to provide a final prediction of anemia. The final prediction is based on a weighted average of the individual model predictions.

## More Information

For more information about anemia and the models used in this project, please visit the "Learn More" page on the web application.

## Future Work

- Improve the accuracy of the models by training on larger datasets.
- Develop a mobile application for easier access to the prediction tool.
- Integrate the prediction tool with electronic health records systems.

## Disclaimer

This project is for educational purposes only and should not be used for medical diagnosis. Please consult a healthcare professional for any health concerns.

## Additional Files

### index.html

This file contains the HTML code for the web application. It includes the user interface elements, such as the symptom selection form, image upload buttons, and the prediction result display.

### app.py

This file contains the Python code for the web application. It handles user input, processes the uploaded images, performs predictions using the trained models, and returns the results to the user.

## Conclusion

This project demonstrates the potential of using a multi-model approach for anemia detection. By combining different data sources and machine learning models, the project aims to provide a more accurate and reliable prediction of anemia.
