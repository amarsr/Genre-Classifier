# Genre-Classifier
This project tests the predictive capability of various statistical learning methods in classifying music genres. The Python programming language was used with the scikit-learn package, stored in two Jupyter notebooks. The music data is generated from the [FMA song archive](https://github.com/mdeff/fma). I used the 8GB balanced dataset which contains 8000 songs split evenly into 8 genres: Rock, Hip-Hop, International, Folk, Pop, Experimental, Jazz, and Instrumental. 

## Overall framework
The repository contains two main files: preprocessing and genre_classifier. The first file has 4 main functions:
1. Importing the dataset from local files
2. Cleaning the data
3. Selecting the feature set for future analysis
3. Building CSV

This CSV is read into the second file for the statistical learning methods. We test 3 main methods:
1. Support Vector Machines with linear kernels
2. Support Vector Machines with RBF kernels, optimized using GridSearchCV
3. Random forests

## Current progress and future improvements
All models currently receive an accuracy score of 43% - approximately 4x the performance of a random selector. I plan to build a neural network to compare the accuracy rate with the current models. 
