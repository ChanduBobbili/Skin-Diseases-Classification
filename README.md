# Skin-Cancer-Classification

This is a flask application for a skin cancer classification project .
The dataset used is HAM10000. There are 7 different classes in this dataset.
Model used in the application is MobileNet. It has been trained by freezing the base layers and only updating the custom layers on the top.
Upon removing the top layer of MobileNet , a dropout layer of o.5 threshold has been sandwiched between a pair of dense layers.
The model outputs the class label which has the highest probablility.
