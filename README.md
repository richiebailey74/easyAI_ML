# easyAI_ML

There is NEVER free lunch, but we made a way to get reduced lunch...

The entire purpose of this repository is for the supporting machine learning functionality of the EasyAI web application.

EasyAI exists to solve a problem of a lack of access to machine learning in research. Many research teams could benefit from the power of deep learning models, but either lack the knowledge to create the models themselves, or lack the resources to pay for someone to build them models.

The supporting machine learning logic in this repository exposes users to use machine learning with very minimal machine learning knowledge needed to do so. Through the ease of uploading one's dataset with an intuitive user interface, it broadens access while simultaneously lowering the barrier of access.

From the data and only a few user-given points about the data, we utilize a generalized preprocessor (serialized to the cloud) as well as dynamically adjust neural network architecture hyperparameters to give the user a model tuned specifically for their data. The user can load in their models at any time to perform predictive functionality. All business logic (training and predicting) is performed on an AWS EC2 instance.
