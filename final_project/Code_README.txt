The python code provided will perform the following:

First the code will load and create the training and test dataset
by loading image files from the local path specified. A histogram
is created displaying the data distribution of images among the 
different classes of the dataset. Next, the code will create a 
matrix of sampled images from the datasets to visualize what the 
image data looks like. 

Next, the baseline model is created and stored in alt_model. Another model
two_dense_model is create by modifying the baseline structure as an attempt
to try and improve the baseline performance. Similarly additional models
with dropout regularization (dropout_model) and rescaling (rescaling_model)
are created and trained so differences or improvements in performance can be 
evaluated. 

5-fold CV is performed by creating 5 additional model based on the baseline model.
The entire dataset without splitting is sharded then rejoined to create the different
holdout groups and training dataset to perform CV.

Lastly, all the plots are generated:
* A plot to compare baseline performance against 5-fold CV to check for overfitting
* A plot to compare baseline performance against regularization and different model structures for
accuracy, loss, and validation accuracy across epochs. 