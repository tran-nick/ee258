# handle imports

import matplotlib.pylab as plt
import numpy as np
import pathlib
import tensorflow as tf
from tensorflow import keras


# Setting path name to load images from directory
path = 'C:/Users/tickn/ml/final_project/Fish_Dataset/Fish_Dataset_6'
data_dir = pathlib.Path(path)

image_count = len(list(data_dir.glob('**/*.png')))
print(image_count)

# Loading and splitting data into training and test dataset for baseline model
batch_size = 32
img_height = 445
img_width = 590

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset = "training",
    seed = 123,
    image_size = (img_height , img_width),
    batch_size = batch_size
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset = "validation",
    seed = 123,
    image_size = (img_height , img_width),
    batch_size = batch_size
)

class_names = train_ds.class_names
for name in class_names:
    print(name)


# Create histogram of distribution of image data per class
class_counts = np.zeros(len(class_names))
i = 0

# we load the entire dataset without splitting for k-fold CV
all_data = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size = (img_height , img_width),
    seed=123
)

for all_img, all_labels in all_data:
    for cat in all_labels:
        class_counts[cat]+=1

print(class_counts)
plt.figure(figsize=(15,10))
distribution_data = dict(zip(class_names,class_counts))
plt.bar(list(distribution_data.keys()),distribution_data.values(),width =0.6)
plt.title('All Data')
plt.xlabel('Class Category')
plt.ylabel('Counts')
plt.show()

# Create visualization of sampled dataset
unique = []
i=0
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for item in range(len(labels)):
        if labels[item] in unique:
            continue
        else:
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[item].numpy().astype("uint8"))
            plt.title(class_names[labels[item]])
            plt.axis("off")

            i+=1
            unique.append(labels[item])
        
    break # run only once through a batch



# Create and train baseline model
alt_model = keras.models.Sequential([
    tf.keras.layers.Conv2D(64, 8, strides=2, kernel_regularizer='l2', activation="relu",input_shape=[445,590,3]),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

# compile the basic model

sgd = keras.optimizers.SGD(learning_rate=0.001)
alt_model.compile(loss="sparse_categorical_crossentropy", optimizer=sgd,
              metrics=["accuracy"])
# training baseline
history = alt_model.fit(
                    train_ds,
                    batch_size=32, 
                    epochs=50,
                    validation_data =test_ds,
                    callbacks=[keras.callbacks.EarlyStopping(monitor="accuracy",min_delta=0.005,patience=2)])

alt_model.save('baseline_model_6class.h5')


# Create model with 2 dense layers
two_dense_model = keras.models.Sequential([
    tf.keras.layers.Conv2D(64, 8, strides=2, activation="relu", input_shape=[445,590,3]),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

two_dense_model.compile(loss="sparse_categorical_crossentropy", optimizer=sgd,
                        metrics=["accuracy"])
# training two_dense
history_two_dense_model = two_dense_model.fit(
                                        train_ds,
                                        batch_size=32,
                                        epochs=50,
                                        validation_data = test_ds,
                                        callbacks=[keras.callbacks.EarlyStopping(monitor="accuracy",min_delta=0.005,patience=2)])

# Create model with dropout regularization
dropout_model = keras.models.Sequential([
    tf.keras.layers.Conv2D(64, 8, strides=2, activation="relu", input_shape=[445,590,3]),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

dropout_model.compile(loss="sparse_categorical_crossentropy", optimizer=sgd,
                    metrics=["accuracy"])
# training dropout 
history_dropout = dropout_model.fit(
                                train_ds,
                                batch_size=32,
                                epochs=50,
                                validation_data = test_ds,
                                callbacks=[keras.callbacks.EarlyStopping(monitor="accuracy",min_delta=0.005,patience=2)])


# Create model with rescaling layer, change RGB pixel values between 0 and 1
rescaling_model = keras.models.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(64, 8, strides=2, activation="relu", input_shape=[445,590,3]),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

rescaling_model.compile(loss="sparse_categorical_crossentropy", optimizer=sgd,
                    metrics=["accuracy"])
# training rescaling
history_rescaling = rescaling_model.fit(
                                train_ds,
                                batch_size=32,
                                epochs=50,
                                validation_data = test_ds,
                                callbacks=[keras.callbacks.EarlyStopping(monitor="accuracy",min_delta=0.005,patience=2)])


# Perform 5-fold CV
# function for ease
def create_model():
    alt_model = keras.models.Sequential([
    tf.keras.layers.Conv2D(64, 8, strides=2, kernel_regularizer='l2', activation="relu", input_shape=[445,590,3]),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

    ## compile the basic model

    sgd = keras.optimizers.SGD(learning_rate=0.001)
    alt_model.compile(loss="sparse_categorical_crossentropy", optimizer=sgd,
              metrics=["accuracy"])
    return alt_model

# Create different models for each fold
model1 = create_model()
model2 = create_model()
model3 = create_model()
model4 = create_model()
model5 = create_model()

# Shard the data and rejoin to create training and test dataset for k-fold CV
hold1 = all_data.shard(num_shards=5, index=0)
hold2 = all_data.shard(num_shards=5, index=1)
hold3 = all_data.shard(num_shards=5, index=2)
hold4 = all_data.shard(num_shards=5, index=3)
hold5 = all_data.shard(num_shards=5, index=4)

train1 = hold2.concatenate(hold3)
train1 = train1.concatenate(hold4)
train1 = train1.concatenate(hold5)

train2 = hold3.concatenate(hold4)
train2 = train2.concatenate(hold5)
train2 = train2.concatenate(hold1)

train3 = hold4.concatenate(hold5)
train3 = train3.concatenate(hold1)
train3 = train3.concatenate(hold2)

train4 = hold5.concatenate(hold1)
train4 = train4.concatenate(hold2)
train4 = train4.concatenate(hold3)

train5 = hold1.concatenate(hold2)
train5 = train5.concatenate(hold3)
train5 = train5.concatenate(hold4)

# Start 5-fold CV
history1 = model1.fit(
                    train1,
                    batch_size=32, 
                    epochs=50,
                    validation_data =hold1,
                    callbacks=[keras.callbacks.EarlyStopping(monitor="accuracy",min_delta=0.005,patience=2)])
history2 = model2.fit(
                    train2,
                    batch_size=32, 
                    epochs=50,
                    validation_data =hold2,
                    callbacks=[keras.callbacks.EarlyStopping(monitor="accuracy",min_delta=0.005,patience=2)])
history3 = model3.fit(
                    train3,
                    batch_size=32, 
                    epochs=50,
                    validation_data =hold3,
                    callbacks=[keras.callbacks.EarlyStopping(monitor="accuracy",min_delta=0.005,patience=2)])
history4 = model4.fit(
                    train4,
                    batch_size=32, 
                    epochs=50,
                    validation_data =hold4,
                    callbacks=[keras.callbacks.EarlyStopping(monitor="accuracy",min_delta=0.005,patience=2)])
history5 = model5.fit(
                    train5,
                    batch_size=32, 
                    epochs=50,
                    validation_data =hold5,
                    callbacks=[keras.callbacks.EarlyStopping(monitor="accuracy",min_delta=0.005,patience=2)])


# Generate Plots
# Plot accuracy vs epochs
plt.plot(history.history['accuracy'])
plt.plot(history1.history['accuracy'])
plt.plot(history2.history['accuracy'])
plt.plot(history3.history['accuracy'])
plt.plot(history4.history['accuracy'])
plt.plot(history5.history['accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epochs')
plt.legend(['alt_model','model1','model2','model3','model4','model5'])

# Plot loss vs epochs
plt.plot(history.history['loss'])
plt.plot(history1.history['loss'])
plt.plot(history2.history['loss'])
plt.plot(history3.history['loss'])
plt.plot(history4.history['loss'])
plt.plot(history5.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.legend(['alt_model','model1','model2','model3','model4','model5'])

# Plot validation accuracy vs epoch
plt.plot(history.history['val_accuracy'])
plt.plot(history1.history['val_accuracy'])
plt.plot(history2.history['val_accuracy'])
plt.plot(history3.history['val_accuracy'])
plt.plot(history4.history['val_accuracy'])
plt.plot(history5.history['val_accuracy'])
plt.xlabel('Epochs')
plt.ylabel('val_accuracy')
plt.title('Val_accuracy vs Epochs')
plt.legend(['alt_model','model1','model2','model3','model4','model5'])

# Plot different model and regularization performance accuracy vs epochs
plt.plot(history.history['accuracy'])
plt.plot(history_two_dense_model.history['accuracy'])
plt.plot(history_dropout.history['accuracy'])
plt.plot(history_rescaling.history['accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epochs')
plt.legend(['alt_model','two_dense','dropout','rescaling'])

# Plot different model and regularization performance loss vs epochs
plt.plot(history.history['loss'])
plt.plot(history_two_dense_model.history['loss'])
plt.plot(history_dropout.history['loss'])
plt.plot(history_rescaling.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.legend(['alt_model','two_dense','dropout','rescaling'])

# Plot different model and regularization performance validation accuracy vs epochs
plt.plot(history.history['val_accuracy'])
plt.plot(history_two_dense_model.history['val_accuracy'])
plt.plot(history_dropout.history['val_accuracy'])
plt.plot(history_rescaling.history['val_accuracy'])
plt.xlabel('Epochs')
plt.ylabel('val_accuracy')
plt.title('Val_accuracy vs Epochs')
plt.legend(['alt_model','two_dense','dropout','rescaling'])