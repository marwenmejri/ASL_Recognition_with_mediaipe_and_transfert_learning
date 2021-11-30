from helper_functions import *
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import zipfile

# zip_ref = zipfile.ZipFile("American Sign Langage Alphabet.zip")
# zip_ref.extractall()
# zip_ref.close()
#
# Walk through  data directory and list number of files
walk_through_dir(dir_path="American Sign Langage Alphabet")

# Create training and testing directories
test_dir = "American Sign Langage Alphabet/Test"
train_dir = "American Sign Langage Alphabet/Train"

#Create data inputs
img_size = (224, 224)
train_data = tf.keras.preprocessing.image_dataset_from_directory(directory=train_dir,
                                                                label_mode='categorical',
                                                                batch_size=32,
                                                                image_size=img_size)

test_data = tf.keras.preprocessing.image_dataset_from_directory(directory=test_dir,
                                                               label_mode="categorical",
                                                               image_size=img_size)

# # Checkout the class names of our dataset
class_names = train_data.class_names
print(class_names)

### Model 0: Building a transfer learning model using the Keras Functional API

# 1. Create base model with tf.keras.applications
base_model = tf.keras.applications.EfficientNetB0(include_top=False)

# 2. Freeze the base model (so the pre-learned patterns remain)
base_model.trainable = False

# 3. Create inputs into the base model
inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer")

# 4. Pass the inputs to the base_model (note: using tf.keras.applications, EfficientNet inputs don't have to be normalized)
x = base_model(inputs)

# Check data shape after passing it to the base_model
print(f"Shape after base_model : {x.shape}")

# 5. Average pool the outputs of the base model (aggregate all the most important information, reduce number of computations)
x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
print(f"After GlobalAveragePooling2D(): {x.shape}")

# 6. Create the output activation layer
outputs = tf.keras.layers.Dense(6, activation="softmax", name="Output_Layer")(x)

# 7. Combine the outputs and the inputs into a model
model_0 = tf.keras.models.Model(inputs=inputs, outputs=outputs)

# 8. Compile the model
model_0.compile(loss="categorical_crossentropy",
                optimizer="adam",
                metrics=["accuracy"])

# 9. Fit the Model (we use less steps for validation so it's faster)
history_0 = model_0.fit(train_data,
                        epochs=2,
                        steps_per_epoch=len(train_data),
                        validation_data=test_data,
                        validation_steps=len(test_data),
                        # Track our model's training logs for visualization later
                        callbacks=[create_tensorboard_callback("transfer_learning", "ASL_feature_extract")])

model_0.summary()

plot_loss_curves(history_0)

# model_0 = tf.keras.models.load_model("SL_Model_Efficientnet.h5")

results_1_percent_data_aug = model_0.evaluate(test_data)


def make_prediction(model, img, real_classname, input_shape, class_names):
    plt.figure(figsize=(10, 7))
    plt.imshow(img)
    plt.title(f"Real: {real_classname}")
    plt.axis(False);
    image = cv2.resize(img, input_shape)

    img_reshaped = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])

    pred = model.predict(img_reshaped)

    if len(pred[0]) > 1:
        pred_class = class_names[pred.argmax()]  # if more than one output, take the max
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])]

    plt.figure(figsize=(10, 7))
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False);
    return pred_class


A = cv2.imread("Make Real Predictions/A1.jpg")
B = cv2.imread("Make Real Predictions/B.jpg")
D = cv2.imread("Make Real Predictions/D.jpg")

print(make_prediction(model=model_0, img=A, real_classname='A', input_shape=(224, 224), class_names=class_names))

print(make_prediction(model=model_0, img=B, real_classname='B', input_shape=(224, 224), class_names=class_names))

print(make_prediction(model=model_0, img=D, real_classname='D', input_shape=(224, 224), class_names=class_names))

# Save a model
model_0.save("SL_Model_Efficientnet.h5")