import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score # Accuracy metrics
import pickle


df = pd.read_csv('coords.csv')
# print(df.shape)
# print(df.head(5))
X = df.drop('class', axis=1)  # features
y = df['class']  # target value

print("Startt-***********", y.value_counts())
print(type(y))
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print("Finish ********", pd.Series(y).value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=24)

#### Train Machine Learning Classification Model¶ ####
from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

scaler = StandardScaler()
scaler.fit(X_train)
scaler.transform(X_train)
scaler.transform(X_test)

## Multiclass Model
import tensorflow as tf

# Set a random seed
tf.random.set_seed(42)

# 1. Create the model
model_multi = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, name="Input_Layer", activation="relu"),
        tf.keras.layers.Dense(100, name="Hidden_Layer", activation="relu"),
        tf.keras.layers.Dense(5, name="Output_Layer", activation="softmax"),
], name="MultiClass_Model")

# 2. Compile the Model
model_multi.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=["accuracy"])

# 3.Fit the Model
model_multi.fit(X_train, y_train, epochs=150, batch_size=32, shuffle=True, validation_data=(X_test, y_test))


# model = RandomForestClassifier(random_state=0, n_estimators=100)
# model.fit(X_train, y_train)
# pred = model.predict(X_test)

######### Evaluate and Serialize Model ########

# print("accuracy_score = ", accuracy_score(y_test, pred))
loss, accuracy = model_multi.evaluate(X_test, y_test)
# with open('sign_classifier.pkl', 'wb') as f:
#     pickle.dump(model_multi, f)
model_multi.save("sign_classifier.h5")