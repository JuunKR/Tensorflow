import autokeras as ak
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28 * 28 * 1)
x_test = x_test.reshape(-1, 28 * 28 * 1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 28,28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = ak.ImageClassifier(
    overwrite=True,
    max_trials=1
)

# ak.StructuredDataRegressor

model.fit(x_train, y_train, epochs=2)

y_predict = model.predict(x_test)
print(y_predict[:10])

result = model.evaluate(x_test, y_test)
print(result)

model2 = model.export_model()
model2.summary()