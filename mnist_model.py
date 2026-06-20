import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.3f}")
pred = model.predict(x_test[:10])

print("Predicted:", np.argmax(pred, axis=1))
print("Actual   :", y_test[:10])
for i in range(5):
    plt.imshow(x_test[i].reshape(28,28), cmap='gray')
    plt.title(f"Predicted: {np.argmax(pred[i])}, Actual: {y_test[i]}")
    plt.show()
img = Image.open("digit.png").convert('L').resize((28,28))
img = np.array(img) / 255.0
img = img.reshape(1, 784)
pred = model.predict(img)
print("Predicted digit:", np.argmax(pred))    