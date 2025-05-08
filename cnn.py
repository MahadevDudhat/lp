from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Load and normalize the Fashion MNIST data
fashion_mnist = keras.datasets.fashion_mnist
(train_img, train_labels), (test_img, test_labels) = fashion_mnist.load_data()

# Normalize and reshape input for CNN (add channel dimension)
train_img = train_img.reshape(-1, 28, 28, 1) / 255.0
test_img = test_img.reshape(-1, 28, 28, 1) / 255.0

# CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(train_img, train_labels, epochs=10)

# Evaluate model
test_loss, test_acc = model.evaluate(test_img, test_labels)
print("Accuracy of testing: ", test_acc)

# Make predictions
predictions = model.predict(test_img)
predicted_labels = np.argmax(predictions, axis=1)

# Plotting predictions
num_rows = 5
num_cols = 5
num_imgs = num_rows * num_cols

plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_imgs):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plt.imshow(test_img[i].reshape(28, 28), cmap='gray')
    plt.axis("off")
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plt.bar(range(10), predictions[i])
    plt.xticks(range(10))
    plt.ylim([0, 1])
    plt.title(f"Predicted: {predicted_labels[i]}")
plt.tight_layout()
plt.show()
