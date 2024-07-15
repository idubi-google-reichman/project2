import tensorflow as tf
import threading
import tkinter as tk

# Flag to stop training
stop_training_flag = False

# Custom callback to stop training
class CustomEarlyStopping(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if stop_training_flag:
            print("Stopping training...")
            self.model.stop_training = True

# Function to be called when the button is pressed
def stop_training():
    global stop_training_flag
    stop_training_flag = True
    print("Stop button pressed.")

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten the input
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Dummy data for demonstration
(X_train, y_train), (X_val, y_val) = tf.keras.datasets.mnist.load_data()
X_train, X_val = X_train / 255.0, X_val / 255.0

# Function to start training
def start_training():
    model.fit(
        X_train, y_train,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=[CustomEarlyStopping()]
    )

# Create a separate thread for training to keep the GUI responsive
training_thread = threading.Thread(target=start_training)

# Create the main window
root = tk.Tk()
root.title("Training Controller")

# Add a button to stop training
stop_button = tk.Button(root, text="Stop Training", command=stop_training)
stop_button.pack()

# Start the training thread
training_thread.start()

# Run the GUI loop
root.mainloop()

# Join the training thread to wait for its completion before exiting the script
training_thread.join()
