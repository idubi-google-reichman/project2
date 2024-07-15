def create_model() : 
        image_size = 128
        base_model = MobileNet( input_shape = (image_size,image_size,3),
                                include_top=False,
                                weights='imagenet')
        base_model.trainable = False  # Freeze the base model

        model = models.Sequential([        
            Resizing(image_size, image_size, interpolation="nearest", input_shape=train_images.shape[1:]),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(10, activation='softmax')
        ])

        # Specify the learning rate
        
        # Instantiate the Adam optimizer with the default learning rate
        optimizer = Adam()

        model.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

        return model





def train_model(model,labels,epocs,validatoion_data,test_data):
    return model.fit(train_images, train_labels, epochs=5, validation_data=(val_images, val_labels))

    test_data=(test_images, test_labels,test_log_level)
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"Test accuracy: {test_acc}, Test loss: {test_loss}")