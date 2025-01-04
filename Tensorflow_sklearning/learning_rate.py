model = Sequential()
pretrained_model = inceptionV3(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
model.add(pretrained_model)
model.add(Flatten())
model.add(Dense(8, activation='softmax'))
model.summary()

learning_rate = 0.001

model.compile(
    optimizer=optimizers.SGD(learning_rate=learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_flow,
    epochs=15,
    validation_data=val_flow
)
