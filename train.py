from keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D, Conv2DTranspose, Concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping
from preprocess import load_data

# Load preprocessed data
X_train, Y_train, X_val, Y_val, _, _ = load_data()

# Define double convolution layer
def double_conv(input, filters):

    conv1 = Conv2D(filters, kernel_size=(3,3), padding="same")(input)
    bn1 = BatchNormalization()(conv1)
    af1 = ReLU()(bn1)

    conv2 = Conv2D(filters, kernel_size=(3,3), padding="same")(af1)
    bn2 = BatchNormalization()(conv2)
    af2 = ReLU()(bn2)

    return af2

# Define encoder
def encoder(input, filters):
    enc = double_conv(input, filters)
    pool = MaxPooling2D(strides=(2,2))(enc)
    return enc, pool

# Define decoder
def decoder(input, features, filters):
    upsamp = Conv2DTranspose(filters, kernel_size=(2,2), strides=(2,2), padding="same")(input)
    concat = Concatenate()([upsamp, features])
    out = double_conv(concat, filters)
    return out

# Define U_Net architecture
def U_Net(image_shape):

    input = Input(image_shape)

    enc1, pool1 = encoder(input, 64)
    enc2, pool2 = encoder(pool1, 128)
    enc3, pool3 = encoder(pool2, 256)
    enc4, pool4 = encoder(pool3, 512)

    bottleneck = double_conv(pool4, 1024)

    dec1 = decoder(bottleneck, enc4, 512)
    dec2 = decoder(dec1, enc3, 256)
    dec3 = decoder(dec2, enc2, 128)
    dec4 = decoder(dec3, enc1, 64)

    output = Conv2D(1, 1, padding="same", activation="sigmoid")(dec4)

    model = Model(input, output)

    return model

image_shape = (128, 128, 3)

model = U_Net(image_shape)

# Define Early Stopping with patience 5
earlystop = EarlyStopping(monitor='val_loss',patience=5)
callbacks_list = [earlystop]

# Specify loss, optimizer and metrics
model.compile(optimizer="adam",loss="BinaryCrossentropy",metrics=["accuracy"])

# Train model
model.fit(X_train,Y_train,epochs=50,validation_data=(X_val,Y_val),callbacks=callbacks_list)

# Save model
model.save("U_Net.keras")