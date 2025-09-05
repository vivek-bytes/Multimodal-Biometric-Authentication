'''import visualkeras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout

# Define the photo size
photo_size = 224  # You can adjust this based on your actual input size

# Function to create the VGG model
def create_vgg_model():
    # Load VGG16 model without the top classifier layers
    base_model = VGG16(include_top=False, input_shape=(photo_size, photo_size, 3))

    # Freeze all layers except specific layers (example: layers 1-3 and 15-18)
    for layer_idx in range(len(base_model.layers)):
        if layer_idx not in [1, 2, 3, 15, 16, 17, 18]:
            base_model.layers[layer_idx].trainable = False

    # Add new classifier layers
    flat1 = Flatten()(base_model.output)
    dense1 = Dense(4096, activation='relu')(flat1)
    drop1 = Dropout(0.5)(dense1)
    dense2 = Dense(4096, activation='relu')(drop1)
    drop2 = Dropout(0.5)(dense2)
    dense3 = Dense(2048, activation='relu')(drop2)  # Additional layer
    drop3 = Dropout(0.5)(dense3)  # Additional layer
    output = Dense(30, activation='softmax')(drop3)  # Assuming you have 30 classes

    # Define new model
    model = Model(inputs=base_model.inputs, outputs=output)
    return model

# Create the model
model = create_vgg_model()

# Define color mappings for different layer types
color_map = {
    'InputLayer': 'lightblue',
    'Conv2D': 'red',
    'MaxPooling2D': 'orange',
    'Flatten': 'purple',
    'Dense': 'green',
    'Dropout': 'yellow',
}

# Visualize the model with labels
visualkeras.layered_view(model, legend=True, color_map=color_map).show()  # Display using your system viewer
#visualkeras.layered_view(model, to_file='output.png', legend=True, color_map=color_map)  # Write to disk
visualkeras.layered_view(model, to_file='output.png', legend=True, color_map=color_map).show()  # Write and show
'''

import visualkeras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from PIL import Image

# Define the photo size
photo_size = 224  # You can adjust this based on your actual input size

# Function to create the VGG model
def create_vgg_model():
    # Load VGG16 model without the top classifier layers
    base_model = VGG16(include_top=False, input_shape=(photo_size, photo_size, 3))

    # Freeze all layers except specific layers (example: layers 1-3 and 15-18)
    for layer_idx in range(len(base_model.layers)):
        if layer_idx not in [1, 2, 3, 15, 16, 17, 18]:
            base_model.layers[layer_idx].trainable = False

    # Add new classifier layers
    flat1 = Flatten()(base_model.output)
    dense1 = Dense(4096, activation='relu')(flat1)
    drop1 = Dropout(0.5)(dense1)
    dense2 = Dense(4096, activation='relu')(drop1)
    drop2 = Dropout(0.5)(dense2)
    dense3 = Dense(2048, activation='relu')(drop2)  # Additional layer
    drop3 = Dropout(0.5)(dense3)  # Additional layer
    output = Dense(30, activation='softmax')(drop3)  # Assuming you have 30 classes

    # Define new model
    model = Model(inputs=base_model.inputs, outputs=output)
    return model

# Create the model
model = create_vgg_model()

# Define color mappings for different layer types
color_map = {
    'InputLayer': 'lightblue',
    'Conv2D': 'red',
    'MaxPooling2D': 'orange',
    'Flatten': 'purple',
    'Dense': 'green',
    'Dropout': 'yellow',
}

# Generate the model visualization
image = visualkeras.layered_view(model, legend=True, color_map=color_map)

# Resize the image for better readability
new_width = image.width * 10  # Adjust the multiplier as needed
new_height = image.height * 10  # Adjust the multiplier as needed
image = image.resize((new_width, new_height), Image.LANCZOS)

# Save and show the resized image
image.save('output.png')
image.show()
