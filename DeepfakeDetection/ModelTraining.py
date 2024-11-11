import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import RMSprop
import logging
from sklearn.metrics import confusion_matrix

# Set up the logging configuration
logging.basicConfig(
    level=logging.INFO,  # Only log INFO and above (ignore DEBUG)
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_activity.log'),  # Save logs to a file
        logging.StreamHandler()  # Also output logs to the console
    ]
)

logger = logging.getLogger()

# Define the Deep4SNet CNN model
def create_cnn_model(input_shape, num_classes):
    logger.info('Creating the CNN model...')
    model = Sequential([
        Input(shape=input_shape),  # Add Input layer explicitly
        Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='random_normal', bias_initializer='zeros'),
        MaxPooling2D((2, 2), strides=(2, 2), padding='valid'),
        Conv2D(32, (3, 3), strides=(1, 1), activation='relu'),
        MaxPooling2D((2, 2), strides=(2, 2), padding='valid'),
        Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
        MaxPooling2D((2, 2), strides=(2, 2), padding='valid'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='sigmoid')  # Output layer
    ])
    logger.info('CNN model created successfully!')
    return model

# Model params 
input_shape = (150, 150, 3)
num_classes = 1  # Number of outputs per input

# Initialize model
logger.info('Initializing the model...')
model_our_HVoice_SiF_filtered = create_cnn_model(input_shape, num_classes)

# Compile the model
logger.info('Compiling the model...')
model_our_HVoice_SiF_filtered.compile(optimizer=RMSprop(learning_rate=0.001),
                                      loss='binary_crossentropy',
                                      metrics=['accuracy'])
logger.info('Model compiled successfully!')

# EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss',  # Monitor validation loss
                               patience=3,         # Number of epochs with no improvement to wait before stopping
                               verbose=1,          # Print message when early stopping occurs
                               restore_best_weights=True)  # Restore the model with the best weights

# CSVLogger to log training history to a file
csv_logger = CSVLogger('training_log.log', append=True)

# Define directories for train, validation, and test sets
train_dir = 'D:/DeepLearning-Project (virtual-env)/DeepfakeDetection/Data/H-Voice_SiF-Filtered/Training_Set'
validation_dir = 'D:/DeepLearning-Project (virtual-env)/DeepfakeDetection/Data/H-Voice_SiF-Filtered/Validation_Set'
test_dir = 'D:/DeepLearning-Project (virtual-env)/DeepfakeDetection/Data/H-Voice_SiF-Filtered/Test_Set'

# Set up ImageDataGenerators
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Log the dataset loading process
logger.info('Loading training data...')
train_generator_HVoice_SiF_Filtered = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

logger.info('Loading validation data...')
valid_generator_HVoice_SiF_Filtered = valid_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

logger.info('Loading test data...')
test_generator_HVoice_SiF_Filtered = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Training the model with early stopping and CSV logging
logger.info('Starting model training...')
history = model_our_HVoice_SiF_filtered.fit(
    train_generator_HVoice_SiF_Filtered,
    epochs=10,
    validation_data=valid_generator_HVoice_SiF_Filtered,
    callbacks=[early_stopping, csv_logger]  # Add CSVLogger callback here
)

# Save the model
model_dir = 'models/Deep4SNet-Our-HVoice_SiF-Filtered.keras'
model_our_HVoice_SiF_filtered.save(model_dir)
logger.info(f'Model saved to {model_dir}!')

# Evaluate model on validation set
logger.info('Evaluating model on validation set...')
loss_val, accuracy_val = model_our_HVoice_SiF_filtered.evaluate(valid_generator_HVoice_SiF_Filtered)
logger.info(f'Validation accuracy: {accuracy_val * 100}%')

# Evaluate model on test set
logger.info('Evaluating model on test set...')
loss_test, accuracy_test = model_our_HVoice_SiF_filtered.evaluate(test_generator_HVoice_SiF_Filtered)
logger.info(f'Test accuracy: {accuracy_test * 100}%')

# Predict probabilities for the test set
logger.info('Predicting probabilities for the test set...')
y_pred_proba = model_our_HVoice_SiF_filtered.predict(test_generator_HVoice_SiF_Filtered)

# Convert probabilities to binary predictions (0 or 1)
threshold = 0.5
y_pred_binary = (y_pred_proba > threshold).astype(int)

# Get true labels for the test set
y_true = test_generator_HVoice_SiF_Filtered.classes

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_binary)

# Extract true negatives (TN), false positives (FP), false negatives (FN), and true positives (TP)
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]
TP = conf_matrix[1, 1]

# Compute false positive rate (FPR)
FPR = FP / (FP + TN)
logger.info(f'False Positive Rate (FPR): {FPR * 100}%')