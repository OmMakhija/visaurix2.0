import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy.signal
from tensorflow.keras.models import load_model
from PIL import Image
import os
import subprocess

# Function to plot spectrogram
def plot_spectrogram(audio, sr, title, save_path=None):
    plt.figure(figsize=(10, 4))
    spectrogram = librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max), sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()  # Close the plot to prevent display

# Function to apply a low-pass filter
def filter(audio_data, cutoff_frequency, sr):
    nyquist_frequency = sr / 2
    cutoff_normalized = cutoff_frequency / nyquist_frequency
    b, a = scipy.signal.butter(4, cutoff_normalized, btype='low')
    filtered_audio = np.apply_along_axis(lambda x: scipy.signal.filtfilt(b, a, x), axis=0, arr=audio_data)
    return filtered_audio

# Function to compute histogram of filtered audio and save spectrogram and histogram images
def compute_histogram_filtered(file_path, dir):
    # Load audio file
    audio, sr = librosa.load(file_path, sr=44100)
    
    # Plot spectrogram of original audio and save
    plot_spectrogram(audio, sr, title='Original Audio Spectrogram', save_path=os.path.join(dir, f'orig_spectrogram_{os.path.basename(file_path)}.png'))

    # Apply filter
    cutoff_frequency = 4000
    filtered_audio = filter(audio, cutoff_frequency, sr=44100)
    
    # Plot spectrogram of filtered audio and save
    plot_spectrogram(filtered_audio, sr, title='Filtered Audio Spectrogram', save_path=os.path.join(dir, f'filtered_spectrogram_{os.path.basename(file_path)}.png'))
    
    # Calculate histogram of filtered audio
    hist, bins = np.histogram(filtered_audio, bins=256, range=(-1, 1))  # 2^8 bins

    # Plot histogram and save
    plt.figure()
    plt.bar(bins[:-1], hist, width=(bins[1] - bins[0]), color='black')
    plt.savefig(os.path.join(dir, f'hist_{os.path.basename(file_path)}.png'))
    plt.close()  # Close the plot to prevent display

    return hist

# Load the trained model
model_path = r'D:\DeepLearning-Project (virtual-env)\DeepfakeDetection\models\Deep4SNet-Our-HVoice_SiF-Filtered.keras'
model_our_HVoice_SiF_Filtered = load_model(model_path)

# Function to process the recording and make prediction
def process_audio_and_predict(file_path):
    # Output directory for spectrogram and histogram
    histogram_dir = os.path.join("output", "histograms")
    os.makedirs(histogram_dir, exist_ok=True)

    # Compute histograms and save spectrograms for the audio file
    print(f"Computing histogram for: {file_path}")
    compute_histogram_filtered(file_path, histogram_dir)

    # Load and process images for prediction
    # Assuming spectrograms and histograms are saved as images
    image_paths = [
        os.path.join(histogram_dir, f'hist_{os.path.basename(file_path)}.png')
    ]

    # Resize images to match the input shape of the model (150x150)
    resized_images = []
    for image_path in image_paths:
        image = Image.open(image_path)
        image = image.convert('RGB')  # Ensure image is RGB
        resized_image = image.resize((150, 150))  # Resize to 150x150
        image_array = np.array(resized_image)  # Convert to numpy array
        resized_images.append(image_array)

    # Convert list of resized images to numpy array
    resized_images = np.array(resized_images)

    # Use the loaded model to predict classes
    predictions = model_our_HVoice_SiF_Filtered.predict(resized_images)

    # Output predictions (returning 1 or 0 instead of printing)
    predicted_class = int(predictions[0][0])  # Get the predicted class (0 or 1)
    return predicted_class  # Return the prediction (1 for real, 0 for fake)

# Main function to handle audio file path input
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Please provide the path to the audio file.")
        sys.exit(1)
    
    audio_file_path = sys.argv[1]  # Get the audio file path from the command line argument
    print(f"Processing audio file: {audio_file_path}")
    
    # Call the function to process the audio file and get prediction
    prediction = process_audio_and_predict(audio_file_path)

    # Return the result to the calling process (1 for real, 0 for fake)
    print(f"Prediction result: {'real' if prediction == 1 else 'fake'}")
    sys.exit(prediction)  # Exit with 1 for real and 0 for fake
