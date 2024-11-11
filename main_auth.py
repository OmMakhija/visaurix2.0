import os
import sounddevice as sd
import numpy as np
import wave
import joblib
from scipy.io import wavfile
from feature_extraction import extract_features  # Import the function from feature_extraction.py
from sklearn import preprocessing

class AudioRecorder:
    def __init__(self, fs=16000):
        self.fs = fs
        self.recording = False
        self.audio_data = []

    def callback(self, indata, frames, time, status):
        """Callback function that gets called continuously while recording."""
        if self.recording:
            self.audio_data.append(indata.copy())

    def record(self):
        """Starts recording audio."""
        self.recording = True
        with sd.InputStream(samplerate=self.fs, channels=1, callback=self.callback):
            print("Recording... (Press Enter to stop)")
            input()  # Wait until Enter is pressed
            self.recording = False

def save_audio(filename, audio_data):
    """Saves recorded audio data as a WAV file."""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(16000)
        wf.writeframes(b''.join(audio_data))

def determine_threshold(log_likelihoods, margin=10.0):
    """Determine an appropriate threshold based on log-likelihoods."""
    # Using the 90th percentile as the baseline threshold
    threshold = np.percentile(log_likelihoods, 90)
    
    # Add a margin to the threshold to ensure it's not too low
    threshold += margin
    
    return threshold

def authenticate_user(model_path, audio_file_path, known_log_likelihoods):
    """Authenticate user based on audio sample."""
    # Load the GMM model
    gmm_model = joblib.load(model_path)

    # Read the new audio file
    rate, audio = wavfile.read(audio_file_path)

    # Extract features from the audio using the function from feature_extraction.py
    features = extract_features(audio, rate)

    # Compute log-likelihood for the audio sample
    log_likelihood = gmm_model.score(features)

    print(f"Log-Likelihood: {log_likelihood}")

    # Determine dynamic threshold based on known log-likelihoods
    threshold = determine_threshold(known_log_likelihoods)
    
    print(f"Dynamic Threshold: {threshold}")

    # Check against threshold for authentication
    if log_likelihood > threshold:
        print("Authentication Successful")
        return True
    else:
        print("Authentication Failed")
        return False

def record_audio_for_user(username):
    """Record audio samples for the given user and save them."""
    base_dir = r'Data'
    user_dir = os.path.join(base_dir, username)

    # Create a new directory for the user if it doesn't exist
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
        print(f"Created directory for user: {user_dir}")
    else:
        print(f"Directory already exists: {user_dir}")

    # Predefined longer statement about technology
    tech_statement = (
        "Technology has transformed the way we communicate, learn, and interact with the world. "
        "From smartphones to artificial intelligence, it shapes our daily lives and influences our decisions."
    )

    print(f"Please read the following statement:\n\"{tech_statement}\"")

    # Record multiple samples for the user
    num_samples = 10  # Change this number for more recordings
    recorder = AudioRecorder()
    
    for i in range(num_samples):
        recorder.record()
        save_audio(os.path.join(user_dir, f'sample_{i+1}.wav'), recorder.audio_data)
        recorder.audio_data.clear()  # Clear data for next sample

    # Save the username to a file for later use
    with open(os.path.join(base_dir, 'last_username.txt'), 'w') as f:
        f.write(username)

def main():
    print("Welcome to the Voice Authentication System!")
    
    action = input("Choose an option:\n1. Record new audio samples\n2. Authenticate user\nEnter 1 or 2: ").strip()

    if action == "1":
        username = input("Enter your username: ").strip()
        record_audio_for_user(username)
        print(f"Audio samples have been recorded for {username}.")
    elif action == "2":
        model_path = 'model/Ananya.pkl'  # Path to your trained model

        # Known log-likelihoods from authentication samples
        known_log_likelihoods = [-36.5, -35.8, -37.2]  # Add more known values here based on your training data

        # Ask the user to enter audio file paths for testing
        print("Enter the paths of audio files for testing (one file at a time). Type 'done' when finished.")

        audio_file_paths = []
        while True:
            file_path = input("Enter audio file path: ").strip()
            if file_path.lower() == 'done':
                break
            audio_file_paths.append(file_path)

        # Test authentication for each audio file entered by the user
        for audio_file_path in audio_file_paths:
            print(f"Testing {audio_file_path}...")
            authenticate_user(model_path, audio_file_path, known_log_likelihoods)
    else:
        print("Invalid option. Exiting...")

if __name__ == "__main__":
    main()
