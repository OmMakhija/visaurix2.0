import joblib
import numpy as np
from scipy.io import wavfile
from feature_extraction import extract_features  # Import the function from feature_extraction.py
from sklearn import preprocessing

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

if __name__ == "__main__":
    model_path = 'model/Ananya.pkl'  # Path to your trained model
    audio_file_paths = [
        'Data/ananya/sample_1.wav',
        'Data/Krishika/sample_2.wav',
        'Data/ananya/sample_3.wav', 
        'Data/ananya/sample_3.wav',
        'Data/Krishika/sample_4.wav',
        'Data/ananya/sample_5.wav',
        'Data/Krishika-Test/sample_8.wav',
        'Data/Krishika-Test/sample_9.wav',
    ]

    # Known log-likelihoods from authentication samples
    known_log_likelihoods = [-36.5, -35.8, -37.2]  # Add more known values here based on your training data

    # Test authentication for each audio file
    for audio_file_path in audio_file_paths:
        print(f"Testing {audio_file_path}...")
        authenticate_user(model_path, audio_file_path, known_log_likelihoods)
