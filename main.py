
import os
import logging
from register_user import record_audio_for_user  # Importing the record_audio_for_user function
from voiceauth.gmm import load_features_from_directory, train_gmm, save_gmm_model
import sounddevice as sd
import numpy as np
import wavio
from DeepfakeDetection.DataProcessing import process_audio
from DeepfakeDetection.run_record import DeepfakeDetector
from DeepfakeDetection.train import Deep4SNet
import joblib
from scipy.io import wavfile
from voiceauth.feature_extraction import extract_features


def sign_up(username):
    """
    Sign up a new user by recording their audio, extracting features, training a GMM model, and saving it.
    """
    print(f"Starting sign up for {username}...")

    try:
        # Record and save audio samples in a specified directory for the user
        print("Recording audio for user...")
        audio_directory = record_audio_for_user(username)  # Ensure this function is defined
        print(f"Audio recorded and saved in directory: {audio_directory}")
        
        # Define the paths for UBM and where to save the GMM model
        ubm_model_path = r'D:/DeepLearning-Project (virtual-env)/voiceauth/model/ubm_model.pkl'  # Replace with actual path to your UBM model
        model_save_directory = r'D:/DeepLearning-Project (virtual-env)/voiceauth/model'
        os.makedirs(model_save_directory, exist_ok=True)  # Create directory if it doesn't exist
        gmm_model_save_path = os.path.join(model_save_directory, f"{username}.gmm")  # Path where GMM will be saved
        print(f"GMM model will be saved to: {gmm_model_save_path}")

        # Number of components for GMM
        n_components = 32  # Adjust as needed
        print(f"Number of components for GMM: {n_components}")
        
        # Load features from the audio directory
        print(f"Loading features from directory: {audio_directory}")
        features = load_features_from_directory(audio_directory)
        print(f"Features loaded. Number of features: {features.shape[0]}")
        
        if features.size == 0:
            print("No valid features extracted. Please check your audio recordings.")
            return
        
        # Train a GMM using the extracted features and UBM model
        print(f"Training GMM model with {n_components} components...")
        gmm_model = train_gmm(features, ubm_model_path, n_components)
        print("GMM model trained successfully.")
        
        # Save the trained GMM model with the username
        print(f"Saving the trained GMM model to: {gmm_model_save_path}")
        save_gmm_model(gmm_model, gmm_model_save_path)
        print(f"Model saved successfully for user: {username}")
        
    except Exception as e:
        print(f"An error occurred during the sign-up process: {e}")
        logging.error(f"Error during sign up for {username}: {e}")
        import traceback
        traceback.print_exc()  # Print full traceback to debug where the error occurred

def login(user_name):
    """
    Login an existing user by recording their audio and checking it against the stored GMM model.
    """
    # Create a directory named after the user
    user_dir = os.path.join("recordings", user_name)
    os.makedirs(user_dir, exist_ok=True)

    # File path for the audio recording
    output_file = os.path.join(user_dir, f"{user_name}_recording.wav")

    # Sentence for the user to read
    print("Please read the following sentence clearly: ")
    print("\n\"Technology has transformed the way we communicate, learn, and interact with the world. "
          "From smartphones to artificial intelligence, it shapes our daily lives and influences our decisions.\"\n")

    # Prompt user to start recording
    input("Press Enter to start recording...")

    # Recording settings
    sample_rate = 44100

    # Start recording
    print("Recording... Press Enter again to stop.")
    recording = sd.rec(int(10 * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()

    # Allow user to stop recording
    input("Recording stopped. Press Enter to save the recording...")

    # Save the recording as a .wav file in the user's directory
    wavio.write(output_file, recording, sample_rate, sampwidth=4)
    print("Recording saved.")

    # AUTOMATICALLY SETTING THE CUTOFF FREQUENCY
    cutoff_frequency = 4000  # Fixed cutoff frequency (in Hz) for low-pass filter

    # Call the process_audio function to process the saved recording and get the histogram path
    print("Processing the recorded audio for prediction...")

    # Assuming the function `process_audio` returns the path to the histogram
    histogram_path = process_audio(output_file, cutoff_frequency=cutoff_frequency, output_dir=user_dir)

    # Print the histogram path
    print(f"Histogram for the recorded audio saved at: {histogram_path}")

    # Initialize the DeepfakeDetector
    detector = DeepfakeDetector(r"D:/DeepLearning-Project (virtual-env)/DeepfakeDetection/models/best_model.pth")
    
    # Predict using the generated histogram image
    result = detector.predict_single(histogram_path)
    
    if result:
        print(f"\nDeepfake Detection Results for {histogram_path}:")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print(f"Probability of Real: {result['probabilities']['real']:.2f}%")
        print(f"Probability of Fake: {result['probabilities']['fake']:.2f}%")
        
        # Check if it's a real or fake voice and return 1 or print failure message
        if result['prediction'] == 'REAL':
            model_path = os.path.join(r'D:/DeepLearning-Project (virtual-env)/voiceauth/model', f"{user_name}.gmm")  # Corrected model path
            
            # Known log-likelihoods from authentication samples
            known_log_likelihoods = [-36.5, -35.8, -37.2]  # Add more known values here based on your training data
            
            gmm_model = joblib.load(model_path)
            rate, audio = wavfile.read(output_file)  # Use the actual recorded file path
            features = extract_features(audio, rate)
            
            log_likelihood = gmm_model.score(features)
            
            print(f"Log-Likelihood: {log_likelihood}")
            
            threshold = np.percentile(known_log_likelihoods, 90)
            threshold += 6.0  # Adding margin
            
            print(f"Dynamic Threshold: {threshold}")
            
            if log_likelihood > threshold:
                print("Authentication Successful")
                return True
            else:
                print("Authentication Failed")
                return False
            
        else:
            print("Cloned voice, authentication failed.")

def main():
    """Prompt user for signup or login."""
    action = input("Choose an option:\n1. Sign Up\n2. Login\nEnter 1 or 2: ").strip()

    if action == "1":
        username = input("Enter your username to sign up: ").strip()
        print(f"Attempting to sign up with username: {username}")
        sign_up(username)
    elif action == "2":
        username = input("Enter your username to sign up: ").strip()
        # Check if the GMM model exists for the username
        gmm_model_path = os.path.join(r'D:/DeepLearning-Project (virtual-env)/voiceauth/model', f"{username}.gmm")
        
        if os.path.exists(gmm_model_path):
            login(username)
        else:
            print("No user found. Please sign up first.")
            exit()  # Exit the program if the user does not exist
    else:
        print("Invalid option. Exiting...")

if __name__ == "__main__":
    print("Starting main execution...")
    main()
