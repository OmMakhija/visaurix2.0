import os
import logging
import traceback
import json
import numpy as np
import sounddevice as sd
import wavio
import joblib
from scipy.io import wavfile

from register_user import record_audio_for_user
from voiceauth.gmm import load_features_from_directory, train_gmm, save_gmm_model
from voiceauth.feature_extraction import extract_features
from DeepfakeDetection.DataProcessing import process_audio
from DeepfakeDetection.run_record import DeepfakeDetector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
THRESHOLD_FILE = os.path.join(BASE_DIR, 'voiceauth', 'model', 'thresholds.json')

# Setup logging
logging.basicConfig(
    filename=os.path.join(BASE_DIR, "debug.log"),
    filemode="a",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_thresholds():
    if os.path.exists(THRESHOLD_FILE):
        with open(THRESHOLD_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_threshold(username, threshold):
    thresholds = load_thresholds()
    thresholds[username] = threshold
    with open(THRESHOLD_FILE, 'w') as f:
        json.dump(thresholds, f)

def load_all_gmm_models(model_directory):
    models = {}
    for file in os.listdir(model_directory):
        if file.endswith(".gmm"):
            username = os.path.splitext(file)[0]
            model_path = os.path.join(model_directory, file)
            try:
                gmm = joblib.load(model_path)
                models[username] = gmm
            except Exception as e:
                logging.error(f"Failed to load GMM for {username}: {e}")
    return models

def sign_up(username):
    logging.info(f"Starting sign up for user: {username}")
    try:
        print("Recording audio for user...")
        audio_directory = record_audio_for_user(username)

        ubm_model_path = os.path.join(BASE_DIR, 'voiceauth', 'model', 'ubm_model.pkl')
        model_save_directory = os.path.join(BASE_DIR, 'voiceauth', 'model')
        os.makedirs(model_save_directory, exist_ok=True)
        gmm_model_save_path = os.path.join(model_save_directory, f"{username}.gmm")

        features = load_features_from_directory(audio_directory)

        if features.size == 0:
            logging.warning("No valid features extracted.")
            return

        gmm_model = train_gmm(features, ubm_model_path, n_components=32)
        save_gmm_model(gmm_model, gmm_model_save_path)

        user_threshold = np.percentile(gmm_model.score_samples(features), 10)
        save_threshold(username, float(user_threshold))
        logging.info(f"Threshold saved for user {username}: {user_threshold}")

        print("User registration and voice model training complete.")
    except Exception as e:
        logging.error(f"Error in sign_up for {username}: {e}")
        logging.debug(traceback.format_exc())
        print("An error occurred during sign-up.")

def login():
    logging.info("Starting login process.")

    user_dir = os.path.join(BASE_DIR, "recordings", "login_attempt")
    os.makedirs(user_dir, exist_ok=True)
    output_file = os.path.join(user_dir, "login_recording.wav")

    print("Please read the following sentence clearly: ")
    print("\n\"Technology has transformed the way we communicate, learn, and interact with the world. "
          "From smartphones to artificial intelligence, it shapes our daily lives and influences our decisions.\"\n")

    input("Press Enter to start recording...")
    sample_rate = 44100
    print("Recording...")
    recording = sd.rec(int(10 * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    input("Recording complete. Press Enter to save...")

    try:
        wavio.write(output_file, recording, sample_rate, sampwidth=4)
        logging.debug(f"Recording saved at: {output_file}")
    except Exception as e:
        logging.error(f"Error saving recording: {e}")
        return

    try:
        histogram_path = process_audio(output_file, cutoff_frequency=4000, output_dir=user_dir)
        detector = DeepfakeDetector(os.path.join(BASE_DIR, 'DeepfakeDetection', 'models', 'best_model.pth'))
        result = detector.predict_single(histogram_path)
    except Exception as e:
        logging.error(f"Deepfake detection error: {e}")
        return

    if result and result['prediction'] == 'REAL':
        print("deepfake: no")
    else:
        print("deepfake: yes")

    if result and result['prediction'] == 'REAL':
        print("✅ Voice passed deepfake check.")

        try:
            rate, audio = wavfile.read(output_file)
            features = extract_features(audio, rate)

            if features is None or len(features) == 0:
                logging.warning("Extracted features are empty or None.")
                print("❌ Error: Feature extraction failed.")
                return

            models = load_all_gmm_models(os.path.join(BASE_DIR, 'voiceauth', 'model'))
            thresholds = load_thresholds()

            scores = {}
            for user, gmm in models.items():
                try:
                    score = gmm.score(features)
                    scores[user] = score
                except Exception as e:
                    logging.warning(f"Failed to score with {user}'s model: {e}")

            if not scores:
                print("❌ No valid models to compare.")
                return

            best_user = max(scores, key=scores.get)
            best_score = scores[best_user]
            user_threshold = thresholds.get(best_user, -9999)

            print(f"[DEBUG] Best match: {best_user} (score: {best_score}, threshold: {user_threshold})")

            if best_score > user_threshold:
                print(f"✅ Speaker identified as: {best_user}")
                logging.info(f"Speaker identified as: {best_user}")
                return best_user
            else:
                print("❌ Speaker not recognized. Voice does not match any user.")
                logging.warning("Speaker recognition failed.")
                return None

        except Exception as e:
            logging.error(f"Voice authentication error: {e}")
            logging.debug(traceback.format_exc())
            print("❌ Authentication process failed.")
    else:
        print("❌ Deepfake voice detected. Login denied.")
        logging.warning("Deepfake voice detected.")

def main():
    print("Welcome to the Secure Voice Authentication System.")
    logging.info("Program started.")

    try:
        choice = input("Choose an option:\n1. Sign Up\n2. Login\nEnter 1 or 2: ").strip()

        if choice == "1":
            username = input("Enter new username to sign up: ").strip().lower()
            sign_up(username)
        elif choice == "2":
            login()
        else:
            print("Invalid input. Exiting.")
    except Exception as e:
        logging.error(f"Main loop error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
