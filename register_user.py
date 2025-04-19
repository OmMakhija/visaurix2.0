import os
import numpy as np
import sounddevice as sd
import joblib
import librosa
from scipy.io.wavfile import write
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from python_speech_features import mfcc, delta

SAMPLE_RATE = 16000
MODEL_DIR = "models"
UBM_PATH = os.path.join(MODEL_DIR, "ubm.gmm")
UBM_SCALER_PATH = os.path.join(MODEL_DIR, "ubm_scaler.pkl")
UBM_FEATURES_PATH = os.path.join(MODEL_DIR, "ubm_features.npy")
os.makedirs(MODEL_DIR, exist_ok=True)

def record_audio(filename, duration=10):
    print(f"✩ Recording for {duration} seconds...")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    write(filename, SAMPLE_RATE, audio)
    print(f"✅ Saved: {filename}")

def remove_silence(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    y_trimmed = y_trimmed / np.max(np.abs(y_trimmed))
    return y_trimmed, sr

def extract_features(signal, rate):
    frame_length = int(0.025 * rate)
    hop_length = int(0.01 * rate)

    mfcc_feat = mfcc(signal, rate, numcep=13, winlen=0.025, winstep=0.01, nfft=512)
    d_mfcc = delta(mfcc_feat, 2)
    dd_mfcc = delta(d_mfcc, 2)

    energy = np.array([
        np.log(np.sum(signal[i:i + frame_length]**2) + 1e-10)
        for i in range(0, len(signal) - frame_length + 1, hop_length)
    ]).reshape(-1, 1)

    pitch = librosa.yin(signal.astype(float), fmin=80, fmax=400, sr=rate, frame_length=frame_length, hop_length=hop_length)
    pitch = np.nan_to_num(pitch).reshape(-1, 1)

    min_len = min(mfcc_feat.shape[0], d_mfcc.shape[0], dd_mfcc.shape[0], energy.shape[0], pitch.shape[0])
    mfcc_feat = mfcc_feat[:min_len]
    d_mfcc = d_mfcc[:min_len]
    dd_mfcc = dd_mfcc[:min_len]
    energy = energy[:min_len]
    pitch = pitch[:min_len]

    return np.hstack([mfcc_feat, d_mfcc, dd_mfcc, energy, pitch])

def train_ubm():
    print("🧠 Training UBM...")
    all_feats = []
    for i in range(3):
        input(f"Press Enter to record generic sample {i+1} for UBM (10s)...")
        path = f"ubm_sample{i+1}.wav"
        record_audio(path, duration=10)
        y, sr = remove_silence(path)
        feats = extract_features(y, sr)
        all_feats.append(feats)

    combined = np.vstack(all_feats)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(combined)

    ubm = GaussianMixture(n_components=32, covariance_type='diag', max_iter=300, n_init=3, random_state=42)
    ubm.fit(features_scaled)

    joblib.dump(ubm, UBM_PATH)
    joblib.dump(scaler, UBM_SCALER_PATH)
    np.save(UBM_FEATURES_PATH, combined)
    print("✅ UBM trained and saved.")

def sign_up(username):
    if not all(map(os.path.exists, [UBM_PATH, UBM_SCALER_PATH, UBM_FEATURES_PATH])):
        train_ubm()

    print(f"\n🔐 Signing up user: {username}")
    all_features = []

    for i in range(3):
        input(f"Press Enter to record sample {i+1} (10 seconds)...")
        path = f"{username}_sample{i+1}.wav"
        record_audio(path, duration=10)
        y, sr = remove_silence(path)
        feats = extract_features(y, sr)
        all_features.append(feats)

    combined_features = np.vstack(all_features)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(combined_features)

    user_gmm = GaussianMixture(n_components=32, max_iter=300, covariance_type='diag', n_init=3, random_state=42)
    user_gmm.fit(features_scaled)

    ubm = joblib.load(UBM_PATH)
    ubm_scaler = joblib.load(UBM_SCALER_PATH)
    ubm_feats = np.load(UBM_FEATURES_PATH)
    ubm_features_scaled = ubm_scaler.transform(ubm_feats[:combined_features.shape[0], :features_scaled.shape[1]])

    score_diff = user_gmm.score(features_scaled) - ubm.score(ubm_features_scaled)
    threshold = score_diff - 5

    joblib.dump(user_gmm, os.path.join(MODEL_DIR, f"{username}.gmm"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"{username}_scaler.pkl"))
    np.save(os.path.join(MODEL_DIR, f"{username}_threshold.npy"), threshold)

    print("✅ Sign-up complete and model saved.")

def login(username):
    print(f"\n🔓 Logging in: {username}")
    gmm_path = os.path.join(MODEL_DIR, f"{username}.gmm")
    scaler_path = os.path.join(MODEL_DIR, f"{username}_scaler.pkl")
    threshold_path = os.path.join(MODEL_DIR, f"{username}_threshold.npy")

    if not all(map(os.path.exists, [gmm_path, scaler_path, threshold_path])):
        print("❌ User not registered.")
        return

    input("Press Enter to record login sample (10 seconds)...")
    login_audio = f"{username}_login.wav"
    record_audio(login_audio, duration=10)

    y, sr = remove_silence(login_audio)
    feats = extract_features(y, sr)

    user_gmm = joblib.load(gmm_path)
    user_scaler = joblib.load(scaler_path)
    threshold = np.load(threshold_path)

    ubm = joblib.load(UBM_PATH)
    ubm_scaler = joblib.load(UBM_SCALER_PATH)
    ubm_feats = np.load(UBM_FEATURES_PATH)
    feats_user_scaled = user_scaler.transform(feats)
    feats_ubm_scaled = ubm_scaler.transform(ubm_feats[:feats.shape[0], :feats.shape[1]])

    user_score = user_gmm.score(feats_user_scaled)
    ubm_score = ubm.score(feats_ubm_scaled)

    print(f"👤 User score: {user_score:.2f}")
    print(f"🌐 UBM score: {ubm_score:.2f}")
    print(f"📉 Threshold (diff): {threshold:.2f}")

    if (user_score - ubm_score) > threshold:
        print("✅ Authentication Successful.")
    else:
        print("❌ Authentication Failed.")

def main():
    while True:
        print("\n1. Train UBM\n2. Sign Up\n3. Login\n4. Exit")
        choice = input("Choose: ").strip()
        if choice == "1":
            train_ubm()
        elif choice == "2":
            username = input("Username: ").strip().lower()
            sign_up(username)
        elif choice == "3":
            username = input("Username: ").strip().lower()
            login(username)
        elif choice == "4":
            break
        else:
            print("Invalid option.")

if __name__ == "__main__":
    main()