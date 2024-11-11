import os
import sounddevice as sd
import numpy as np
import wave

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

if __name__ == "__main__":
    username = input("Enter your username: ").strip()

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