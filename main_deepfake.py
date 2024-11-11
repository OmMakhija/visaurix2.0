import sounddevice as sd
import numpy as np
import wavio
import os
import subprocess

# Get the user's name
user_name = input("Please enter your name: ").strip()

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
recording = sd.rec(int(10 * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')  # Start a 10-second recording buffer
sd.wait()  # Wait until the recording is complete

# Allow user to stop recording
input("Recording stopped. Press Enter to save the recording...")

# Save the recording as a .wav file in the user's directory
wavio.write(output_file, recording, sample_rate, sampwidth=4)  # Sampwidth=4 for float32
print("Recording saved.")

# Call inference.py to analyze the recording
result = subprocess.run(["python", r"DeepfakeDetection\inference.py", output_file], capture_output=True, text=True)

# Assuming inference.py returns '1' for real and '0' for fake in its output
if result.returncode == 0:
    # Parse the output (assuming it's '1' for real or '0' for fake)
    output = result.stdout.strip()  # Get the output and remove extra spaces/newlines
    if output == '1':
        print("The recording is REAL.")
    elif output == '0':
        print("The recording is FAKE.")
    else:
        print("Unexpected output from inference.py.")
else:
    print("Error in running inference.py.")
