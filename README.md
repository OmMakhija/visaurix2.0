# Enhancing Voice Authentication Security: A Hybrid Approach Using GMM-UBM and Deep4SNet for Deepfake Detection

## Overview  
This project introduces a robust voice authentication system designed to counter synthetic audio attacks. By integrating advanced deepfake detection and speaker verification technologies, it ensures high accuracy, real-time performance, and scalability for applications like secure banking, smart devices, and access control systems.  

---

## Features  
- **Dual-Stage Security**:  
  - Combines deepfake detection and speaker verification for enhanced reliability.  
- **High Detection Accuracy**:  
  - Deepfake Detection: 90.25% accuracy against synthetic audio attacks.  
  - Speaker Verification: 100% accuracy for enrolled users with zero false acceptances.  
- **Real-Time Processing**: Low computational overhead, ensuring seamless integration with various systems.  
- **Scalability**: Easily adaptable to a wide range of applications and environments.  

---

## Technical Architecture  

### Dual-Stage Security Framework  
1. **Deepfake Detection**  
   - **Model**: **Deep4SNet**, a specialized neural network for identifying synthetic audio.  
   - **Features**: Spectrogram-based analysis for binary classification (real vs. fake audio).  

2. **Speaker Verification**  
   - **Model**: **GMM-UBM (Gaussian Mixture Model - Universal Background Model)**.  
   - **Features**: **MFCC (Mel-Frequency Cepstral Coefficients)** extraction to create robust, speaker-specific models.  

---

## Installation  

### Prerequisites  
Ensure you have the following installed:  
- **Python**: Version 3.8 or higher.  
- **Libraries**: Refer to the `requirements.txt` file for all dependencies.  

### Step-by-Step Guide  

1. **Clone the Repository**   
   ```bash
   git clone https://github.com/ananyakaligal/Voice-Authentication.git
   cd voice-authentication-research
   ```
   

2. **Set Up a Virtual Environment**
   ```bash
   python -m venv venv  
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```


3. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the System**  
   ```bash
   python main.py
   ```
(Images/Screenshot.png)
---




