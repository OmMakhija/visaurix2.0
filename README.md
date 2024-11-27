# Secure Voice Authentication: Deepfake Detection & Speaker Verification  

## 🔬 Research Overview  
This system addresses the critical challenge of synthetic audio attacks by integrating advanced deepfake detection and speaker verification technologies. Developed by researchers at RV University, it ensures robust voice security with high accuracy and real-time efficiency.  

---

## 📋 Highlights  
- **Deepfake Detection Accuracy**: 90.25%  
- **Speaker Verification Accuracy**: 100% for enrolled users  
- **Real-Time Performance**: Low computational overhead  
- **Hybrid Approach**: Combines deep learning and traditional methods  

---

## 🧠 Technical Architecture  

### Dual-Stage Security Framework  
1. **Deepfake Detection**  
   - Neural Network: **Deep4SNet**  
   - Features: Spectrogram analysis for real vs. fake classification  

2. **Speaker Verification**  
   - Model: **GMM-UBM (Gaussian Mixture Model - Universal Background Model)**  
   - Features: **MFCC (Mel-Frequency Cepstral Coefficients)**  

---

## 🔢 Performance Metrics  

### Deepfake Detection  
- **Samples Analyzed**: 1,426  
  - Real: 668, Fake: 758  
- **Results**:  
  - Real Voices Identified: 633  
  - Fake Voices Identified: 654  
- **Metrics**:  
  - Accuracy: 90.25%  
  - Precision: 94.92%  
  - Recall: 86.28%  

### Speaker Verification  
- **User Accuracy**: 100%  
- **False Acceptances**: 0  
- **Robustness**: Consistent across multiple tests  

---

## 📦 Installation  

1. Clone the repository:  
   ```bash
   git clone https://github.com/your-organization/voice-authentication-research.git
   cd voice-authentication-research
