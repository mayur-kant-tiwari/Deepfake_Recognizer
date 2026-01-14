# Deepfake Audio Sentinel

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Framework](https://img.shields.io/badge/Framework-Flask-green)
![Status](https://img.shields.io/badge/Status-Deployed-brightgreen)

---

## Demo & Deployment

### Live Web App
**[Click Here to Try the Detector (Hugging Face)]**
https://agrim-007-deep-fake-recogniser.hf.space

### Project Walkthrough
**[Watch the Demo Video]**
https://youtu.be/0YZOYuAeTuc
---

## The Problem
With the rise of Generative AI, cloning a human voice takes just 3 seconds. Traditional detectors fail on high-quality recordings or compress the audio too much. We built a physics-based detector that analyzes the **spectral fingerprint** of the audio to distinguish between organic vocal cords and synthetic generation.

---

## Key Features

* **Real-Time Analysis:** Instantly processes uploaded WAV, MP3, and M4A files.
* **Hybrid Detection Engine:** Uses Librosa to extract MFCCs, Zero-Crossing Rate (ZCR), and Spectral Rolloff.
* **Phone-Ready:** Includes a custom "Anti-Denoising" algorithm (Dithering) to prevent false positives from modern phone noise cancellation.
* **Strict Confidence Logic:** The AI only flags audio as "Real" if it is >70% confident, reducing dangerous false negatives.

---

## How It Works

Unlike basic black-box models, our system looks for specific digital artifacts:

1.  **Preprocessing:** We normalize the audio to 16kHz.
2.  **Dithering Injection:** Modern phones (iPhone/Android) "clean" audio so aggressively that it looks like a Deepfake (Digital Silence). We inject microscopic white noise (0.0001 amplitude) to restore natural texture without altering the human sound.
3.  **Feature Extraction:**
    * **ZCR:** Measures how "rough" the wave is (Human voice is rougher than AI).
    * **Spectral Rolloff:** Analyzes the high-frequency cutoff frequencies.
4.  **Classification:** A Scikit-learn Logistic Regression model makes the final verdict based on 23 extracted features.

---

## Tech Stack

* **Backend:** Flask (Python)
* **Audio Processing:** Librosa, Numba, SoundFile, FFmpeg
* **Machine Learning:** Scikit-learn, Numpy, Pandas
* **Deployment:** Docker, Gunicorn, Hugging Face Spaces

---

## Team & Contributions

The team behind this project:

* **Mayur:** Frontend Development & UI/UX Design.
* **Shreeya:** Backend Architecture & API Integration.
* **Agrim & Vipul:** AI Model Training, Feature Engineering, & Dataset Curation.
* **Agrim:** Cloud Deployment (Docker/Hugging Face) & DevOps.

---

## Local Installation

If you want to run this locally:

```bash
# 1. Clone the repository
git clone [https://github.com/YOUR_USERNAME/Deepfake-Audio-Detector.git](https://github.com/YOUR_USERNAME/Deepfake-Audio-Detector.git)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install FFmpeg (Required for MP3/M4A support)
# (On Mac: brew install ffmpeg | On Ubuntu: sudo apt install ffmpeg)

# 4. Run the app
python app.py
