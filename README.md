# ASL ML Project

## Overview

This project implements a lightweight American Sign Language (ASL) recognition system using Mediapipe for landmark extraction and classic machine learning models (SVM, Random Forest, KNN). The system is designed to detect hand signs from webcam frames, convert them into text, and optionally provide spoken output (text-to-speech) to assist people who are deaf, mute, or visually impaired.

## Highlights (from project report)

- Input pipeline uses Mediapipe to extract hand landmarks and generate feature vectors.
- Multiple models were evaluated (KNN, Random Forest, SVM); SVM was selected for final use based on accuracy and reliability.
- The project includes preprocessing, model training, evaluation (confusion matrices), and a web-based/demo interface for real-time inference.

## Repository Structure

- `ML_Final_Project (1).ipynb` — main training and evaluation notebook
- `ML_P (1).py` — supporting scripts for preprocessing or inference
- `ML_with Mediapipe (1).ipynb` — notebook demonstrating Mediapipe feature extraction
- `models/` — saved model artifacts (SVM, RF, KNN, etc.)
- `README.md` — project summary and run instructions

## Quick Start

1. Create and activate a Python virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install required packages (example):

```powershell
pip install numpy pandas scikit-learn mediapipe matplotlib seaborn
```

3. Run notebooks or scripts for training / inference:

- Open the notebook `ML_Final_Project (1).ipynb` in Jupyter Lab / Notebook to reproduce experiments.
- Use `ML_P (1).py` for quick inference scripts (see the top of the file for usage).

## Models

Saved model artifacts are stored in the `models/` folder. Notable files include:

- `models/svm_mediapipe.pkl_1` — trained SVM (pickled pipeline containing scaler + SVM)
- Additional model checkpoints (if present) are kept in `models/`.

Note: large binary model files are tracked where needed; if you prefer to keep them out of the Git history, consider using Git LFS or uploading them to a release.

## Running Inference (example)

Load the pickled SVM pipeline and run inference on extracted features:

```python
import pickle
model = pickle.load(open('models/svm_mediapipe.pkl_1', 'rb'))
# X is a feature vector extracted via Mediapipe
pred = model.predict([X])
```

## Results & Evaluation

The report includes confusion matrices and accuracy metrics for each model. Open `ML_Final_Project (1).ipynb` to view detailed evaluation, plots, and discussion.

## Notes

- The project report and presentation were reviewed and key points summarized here. The original `ML_FINAL_Project_Report.docx` and `ML_FINAL_PROJECT.pptx` have been removed from the repository to keep the repo focused; contact the author if you need the original files.
- If you want me to convert key report sections into separate markdown docs (Methodology, Dataset, Results), I can add them.

## Contact

Project authors: Maruthi V Kamath.
