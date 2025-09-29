# Smart Attendance System

This repository contains the code for a Smart Attendance System using facial recognition.

## Features
- Student registration with photo upload
- Teacher dashboard for automated attendance via camera
- Student performance dashboard with attendance graphs
- PDF attendance report generation

## How to Run
1. Install Python 3.10 and required packages (see below).
2. Start the backend:
   ```sh
   python backend_app.py
   ```
3. Open `index.html` in your browser.

## Requirements
- Python 3.10
- Flask
- flask-cors
- face_recognition
- opencv-python
- numpy
- matplotlib
- reportlab

Install dependencies:
```sh
pip install -r requirements.txt
```

## Project Structure
- `backend_app.py` - Flask backend for registration, attendance, and reporting
- `index.html` - Frontend web app

## Notes
- Do not commit `uploads/` or `venv/` directories. They are excluded via `.gitignore`.
- For any issues, please open an issue in this repository.

---

**Submitted as part of the hackathon project by Saraswath-H-D.**
