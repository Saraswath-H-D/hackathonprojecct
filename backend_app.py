import os
import io
import uuid
import numpy as np
import cv2
import face_recognition
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

app = Flask(__name__)
CORS(app)

# --- CONFIG ---
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --- DATA STORAGE ---
students = []  # Each student: {"id", "name", "photo", "encoding"}
attendance_log = defaultdict(list)

# --- HELPER FUNCTIONS ---
def get_face_encoding(image_path):
    """Extract facial encoding from image, return None if no face detected."""
    img = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(img)
    return encodings[0] if encodings else None

def mark_attendance(student_id, status):
    """Mark student attendance only once per day."""
    today = datetime.now().strftime("%Y-%m-%d")
    if status == "Present" and any(log["date"] == today and log["status"] == "Present" for log in attendance_log[student_id]):
        return
    attendance_log[student_id].append({"status": status, "date": today})

# --- ROUTES ---
@app.route("/register", methods=["POST"])
def register_student():
    try:
        student_name = request.form.get("student_name")
        student_photo = request.files.get("student_photo")

        if not student_name or not student_photo:
            return jsonify({"error": "Name and photo required"}), 400

        # Generate unique filename
        ext = os.path.splitext(student_photo.filename)[1]
        filename = secure_filename(f"{uuid.uuid4()}{ext}")
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        student_photo.save(filepath)

        encoding = get_face_encoding(filepath)
        if encoding is None:
            os.remove(filepath)
            return jsonify({"error": "No face detected. Upload a clear front-facing photo."}), 400

        student_id = len(students) + 1
        students.append({
            "id": student_id,
            "name": student_name,
            "photo": filename,
            "encoding": encoding.tolist()
        })
        attendance_log[student_id] = []

        return jsonify({"message": f"✅ {student_name} registered successfully!", "id": student_id}), 200
    except Exception as e:
        return jsonify({"error": f"Registration failed: {str(e)}"}), 500

@app.route("/take_attendance", methods=["POST"])
def take_attendance():
    try:
        if not students:
            return jsonify({"error": "No students registered"}), 400

        file = request.files.get("frame")
        if not file:
            return jsonify({"error": "No frame uploaded"}), 400

        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Failed to read image"}), 400

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        present_students = []
        absent_students = [s["name"] for s in students]

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                [np.array(s["encoding"]) for s in students],
                face_encoding,
                tolerance=0.6
            )
            if True in matches:
                idx = matches.index(True)
                student = students[idx]
                if student["name"] not in present_students:
                    present_students.append(student["name"])
                    if student["name"] in absent_students:
                        absent_students.remove(student["name"])
                    mark_attendance(student["id"], "Present")

        # Mark absent for remaining students
        for student in students:
            if student["name"] not in present_students:
                mark_attendance(student["id"], "Absent")

        return jsonify({
            "present_students": present_students,
            "absent_students": absent_students
        })
    except Exception as e:
        return jsonify({"error": f"Attendance failed: {str(e)}"}), 500

@app.route("/students", methods=["GET"])
def get_students():
    return jsonify([{"id": s["id"], "name": s["name"], "photo": s["photo"]} for s in students])

@app.route("/get_attendance_graph", methods=["GET"])
def get_attendance_graph():
    student_id = request.args.get("student_id", type=int)
    student_name = request.args.get("student_name", default="Student")
    view = request.args.get("view", default="daily")  # "daily" or "monthly"

    if student_id not in attendance_log:
        return jsonify({"error": "Student not found"}), 404

    history = attendance_log[student_id]

    if view == "monthly":
        date_status = defaultdict(lambda: {"Present": 0, "Absent": 0})
        for log in history:
            month = log["date"][:7]
            date_status[month][log["status"]] += 1
        labels = sorted(date_status.keys())
        present_counts = [date_status[m]["Present"] for m in labels]
        absent_counts = [date_status[m]["Absent"] for m in labels]
        title = f"Monthly Attendance for {student_name}"
    else:
        date_status = defaultdict(lambda: {"Present": 0, "Absent": 0})
        for log in history:
            date_status[log["date"]][log["status"]] += 1
        labels = sorted(date_status.keys())
        present_counts = [date_status[d]["Present"] for d in labels]
        absent_counts = [date_status[d]["Absent"] for d in labels]
        title = f"Daily Attendance for {student_name}"

    plt.figure(figsize=(6, 4))
    plt.bar(labels, present_counts, label="Present", color="green")
    plt.bar(labels, absent_counts, bottom=present_counts, label="Absent", color="red")
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.xlabel("Date" if view == "daily" else "Month")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plt.close()
    return send_file(img, mimetype="image/png")

@app.route("/send_report", methods=["POST"])
def send_report():
    """Generate a PDF report and send it as downloadable file."""
    try:
        data = request.json
        student_id = data.get("student_id")
        recipient_type = data.get("recipient_type")  # "parent" or "management"

        if not student_id or student_id not in attendance_log:
            return jsonify({"error": "Invalid student"}), 400

        student = students[student_id - 1]
        logs = attendance_log[student_id]

        # Generate PDF
        pdf_filename = f"Attendance_Report_{student['name'].replace(' ', '_')}.pdf"
        pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], pdf_filename)

        c = canvas.Canvas(pdf_path, pagesize=A4)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, 800, f"Attendance Report - {student['name']}")

        c.setFont("Helvetica", 12)
        c.drawString(50, 780, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        c.drawString(50, 760, f"Recipient: {'Parent' if recipient_type == 'parent' else 'Management'}")

        total_present = sum(1 for log in logs if log["status"] == "Present")
        total_absent = sum(1 for log in logs if log["status"] == "Absent")
        percentage = (total_present / max(len(logs), 1)) * 100

        c.drawString(50, 730, f"Total Present: {total_present}")
        c.drawString(50, 710, f"Total Absent: {total_absent}")
        c.drawString(50, 690, f"Attendance %: {percentage:.2f}%")

        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, 660, "Detailed Attendance:")

        c.setFont("Helvetica", 12)
        y = 640
        for log in logs:
            c.drawString(50, y, f"{log['date']} - {log['status']}")
            y -= 20
            if y < 50:
                c.showPage()
                c.setFont("Helvetica", 12)
                y = 800

        c.save()

        return send_file(pdf_path, as_attachment=True, download_name=pdf_filename)
    except Exception as e:
        return jsonify({"error": f"Failed to generate report: {str(e)}"}), 500

if __name__ == "__main__":
    print("✅ Backend running at http://127.0.0.1:5000")
    app.run(debug=True)

