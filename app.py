from flask import Flask, request, jsonify, send_file, render_template, redirect,send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import random
import string
import io
import qrcode
import time
import threading
from dotenv import load_dotenv
from PyPDF2 import PdfReader, PdfMerger
from docx import Document
from reportlab.pdfgen import canvas
from PIL import Image
from pdf2image import convert_from_bytes
import pytesseract
import google.generativeai as genai
from google.generativeai import models
import cv2
import numpy as np
from google.generativeai import GenerativeModel
import re
from fpdf import FPDF
import uuid


# ----------------------- App Setup -----------------------
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 512 * 1024 * 1024  # 512 MB

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['UPLOAD_FOLDER']= UPLOAD_FOLDER
DELETE_AFTER = 2 * 60 * 60
CHECK_INTERVAL = 300


# load env file


load_dotenv()

try:
    gemini_api_key = os.environ.get("GOOGLE_API_KEY")
    if not gemini_api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file")
    genai.configure(api_key=gemini_api_key)
    print("✅ Google Generative AI configured successfully!")
except Exception as e:
    print(f"❌ Error configuring Google Generative AI: {e}")
print("Models available to your API key:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)


def generate_code(length=6):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
def auto_delete():
    while True:
        now = time.time()
        if os.path.exists(UPLOAD_FOLDER):
            for i in os.listdir(UPLOAD_FOLDER):
                path = os.path.join(UPLOAD_FOLDER, i)
                if os.path.isfile(path) and now - os.path.getmtime(path) > DELETE_AFTER:
                    try:
                        os.remove(path)
                    except Exception as e:
                        print("Error deleting file:", e)
        time.sleep(CHECK_INTERVAL)


threading.Thread(target=auto_delete, daemon=True).start()

# ----------------------- Routes -----------------------
@app.route("/")
def home():
    return render_template("index.html")

# ------------------------ Upload and receiver section ------------------------#

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error": "No file uploaded"}), 400
    
    try:
        filename, filepath = save_file(file)
        download_url = f"{request.url_root}download/{filename}"
        return jsonify({"code": filename.split("_")[0], "filename": filename, "download_url": download_url})
    except Exception as e:
        return jsonify({"error": f"Upload failed: {e}"}), 500


@app.route("/api/status", methods=["GET"])
def api_status():
    return jsonify({
        "status": "running",
        "timestamp": time.time()
    }), 200


# THIS IS THE CORRECT, UNIVERSAL DOWNLOAD ROUTE
@app.route("/download/<path:filepath>")
def download(filepath):
    # Determine the base folder based on the path
    if "document_tools" in filepath or "image_tools" in filepath:
        base_folder = OUTPUT_FOLDER
    else:
        base_folder = UPLOAD_FOLDER
    
    # Securely build the full path
    full_path = os.path.join(base_folder, filepath)
    directory = os.path.dirname(full_path)
    filename = os.path.basename(full_path)

    if not os.path.exists(full_path):
         return jsonify({"error": "File not found or has expired"}), 404

    return send_from_directory(directory, filename, as_attachment=True)


def save_file(file):
    code = generate_code()
    original_filename = secure_filename(file.filename)
    new_filename = f"{code}_{original_filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
    file.save(file_path)
    return new_filename, file_path


@app.route("/check_file/<code>", methods=["GET"])
def check_file(code):
    for fname in os.listdir(app.config['UPLOAD_FOLDER']):
        if fname.startswith(code + "_"):
            return jsonify({"filename": fname})
    return jsonify({"error": "File not found"}), 404


@app.route("/get_qr/<code>", methods=["GET"])
def get_qr(code):
    filename = None
    for f in os.listdir(app.config['UPLOAD_FOLDER']):
        if f.startswith(code + "_"):
            filename = f
            break
    if not filename:
        return jsonify({"error": "File not found"}), 404
    
    download_url = f"{request.url_root}download/{filename}"
    
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(download_url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    
    return send_file(buf, mimetype='image/png')


def save_file_locally(file_bytes, filename, subfolder=""):
    """Saves file bytes to the OUTPUT_FOLDER with an optional subfolder."""
    try:
        # Create the full directory path if it doesn't exist
        directory = os.path.join(OUTPUT_FOLDER, subfolder)
        os.makedirs(directory, exist_ok=True)
        
        filepath = os.path.join(directory, secure_filename(filename))
        with open(filepath, "wb") as f:
            f.write(file_bytes)
            
        # Return the relative path for the download URL
        return os.path.join(subfolder, filename)
    except Exception as e:
        print(f"Error saving file locally: {e}")
        return None
def save_text_as_pdf(text, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in text.split("\n"):
        pdf.multi_cell(0, 10, line)
    pdf.output(os.path.join("outputs", filename))

def save_text_as_docx(text, filename):
    doc = Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    doc.save(os.path.join("outputs", filename))

# ----------------------- Document Helpers -----------------------
def extract(file_like):

    """
    Accepts a file-like object (supports .read()). Returns extracted text (PDF).
    """
    try:
        # if it's a Flask FileStorage, read bytes
        if hasattr(file_like, "read"):
            data = file_like.read()
        else:
            data = file_like
        reader = PdfReader(io.BytesIO(data))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return ""


def pdf_to_word_bytes(pdf_bytes):
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        doc = Document()
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                doc.add_paragraph(f"--- Page {i + 1} ---")
                doc.add_paragraph(text)
        f = io.BytesIO()
        doc.save(f)
        f.seek(0)
        return f.getvalue()
    except Exception as e:
        print(f"PDF to Word conversion error: {e}")
        return None


def word_to_pdf_bytes(word_bytes):
    try:
        doc = Document(io.BytesIO(word_bytes))
        f = io.BytesIO()
        c = canvas.Canvas(f)
        y = 800
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                c.drawString(50, y, text)
                y -= 20
                if y < 50:
                    c.showPage()
                    y = 800
        c.save()
        f.seek(0)
        return f.getvalue()
    except Exception as e:
        print(f"Word to PDF conversion error: {e}")
        return None


def image_to_pdf_bytes(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        f = io.BytesIO()
        img.save(f, "PDF")
        f.seek(0)
        return f.getvalue()
    except Exception as e:
        print(f"Image to PDF conversion error: {e}")
        return None


def pdf_to_images_bytes(pdf_bytes):
    try:
        pages = convert_from_bytes(pdf_bytes, 300)
        images_bytes = []
        for i, page in enumerate(pages):
            buf = io.BytesIO()
            page.save(buf, format="PNG")
            buf.seek(0)
            images_bytes.append(buf.getvalue())
        return images_bytes
    except Exception as e:
        print(f"PDF to images conversion error: {e}")
        return []


def merge_pdf_bytes(pdf_files_bytes):
    try:
        merger = PdfMerger()
        for pdf_bytes in pdf_files_bytes:
            merger.append(io.BytesIO(pdf_bytes))
        out = io.BytesIO()
        merger.write(out)
        merger.close()
        out.seek(0)
        return out.getvalue()
    except Exception as e:
        print(f"PDF merge error: {e}")
        return None


def resize_image(image_bytes, max_width=800, max_height=800):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.thumbnail((max_width, max_height))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print(f"Image resize error: {e}")
        return None


def image_ocr_text(image_bytes):
    try:
        # Convert to grayscale
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply thresholding for better contrast
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # Convert back to PIL for pytesseract
        pil_img = Image.fromarray(thresh)

        # Run OCR with language set
        text = pytesseract.image_to_string(pil_img, lang="eng")
        return text.strip()
    except Exception as e:
        print(f"OCR error: {e}")
        return ""


# ----------------------- Document Endpoints -----------------------
@app.route("/doc/pdf_to_word", methods=["POST"])
def pdf_to_word_route():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file provided"}), 400

    pdf_bytes = file.read()
    word_bytes = pdf_to_word_bytes(pdf_bytes)
    if not word_bytes:
        return jsonify({"error": "Conversion failed"}), 500

    filename = f"{file.filename.rsplit('.', 1)[0]}.docx"
    save_file_locally(word_bytes, filename, subfolder="document_tools")
    url = f"/download/document_tools/{filename}"


    if not url:
        return jsonify({"error": "Upload failed"}), 500

    return jsonify({"download_url": url})


@app.route("/doc/word_to_pdf", methods=["POST"])
def word_to_pdf_route():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file provided"}), 400

    pdf_bytes = word_to_pdf_bytes(file.read())
    if not pdf_bytes:
        return jsonify({"error": "Conversion failed"}), 500

    filename = f"{file.filename.rsplit('.', 1)[0]}.pdf"
    save_file_locally(pdf_bytes, filename, subfolder="document_tools")
    url = f"/download/document_tools/{filename}"


    if not url:
        return jsonify({"error": "Upload failed"}), 500

    return jsonify({"download_url": url})


@app.route("/doc/image_to_pdf", methods=["POST"])
def image_to_pdf_route():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file provided"}), 400

    pdf_bytes = image_to_pdf_bytes(file.read())
    if not pdf_bytes:
        return jsonify({"error": "Conversion failed"}), 500

    # Save locally
    filename = f"{file.filename.rsplit('.', 1)[0]}.pdf"
    save_file_locally(pdf_bytes, filename, subfolder="document_tools")
    url = f"/download/document_tools/{filename}"

    return jsonify({"download_url": url})


@app.route("/doc/pdf_to_images", methods=["POST"])
def pdf_to_images_route():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file provided"}), 400

    images_bytes = pdf_to_images_bytes(file.read())
    if not images_bytes:
        return jsonify({"error": "Conversion failed"}), 500

    urls = []
    for i, b in enumerate(images_bytes):
        filename = f"{file.filename.rsplit('.', 1)[0]}_page_{i+1}.png"
        save_file_locally(b, filename, subfolder="document_tools")
        urls.append(f"/download/document_tools/{filename}")


    if not urls:
        return jsonify({"error": "Upload failed"}), 500

    return jsonify({"download_urls": urls})


@app.route("/doc/merge_pdfs", methods=["POST"])
def merge_pdfs_route():
    files = request.files.getlist("files")
    if not files or len(files) < 2:
        return jsonify({"error": "At least two files required"}), 400

    pdf_bytes_list = [f.read() for f in files]
    merged_bytes = merge_pdf_bytes(pdf_bytes_list)

    if not merged_bytes:
        return jsonify({"error": "Merge failed"}), 500

    filename = f"merged_{int(time.time())}.pdf"
    save_file_locally(merged_bytes, filename, subfolder="document_tools")
    url = f"/download/document_tools/{filename}"


    if not url:
        return jsonify({"error": "Upload failed"}), 500

    return jsonify({"download_url": url})


# ----------------------- Image Routes -----------------------
@app.route("/img/resize", methods=["POST"])
def resize_route():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file provided"}), 400

    resized_bytes = resize_image(file.read(), 800, 800)
    if not resized_bytes:
        return jsonify({"error": "Resize failed"}), 500

    filename = f"{file.filename.rsplit('.', 1)[0]}_resized.jpg"
    save_file_locally(resized_bytes, filename, subfolder="image_tools")
    url = f"/download/image_tools/{filename}"


    if not url:
        return jsonify({"error": "Upload failed"}), 500

    return jsonify({"download_url": url})


@app.route("/img/ocr", methods=["POST"])
def image_ocr_route():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file provided"}), 400

    text = image_ocr_text(file.read())
    return jsonify({"extracted_text": text})

def ask_Ycloudai(prompt, max_tokens=2048):
    """Sends a prompt to the Gemini model and returns the response."""
    try:
        model = genai.GenerativeModel('gemini-flash-latest')
        generation_config = genai.types.GenerationConfig(max_output_tokens=max_tokens)
        response = model.generate_content(prompt, generation_config={"max_output_tokens": max_tokens})
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API error: {e}")
        return "Error: Could not get a response from the AI."

def extract_text_from_file(file):
    """Extracts text from various file types."""
    filename = file.filename
    text = ""
    try:
        if filename.endswith(".pdf"):
            reader = PdfReader(file.stream)
            for page in reader.pages:
                text += page.extract_text() or ""
            if not text.strip():  # OCR Fallback
                file.seek(0)
                images = convert_from_bytes(file.read())
                for img in images:
                    text += pytesseract.image_to_string(img)
        elif filename.endswith(".docx"):
            doc = Document(file.stream)
            text = "\n".join([para.text for para in doc.paragraphs])
        elif filename.endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(file.stream)
            text = pytesseract.image_to_string(image)
        elif filename.endswith(".txt"):
            text = file.stream.read().decode('utf-8')
        else:
            return None, f"Unsupported file type: {filename}"
        return text, None
    except Exception as e:
        return None, f"Error processing file {filename}: {e}"

def chunk_text(text, chunk_size=4000, overlap=200):
    """Splits a long text into smaller, overlapping chunks."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# ----------------------- Ycloud Spark - Document Processor -----------------------
def generate_summary(text_chunks):
    summaries = [ask_Ycloudai(f"Summarize this text concisely:\n{c}") for c in text_chunks]
    return ask_Ycloudai("Combine these summaries into one cohesive summary:\n" + "\n".join(summaries))

def generate_qa(text_chunks, question):
    answers = [ask_Ycloudai(f"Based ONLY on this text, answer:\nText: {c}\nQuestion: {question}") for c in text_chunks]
    return ask_Ycloudai(f"Synthesize into a single, clear answer for: '{question}'\nInformation:\n" + "\n".join(answers))

def generate_quiz(full_text):
    return ask_Ycloudai("Generate 5 multiple-choice questions (2 easy, 2 hard, 1 medium) with answers based on this text:\n" + full_text)

def generate_flashcards(full_text):
    return ask_Ycloudai("Create 5 flashcards with Q & A based on this text:\n" + full_text)

@app.route("/YcloudSpark", methods=["POST"])
def ycloud_spark_processor():
    try:
        # 1️⃣ Check if files are uploaded
        files = request.files.getlist("files")
        if not files or all(f.filename == '' for f in files):
            return jsonify({"error": "Please select at least one valid file"}), 400

        # 2️⃣ Extract text from files
        all_text = ""
        for file in files:
            try:
                text, error = extract_text_from_file(file)
                if error:
                    return jsonify({"error": f"Error extracting text from {file.filename}: {error}"}), 400
                all_text += text.strip() + "\n"
            except Exception as e:
                return jsonify({"error": f"Unexpected error while reading {file.filename}: {str(e)}"}), 500

        # 3️⃣ Chunk the text
        try:
            text_chunks = chunk_text(all_text)
            if not text_chunks:
                return jsonify({"error": "No text extracted from the uploaded files"}), 400
        except Exception as e:
            return jsonify({"error": f"Error during text chunking: {str(e)}"}), 500

        # 4️⃣ Prepare optional inputs
        question = request.form.get("question")
        gq = request.form.get("quiz")
        flashcards = request.form.get("flashcards")

        # 5️⃣ Call AI functions safely
        result = {}
        try:
            result["summary"] = generate_summary(text_chunks)
        except Exception as e:
            result["summary_error"] = f"Error generating summary: {str(e)}"

        if question:
            try:
                result["answer"] = generate_qa(text_chunks, question)
            except Exception as e:
                result["answer_error"] = f"Error generating answer: {str(e)}"

        if gq and gq.lower() == "yes":
            try:
                result["quiz"] = generate_quiz(all_text)
            except Exception as e:
                result["quiz_error"] = f"Error generating quiz: {str(e)}"

        if flashcards and flashcards.lower() == "yes":
            try:
                result["flashcards"] = generate_flashcards(all_text)
            except Exception as e:
                result["flashcards_error"] = f"Error generating flashcards: {str(e)}"

        return jsonify(result)

    except Exception as e:
        # Catch any unexpected error
        return jsonify({"error": f"Unexpected server error: {str(e)}"}), 500


# ----------------------- Ycloud Mindmap -----------------------
@app.route("/ai/mindmap", methods=["POST"])
def mindmap_generator():
    text_input = request.form.get("text")
    file = request.files.get("file")
    if not text_input and not file:
        return jsonify({"error": "Please provide text or upload a file"}), 400

    if file:
        text_input, error = extract_text_from_file(file)
        if error:
            return jsonify({"error": error}), 400

    # The new, more detailed prompt
    prompt = (
        f"Analyze the following text and generate a hierarchical mindmap using Mermaid.js syntax.\n\n"
        f"**Instructions for the mindmap:**\n"
        f"1.  The mindmap should be clear, concise, and easy to understand every one has to understand the concept very well.\n"
        f"2.  For each key concept or node, add a relevant Unicode emoji for visual appeal  so that everyone can understand easily (e.g., 💡 for an idea, 📊 for data).\n"
        f"3.  For major branches or important nodes, also include a relevant icon from Font Awesome 5 using the syntax 'fa:fa-icon-name' (e.g., 'fa:fa-brain' for a central topic).\n\n"
        f"**Text to analyze:**\n{text_input}"
    )
    
    mindmap_syntax = ask_Ycloudai(prompt)
    return jsonify({"mindmap_code": mindmap_syntax})

ROLE_SKILLS = {
    "Data Scientist": [
        "Python", "R", "SQL", "Pandas", "NumPy", "Scikit-learn", "TensorFlow",
        "PyTorch", "Statistics", "Machine Learning", "Deep Learning", "NLP",
        "Data Visualization", "Matplotlib", "Seaborn"
    ],
    "Java Developer": [
        "Java", "Spring Boot", "Hibernate", "REST API", "Microservices", "Maven",
        "JUnit", "Docker", "SQL", "Git", "Eclipse", "IntelliJ", "OOP"
    ],
    "Python Developer": [
        "Python", "Django", "Flask", "FastAPI", "SQLAlchemy", "REST API",
        "Pandas", "NumPy", "Pytest", "Git", "Docker", "OOP"
    ],
    "Cloud DevOps Engineer": [
        "AWS", "Azure", "GCP", "Terraform", "Ansible", "Jenkins", "Docker",
        "Kubernetes", "Linux", "Bash", "Python", "Git", "CI/CD", "Monitoring",
        "CloudFormation"
    ],
    "DevOps Engineer": [
        "Jenkins", "Docker", "Kubernetes", "Ansible", "Terraform", "Linux",
        "Bash", "Git", "AWS", "CI/CD", "Monitoring", "Networking", "Python"
    ],
    "Data Analyst": [
        "Excel", "SQL", "Python", "Power BI", "Tableau", "Pandas", "NumPy",
        "Statistics", "Data Visualization", "Data Cleaning"
    ],
    "Data Engineer": [
        "Python", "SQL", "Spark", "Hadoop", "Kafka", "AWS", "GCP", "Azure",
        "ETL", "Airflow", "Databricks", "BigQuery", "Redshift", "Data Warehousing"
    ],
    "SAP Developer": [
        "SAP ABAP", "SAP HANA", "SAP Fiori", "SAP UI5", "SAP BW", "SAP NetWeaver",
        "OData Services", "SAP BusinessObjects"
    ],
    "Prompt Engineer": [
        "LLMs", "OpenAI", "HuggingFace", "LangChain", "Python", "NLP",
        "Prompt Design", "Fine-tuning", "RAG", "Vector Databases", "JSON"
    ]
}


def ats_score(resume_text: str, role: str):
    text = resume_text.lower()
    skills = ROLE_SKILLS.get(role, [])
    total = len(skills)
    threshold = total / 2

    matched = [s for s in skills if re.search(rf"\b{s.lower()}\b", text)]
    missing = [s for s in skills if s not in matched]
    M = len(matched)

    # Threshold scoring
    if M < threshold:
        score = (M / total) * 50
    else:
        score = 50 + ((M - threshold) / threshold) * 50

    return round(score, 2), matched, missing


def gemini_feedback(resume_text: str, role: str, matched: list, missing: list, score: float):
    prompt = f"""
    You are an ATS Resume Analyzer.
    Candidate applied for role: {role}.
    
    Resume Text:
    {resume_text[:1500]}  # limit text for safety
    
    Resume Score: {score}%
    Matched Skills: {matched}
    Missing Skills: {missing}
    
    Please provide output in this format:
    Highlights:
    - ...
    - ...
    Suggestions:
    - ...
    - ...
    """

    # CORRECTED LINE: Use the modern way to get a model.
    # 'gemini-1.5-flash-latest' is recommended for speed and low cost.
    model = genai.GenerativeModel('gemini-flash-latest')

    # This line is already correct for the new model object.
    response = model.generate_content(prompt)
    
    return response.text


# In your app.py, replace the old resume_builder function with this one

@app.route("/ai/resume_builder", methods=["POST"])
def resume_builder():
    try:
        file = request.files.get("file")
        selected_role = request.form.get("role")

        if not file or not selected_role:
            return jsonify({"error": "Please provide a file and select a role"}), 400

        text_input, error = extract_text_from_file(file)
        if error:
            return jsonify({"error": error}), 400

        score, matched, missing = ats_score(text_input, selected_role)
        ai_feedback = gemini_feedback(text_input, selected_role, matched, missing, score)

        return jsonify({
            "resume_score": score,
            "matched_skills": matched,
            "missing_skills": missing,
            "ai_feedback": ai_feedback
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def ask_Ycloudai_audio(audio_path, prompt="Transcribe this audio file accurately and clearly.", max_tokens=2048):
    """
    Sends an audio file to YCloud AI (Gemini) for transcription and returns the text.
    """
    try:
        model = genai.GenerativeModel('gemini-flash-latest')

        # Open audio file as bytes
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        # Send audio + prompt together
        response = model.generate_content(
            [prompt, {"mime_type": "audio/wav", "data": audio_bytes}],
            generation_config={"max_output_tokens": max_tokens}
        )

        return response.text.strip()
    except Exception as e:
        print(f"YCloud AI audio transcription error: {e}")
        return None



@app.route("/ai/voice_to_document", methods=["POST"])
def voice_to_document():
    if 'file' not in request.files:
        return jsonify({"error": "Please upload an audio file"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    format_type = request.form.get("format", "pdf")
    temp_path_webm = None
    temp_path_wav = None


    try:
        # 1️⃣ Save audio temporarily
        temp_filename_webm = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        temp_path_webm = os.path.join(UPLOAD_FOLDER, temp_filename_webm)
        file.save(temp_path_webm)


        temp_filename_wav = temp_path_webm.replace('.webm','.wav')
        temp_path_wav = os.path.join(UPLOAD_FOLDER,temp_filename_wav)

        subprocess.run(['ffmpeg', '-i', temp_path_webm, temp_path_wav], check=True)

        # 2️⃣ Send audio to YCloud AI for transcription
        text_output = ask_Ycloudai_audio(temp_path_wav)
        if not text_output:
            return jsonify({"error": "AI transcription failed"}), 500

        # 3️⃣ Save as PDF or DOCX
        if format_type.lower() == "pdf":
            output_file = f"{uuid.uuid4()}.pdf"
            pdf_doc = FPDF()
            pdf_doc.add_page()
            pdf_doc.set_font("Arial", size=12)
            pdf_doc.multi_cell(0, 10, text_output)

            # ✅ Correct way: get PDF as bytes
            pdf_bytes = pdf_doc.output(dest="S").encode("latin1")
            saved_path = save_file_locally(pdf_bytes, output_file, subfolder="document_tools")

        else:
            output_file = f"{uuid.uuid4()}.docx"
            doc = Document()
            doc.add_paragraph(text_output)
            output_buffer = io.BytesIO()
            doc.save(output_buffer)
            output_buffer.seek(0)
            saved_path = save_file_locally(output_buffer.getvalue(), output_file, subfolder="document_tools")

        download_url = f"/download/{saved_path}"
        return jsonify({"text": text_output, "download_url": download_url})

    except Exception as e:
        print(f"Voice-to-Document Error: {e}")
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500

    finally:
        
        if temp_path_webm and os.path.exists(temp_path_webm):
            os.remove(temp_path_webm)
        if temp_path_wav and os.path.exists(temp_path_wav):
            os.remove(temp_path_wav)





if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
