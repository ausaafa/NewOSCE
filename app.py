from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from openai import OpenAI
from elevenlabs.client import ElevenLabs
import os, re, uuid, json
from PyPDF2 import PdfReader
from docx import Document
from flask import session, redirect, url_for
from functools import wraps

# -------------------- Flask & OpenAI --------------------

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-in-production")

USERNAME = "admin"
PASSWORD = "pharmacy2025"

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ---------------- OPENAI CONFIG ----------------

OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
DEEPGRAM_API_KEY   = os.environ.get("DEEPGRAM_API_KEY")

client    = OpenAI(api_key=OPENAI_API_KEY)
el_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# -------------------- Global Case / Conversation State --------------------

case_context = {
    "raw": "",
    "facts": {},
    "summary": "",
    "persona": "",
    "gender": ""
}

patient_state = {
    "summary": "",
    "turns": []
}

MAX_TURNS = 8


def extract_references(text: str) -> str:
    # Known clinical/pharmacy reference sources — ONLY these are valid references
    keywords = [
        "Health Canada", "CPS", "e-CPS", "ECPS",
        "Compendium of Pharmaceuticals", "Product Monograph",
        "FDA", "UpToDate", "Lexicomp", "NAPRA", "ISMP", "RxTx",
        "CTMA", "Pharmacists Association", "Canadian Pharmacists",
        "Natural Health Products", "Therapeutic Choices"
    ]

    # 1. Look for an explicit "References:" line and extract only known keywords from it
    m = re.search(r"References?\s*[:\-]\s*([^\n]{3,200})", text, re.I)
    if m:
        ref_line = m.group(1).strip()
        # Only keep the portion before any exam/candidate instruction words
        ref_line = re.split(
            r"(?i)\b(candidate|instructions|station|timeframe|kindly|counsel|advise|exam|profile|see below|information|checklist|solved|unsolved|marginally)\b",
            ref_line
        )[0].strip().rstrip('.,; ')
        if ref_line and len(ref_line) > 2 and len(ref_line) < 150:
            return ref_line

    # 2. Scan full text for known clinical reference keywords only
    found = []
    for k in keywords:
        if re.search(r'\b' + re.escape(k) + r'\b', text, re.I):
            found.append(k)

    return '\n'.join(found) if found else ""


def extract_text(file_path: str) -> str:
    ext = file_path.lower()
    if ext.endswith(".pdf"):
        text = ""
        with open(file_path, "rb") as f:
            pdf = PdfReader(f)
            for p in pdf.pages:
                text += p.extract_text() or ""
        return text
    elif ext.endswith(".docx"):
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    elif ext.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return ""


def extract_case_info(text: str) -> dict:
    prompt = f"""
Extract patient information from this pharmacy/OSCE case file.
Return ONLY valid JSON with these exact keys (omit any key if not found):
- name (string, patient full name only)
- age (integer, numbers only)
- gender (string: "male" or "female" only)
- complaint (string, 1 short sentence describing why they came to the pharmacy)
- diagnosis (string, medical condition name only, no instructions or exam directions)
- medications (list of strings, drug names and doses only)
- allergies (list of strings)

STRICT RULES:
- Do NOT include examiner instructions, OSCE directions, or any text meant for the examiner
- Do NOT include partial sentences or text after a period that continues as instructions
- diagnosis must be a condition name only (e.g. "Hypertension", "ADHD") — never include phrases like "counsel her" or "advise patient"
- If a field is unclear or contaminated with instructions, omit it entirely

CASE TEXT:
{text[:3000]}
"""
    try:
        raw = chat_once(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=400
        )
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        return json.loads(clean)
    except:
        return {}

def infer_gender_from_name(name: str) -> str:
    if not name:
        return ""
    first = name.split()[0].lower()
    female = {"jessica", "emily", "sarah", "olivia", "emma", "sophia", "isabella", "ava", "mia", "ella", "jess"}
    male = {"mike", "michael", "john", "james", "robert", "william", "david", "daniel", "matthew", "joseph"}
    if first in female: return "female"
    if first in male: return "male"
    return ""


def clamp_turns():
    if len(patient_state["turns"]) > MAX_TURNS:
        patient_state["turns"] = patient_state["turns"][-MAX_TURNS:]


def chat_once(msgs, **kwargs):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=msgs,
        **kwargs
    )
    return resp.choices[0].message.content.strip()


# -------------------- CORE ROUTES --------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    return redirect(url_for("home"))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# -------------------- Start Session (PATIENT SPEAKS FIRST) --------------------

@app.route("/start-session", methods=["POST"])
def start_session():
    global case_context, patient_state

    system_prompt = f"""
You are roleplaying the person visiting the pharmacy as described in the case.
This may be the patient themselves, OR a caregiver/parent visiting on behalf of someone else.
Read the background carefully and speak as THAT person — not the person they are concerned about.

For your OPENING greeting only, keep it very short and natural — just introduce yourself and say why you are here in ONE simple sentence.
Use the actual name from FACTS if available. Example: "Hi, my name is [actual name from FACTS], I'm here because [brief reason]."
Do NOT give details, symptoms, or full story yet — wait for the pharmacist to ask questions.

CRITICAL RULES:
1. ONLY use real information explicitly stated in FACTS. NEVER invent personal details.
2. NEVER output placeholder text like [Your Name], [City], [Address], [Your Location], or ANY bracketed variables.
3. If your name is in FACTS, use it naturally. If not in FACTS, say "I'd rather not say."
4. Stay in character as the visitor at all times.

PERSONA: {case_context['persona']}
FACTS: {case_context['facts']}
BACKGROUND: {case_context['summary']}
"""

    greeting = chat_once(
        [{"role": "system", "content": system_prompt}],
        temperature=0.5,
        max_tokens=60
    )

    patient_state["turns"] = [{"role": "assistant", "content": greeting}]

    return jsonify({"greeting": greeting})


# -------------------- Upload --------------------

@app.route("/upload", methods=["POST"])
def upload_case():
    global case_context, patient_state

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    filename = secure_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    text = extract_text(path)
    if not text:
        return jsonify({"error": "Could not read file"}), 400

    facts = extract_case_info(text)
    if "gender" not in facts or not facts["gender"]:
        inferred = infer_gender_from_name(facts.get("name", ""))
        if inferred:
            facts["gender"] = inferred

    summary = chat_once(
        [
            {"role": "system", "content": "Write a brief first-person patient background (1–2 sentences)."},
            {"role": "user", "content": text}
        ],
        temperature=0.3
    )

    persona = chat_once(
        [
            {"role": "system", "content": "Describe the patient's tone in <=2 short lines."},
            {"role": "user", "content": text}
        ],
        temperature=0.5
    )

    summary_prompt = [
        {"role": "system", "content": "Extract a 1–2 sentence OSCE case summary."},
        {"role": "user", "content": text}
    ]

    case_summary = chat_once(summary_prompt, temperature=0.3)
    references = extract_references(text)

    case_context = {
        "raw": text,
        "facts": facts,
        "summary": summary,
        "persona": persona,
        "gender": (facts.get("gender") or "").lower()
    }
    patient_state = {
        "summary": "",
        "turns": []
    }

    patient_state["turns"].append({
        "role": "assistant",
        "content": summary
    })

    if references:
        patient_state["turns"].append({
            "role": "assistant",
            "content": f"References: {references}"
        })

    return jsonify({
        "message": "Case uploaded successfully.",
        "case_summary": case_summary,
        "summary": summary,
        "persona": persona,
        "extracted": facts,
        "references": references or ""
    })


# -------------------- ASK --------------------

@app.route("/ask", methods=["POST"])
def ask():
    global case_context, patient_state

    data = request.get_json(silent=True) or {}
    user_q = (data.get("question") or "").strip()

    if not user_q:
        return jsonify({"error": "No question"}), 400

    turns_preview = patient_state["turns"][-6:]

    system_prompt = f"""
You are roleplaying the person visiting the pharmacy as described in the case.
This may be the patient themselves OR a caregiver. Speak naturally and stay in character.

CRITICAL RULES:
1. FACTS below contain the real patient information — answer ANY question about name, age, medications, allergies, diagnosis, or complaints using FACTS directly and naturally. Never say "I'd rather not say" for information that IS in FACTS.
2. Only decline to answer if the information is genuinely NOT anywhere in FACTS or BACKGROUND (e.g. home address, phone number, postal code).
3. NEVER output placeholder text like [Your Name], [City], or ANY bracketed variables.
4. If asked something completely irrelevant to the pharmacy visit (politics, random facts, etc.): redirect politely.
5. Never break character or acknowledge you are an AI.
6. Keep responses VERY SHORT — one sentence only. Answer what was asked and stop.

PERSONA: {case_context['persona']}
FACTS: {case_context['facts']}
BACKGROUND: {case_context['summary']}
"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(turns_preview)
    messages.append({"role": "user", "content": user_q})

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.4,
        max_tokens=40
    )
    answer = completion.choices[0].message.content.strip()

    patient_state["turns"].append({"role": "user", "content": user_q})
    patient_state["turns"].append({"role": "assistant", "content": answer})
    clamp_turns()

    return jsonify({"answer": answer})


# -------------------- TTS (ElevenLabs Flash) --------------------

@app.route("/tts", methods=["POST"])
def tts():
    text = request.json.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    gender = case_context.get("gender", "").lower()
    # ElevenLabs pre-made voices — Flash v2.5 model for lowest latency
    if gender == "female":
        voice_id = "EXAVITQu4vr4xnSDxMaL"   # Sarah — warm, natural female
    elif gender == "male":
        voice_id = "TX3LPaxmHKxFdv7VOQHJ"   # Liam — conversational male
    else:
        voice_id = "pqHfZKP75CvOlQylNhV4"   # Bill — neutral fallback

    audio_filename = f"voice_{uuid.uuid4().hex}.mp3"
    audio_path = os.path.join(app.config["UPLOAD_FOLDER"], audio_filename)

    audio = el_client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id="eleven_flash_v2_5",
        output_format="mp3_44100_128"
    )

    with open(audio_path, "wb") as f:
        for chunk in audio:
            f.write(chunk)

    return jsonify({"audio": f"/uploads/{audio_filename}", "ready": True})

# -------------------- LIST CHAPTERS --------------------

@app.route("/list-chapters")
def list_chapters():
    base_path = os.path.join(BASE_DIR, "Chapters")

    if not os.path.exists(base_path):
        return jsonify({"error": "Chapters folder not found"}), 404

    result = {}

    for chapter in os.listdir(base_path):
        chapter_path = os.path.join(base_path, chapter)

        if os.path.isdir(chapter_path):
            files = []
            for f in os.listdir(chapter_path):
                if f.lower().endswith((".txt", ".pdf", ".docx")):
                    files.append(f)
            result[chapter] = files

    return jsonify(result)

# -------------------- LOAD DEFAULT CASE --------------------

@app.route("/load-default-case", methods=["POST"])
def load_default_case():
    global case_context, patient_state

    data = request.get_json()
    chapter = data.get("chapter")
    filename = data.get("file")

    if not chapter or not filename:
        return jsonify({"error": "Invalid request"}), 400

    CHAPTERS_DIR = os.path.join(BASE_DIR, "Chapters")
    full_path = os.path.join(CHAPTERS_DIR, chapter, filename)

    if not os.path.exists(full_path):
        return jsonify({"error": "Case file not found"}), 404

    try:
        text = extract_text(full_path)
    except:
        return jsonify({"error": "Failed to read file"}), 500

    facts = extract_case_info(text)

    if "gender" not in facts or not facts["gender"]:
        inferred = infer_gender_from_name(facts.get("name", ""))
        if inferred:
            facts["gender"] = inferred

    summary = chat_once(
        [
            {"role": "system", "content": "Write a brief first-person patient background (1–2 sentences)."},
            {"role": "user", "content": text}
        ],
        temperature=0.3
    )

    persona = chat_once(
        [
            {"role": "system", "content": "Describe the patient's tone in <=2 short lines."},
            {"role": "user", "content": text}
        ],
        temperature=0.5
    )

    case_summary = chat_once(
        [
            {"role": "system", "content": "Extract a 1–2 sentence OSCE case summary."},
            {"role": "user", "content": text}
        ],
        temperature=0.3
    )
    references = extract_references(text)

    case_context = {
        "raw": text,
        "facts": facts,
        "summary": summary,
        "persona": persona,
        "gender": (facts.get("gender") or "").lower()
    }
    patient_state = {"summary": "", "turns": []}

    return jsonify({
        "case_summary": case_summary,
        "summary": summary,
        "persona": persona,
        "extracted": facts,
        "references": references
    })

@app.route("/auto-greet", methods=["POST"])
def auto_greet():
    global case_context, patient_state

    system_prompt = f"""
You are the patient visiting the pharmacy. Provide a simple, natural greeting in exactly 1 sentence.
Use the actual name from FACTS if available — NEVER use placeholder text like [Your Name].

Examples:
- "Hi, I'm Sarah — I'm here because I've been having some concerns about my medication."
- "Hello, I've been feeling unwell and wanted to ask the pharmacist some questions."
- "Hi there, I have some concerns about a prescription I was given recently."

CRITICAL: Only use real information from FACTS. NEVER generate placeholder text in brackets.

PERSONA: {case_context['persona']}
FACTS: {case_context['facts']}
BACKGROUND: {case_context['summary']}
"""

    greeting = chat_once(
        [{"role": "system", "content": system_prompt}],
        temperature=0.5,
        max_tokens=40
    )

    patient_state["turns"] = [{"role": "assistant", "content": greeting}]

    return jsonify({"greeting": greeting})


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# -------------------- RESULTS --------------------

@app.route("/results", methods=["GET"])
def results():
    global case_context, patient_state

    turns = patient_state.get("turns", [])

    if not turns:
        return jsonify({"error": "No session data found."}), 400

    pharmacist_turns = [t for t in turns if t["role"] == "user"]

    if len(pharmacist_turns) == 0:  # no actual pharmacist input
        return jsonify({
            "good": [],
            "improvement": [
                "No interaction occurred.",
                "The pharmacist did not engage the patient.",
                "Begin with open-ended questions to initiate the consultation."
            ],
            "listening": 0,
            "empathy": 0,
            "communication": 0,
            "problem_solving": 0
        })

    # Only include turns from the actual session (exclude the auto-added
    # background summary and references that are prepended at case load time)
    session_turns = [
        t for t in turns
        if not (t["role"] == "assistant" and (
            t["content"].startswith("References:") or
            len(t["content"]) > 200  # long AI messages are setup summaries, not patient dialogue
        ))
    ]

    transcript = ""
    for t in session_turns:
        role = "Pharmacist" if t["role"] == "user" else "Patient"
        transcript += f"{role}: {t['content']}\n"

    prompt = f"""
You are a strict OSCE examiner. Each score MUST be different unless performance is genuinely identical.
Scores should reflect real differences — a pharmacist who only asked one question should score very differently across categories.

LISTENING:
0-10 = Said nothing, no questions at all
11-20 = One very basic or closed question
21-35 = Asked about the main concern but no follow-up
36-50 = Some follow-up but missed key details
51-65 = Good exploration of most concerns
66-80 = Thorough questioning with good follow-up
81-90 = Excellent active listening throughout
91-100 = Outstanding — explored all concerns, clarified ambiguities, reflected back

EMPATHY:
0-10 = No acknowledgement of patient emotions whatsoever
11-20 = One word response, no warmth
21-35 = Minimal warmth but no real validation
36-50 = One generic empathetic phrase
51-65 = Clear acknowledgement of patient feelings
66-80 = Warm, validating language used consistently
81-90 = Strong emotional support and reassurance
91-100 = Exceptional empathy — patient clearly felt heard and supported

COMMUNICATION:
0-10 = No explanations given at all
11-20 = One vague or confusing statement
21-35 = Minimal information, poorly structured
36-50 = Some information but incomplete or unclear
51-65 = Reasonable explanation with gaps
66-80 = Clear and organized explanations
81-90 = Excellent clarity, logical structure, appropriate language
91-100 = Outstanding communication — thorough, clear, patient-centered

PROBLEM SOLVING:
0-10 = No advice, no plan, no clinical information provided
11-20 = Acknowledged the problem but gave no guidance
21-35 = Very vague suggestion without any real plan
36-50 = Partial plan mentioned but missing key elements
51-65 = Reasonable plan but incomplete
66-80 = Appropriate and safe clinical plan discussed
81-90 = Clear, complete, patient-centered plan with follow-up
91-100 = Outstanding — comprehensive plan, safety considerations, follow-up, alternatives discussed

CASE CONTEXT:
{case_context['summary']}

TRANSCRIPT:
{transcript}

Read the transcript carefully. Base every score ONLY on what the pharmacist explicitly said.
Scores must vary — if problem solving was weak but listening was decent, reflect that gap clearly.

Return ONLY valid JSON with:
- good (list of exactly 2 specific strengths shown by the PHARMACIST only)
- improvement (list of exactly 2 specific weaknesses of the PHARMACIST only)
- listening (integer 0-100)
- empathy (integer 0-100)
- communication (integer 0-100)
- problem_solving (integer 0-100)
"""

    try:
        raw = chat_once(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )

        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        data = json.loads(clean)

        return jsonify(data)

    except Exception as e:
        return jsonify({"error": f"Failed to generate results: {str(e)}"}), 500

# -------------------- DEEPGRAM TOKEN --------------------
@app.route("/deepgram-token", methods=["GET"])
def deepgram_token():
    # Returns the Deepgram API key to the browser so it can open a WebSocket.
    # Route is login-protected so the key is never exposed publicly.
    return jsonify({"key": DEEPGRAM_API_KEY})

# -------------------- RESET CASE --------------------

@app.route("/reset-case", methods=["POST"])
def reset_case():
    global case_context, patient_state
    case_context = {"raw": "", "facts": {}, "summary": "", "persona": "", "gender": ""}
    patient_state = {"summary": "", "turns": []}
    return jsonify({"message": "Case reset successfully"})


# -------------------- Run --------------------
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5002,
        debug=False
    )