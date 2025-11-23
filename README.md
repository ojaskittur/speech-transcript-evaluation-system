#  Speech Transcript Evaluation System

### [ Live Demo: Click Here to Launch App](https://ojaskittur-speech-transcript-evaluation-system.hf.space)

##  Demo Video & Project Documentation
**[🔗 Click Here to View Demo Video & Process Document (Google Drive)](INSERT_YOUR_DRIVE_LINK_HERE)**

---

##  Project Overview
This project is an AI-powered tool designed to analyze and score spoken communication skills. It accepts a text transcript and audio duration as input and generates a comprehensive **rubric-based score (0-100)** with detailed feedback.

The system is designed to simulate a real-world grading environment, providing users with actionable insights on their speech rate, grammar, vocabulary, and content structure.

---

##  Tech Stack
* **Frontend:** Streamlit (Cloud Deployment), Custom HTML/JS (Local Version)
* **Backend:** FastAPI (Local API), Docker (Containerization)
* **NLP & ML:**
    * `spaCy`: Tokenization and NLP pipeline.
    * `sentence-transformers`: Semantic similarity and embedding analysis.
    * `vaderSentiment`: Sentiment and tone analysis.
    * `language-tool-python`: Java-based advanced grammar checking.

---

##  Scoring Logic & Formulas
The system uses a hybrid approach combining **Rule-Based Logic** and **NLP/Semantic Analysis**.

### 1. Salutation (5 points)
* **Logic:** Keyword matching for formal greetings (e.g., "Good morning", "Hello everyone").
* **Scoring:** Excellent (5pts), Good (4pts), Basic (2pts).

### 2. Content & Structure (30 points)
* **Method:** Hybrid (Regex + Semantic Embeddings).
* **Core Topics:** Name, Age, School, Family, Hobbies.
* **Implementation:** Uses `all-MiniLM-L6-v2` to compute cosine similarity between the transcript and target topics, ensuring the user stays on track even if they use different phrasing.

### 3. Speech Rate (10 points)
* **Formula:** $WPM = \frac{Total Words}{Duration (min)}$
* **Rubric:**
    * **Ideal:** 111-140 WPM (10 pts)
    * **Acceptable:** 81-160 WPM (6 pts)
    * **Poor:** <80 or >160 WPM (2 pts)

### 4. Grammar (10 points)
* **Method:** `language-tool-python`.
* **Logic:** Calculates error density per 100 words, penalizing critical grammatical errors while ignoring minor stylistic choices.

### 5. Vocabulary & Clarity (25 points)
* **Vocabulary:** Measured using **Type-Token Ratio (TTR)** to ensure word variety.
* **Clarity:** Detects filler words ("um", "uh", "like"). High filler frequency reduces the score.

### 6. Flow & Sentiment (20 points)
* **Flow:** Checks for a logical progression (Intro → Body → Closing) using semantic anchors.
* **Sentiment:** Analyzes the text for positive tone and enthusiasm using VADER.

---

##  How to Run Locally
This project includes a fully functional local backend (FastAPI) and frontend.

### Prerequisites
1.  **Python 3.11**
2.  **Java (JDK 17 or higher)** (Required for the Grammar Checker engine).

### Installation Steps
1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/ojaskittur/speech-transcript-evaluation-system.git](https://github.com/ojaskittur/speech-transcript-evaluation-system.git)
    cd speech-transcript-evaluation-system
    ```

2.  **Create Virtual Environment**
    ```bash
    python3.11 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

4.  **Run the Server**
    ```bash
    python api.py
    ```

5.  **Access the App**
    Open your browser to: **http://localhost:8000**

---

## Cloud Deployment (Docker)
The application is deployed on **Hugging Face Spaces** using a custom Docker container.

**Why Docker?**
Standard Python hosting environments often lack the Java runtime required for the `language-tool` library. A custom `Dockerfile` was used to:
1.  Install `default-jdk-headless` (Java Runtime).
2.  Install Python 3.11 and ML dependencies.
3.  Serve the Streamlit application on port 7860.
