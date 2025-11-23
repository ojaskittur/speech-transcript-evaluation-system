import os
import re
import spacy
import numpy as np
import language_tool_python
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textstat import textstat

# ==========================================
# 1. ENVIRONMENT & MODEL SETUP
# ==========================================

# Ensure Java is accessible for LanguageTool (Backend fallback)
# Adjust this path if your server location is different
if os.path.exists("/usr/lib/jvm/java-17-openjdk-amd64"):
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"

print("Loading models... this may take a moment.")

# Load Spacy (with auto-download fallback)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model not found. Downloading...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load AI Models
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
grammar_tool = language_tool_python.LanguageTool('en-US')
sentiment_analyzer = SentimentIntensityAnalyzer()
# CrossEncoder for potentially deeper NLI tasks (loaded for future-proofing/robustness)
nli_model = CrossEncoder('cross-encoder/stsb-distilroberta-base') 

print("Models loaded successfully.")

# ==========================================
# 2. SCORING RUBRIC CONFIGURATION
# ==========================================

RUBRIC = {
    "salutation": {
        "normal": ["hi", "hello"],
        "good": ["good morning", "good afternoon", "good evening", "good day", "hello everyone"],
        "excellent": ["excited to introduce", "feeling great", "pleasure to introduce", "greetings"]
    },
    "content": {
        "must_have": {
            "points": 4, 
            "topics": ["Name", "Age", "School/Class", "Family", "Hobbies/Interests"]
        },
        "good_to_have": {
            "points": 2, 
            "topics": ["Origin/Location", "Ambition/Goal", "Fun Fact/Unique", "Strengths", "Achievements"]
        }
    },
    "speech_rate": {
        "fast_threshold": 160,
        "ideal_min": 111,
        "ideal_max": 140,
        "slow_threshold": 80
    },
    "fillers": ["um", "uh", "like", "you know", "actually", "basically", "right", "i mean", "well", "kinda", "sort of", "hmm"]
}

# ==========================================
# 3. MAIN LOGIC CLASS
# ==========================================

class IntroductionScorer:
    def __init__(self, transcript_text, audio_duration_sec=None):
        self.text = transcript_text
        self.doc = nlp(transcript_text)
        self.provided_duration = float(audio_duration_sec) if audio_duration_sec else 0
        
        self.duration_min = (self.provided_duration / 60) if self.provided_duration else 0
        self.sentences = [sent.text.strip() for sent in self.doc.sents]
        self.words = [token.text.lower() for token in self.doc if not token.is_punct]
        self.total_words = len(self.words)

    def score_salutation(self):
        text_lower = self.text.lower()
        
        for phrase in RUBRIC["salutation"]["excellent"]:
            if phrase in text_lower:
                return 5, f"Excellent salutation used: '{phrase}'"
        
        for phrase in RUBRIC["salutation"]["good"]:
            if phrase in text_lower:
                return 4, f"Good salutation used: '{phrase}'"
                
        for word in RUBRIC["salutation"]["normal"]:
            if word in text_lower:
                return 2, "Basic salutation used (Hi/Hello). Try to be more formal."
                
        return 0, "No salutation found."

    def score_content(self):
        scores = 0
        feedback = []

        # --- Regex Checks for Specific Facts ---
        regex_name = r"\b(name\s+is|i\s+am|i[\s'’]*m|myself|this\s+is)\s+([A-Z])"
        regex_age = r"\b(\d+|thirteen|fourteen|fifteen|sixteen)\s*(-)?\s*(years|yrs)\b"
        regex_school = r"\b(class|grade|standard|school|college|university|study|student)\b"

        if re.search(regex_name, self.text, re.IGNORECASE):
            scores += 4; feedback.append("[+] Name")
        else: feedback.append("[-] Name")

        if re.search(regex_age, self.text, re.IGNORECASE):
            scores += 4; feedback.append("[+] Age")
        else: feedback.append("[-] Age")

        if re.search(regex_school, self.text, re.IGNORECASE):
            scores += 4; feedback.append("[+] School")
        else: feedback.append("[-] School")

        # --- Robust Semantic Checks (Regex + Embeddings) ---
        def check_topic_robust(regex, anchors, use_ai=True):
            # 1. Fast Regex Check
            if re.search(regex, self.text, re.IGNORECASE): return True
            
            # 2. Deep Semantic Check
            if use_ai and self.sentences:
                topic_emb = sbert_model.encode(anchors, convert_to_tensor=True)
                text_emb = sbert_model.encode(self.sentences, convert_to_tensor=True)
                # Find max similarity between any sentence and topic anchors
                best_score = float(util.cos_sim(text_emb, topic_emb).max())
                return best_score > 0.35
            return False

        # Family Check
        if check_topic_robust(r"\b(family|parents|mother|father|siblings)\b", ["My family", "I live with"]):
            scores += 4; feedback.append("[+] Family")
        else: feedback.append("[-] Family")

        # Hobbies Check
        if check_topic_robust(r"\b(hobby|hobbies|enjoy|like\s+(to|playing|reading)|pastime)\b", ["My hobby is", "I enjoy"]):
            scores += 4; feedback.append("[+] Hobbies")
        else: feedback.append("[-] Hobbies")

        # --- Bonus Checks ---
        bonuses = {
            "Ambition": (r"\b(goal|ambition|dream|want\s+to\s+be)\b", ["I want to become"], True),
            "Strength": (r"\b(strength|good\s+at|confident)\b", ["My strength is"], True),
            "Unique": (r"\b(unique|special|fun\s+fact)\b", ["fun fact"], True),
            "Origin": (r"\b(i\s+am\s+from|i['’]m\s+from|originally\s+from|live\s+in|living\s+in|born\s+in|hometown|native)\b", [], False),
            "Achievements": (r"\b(won|achievement|award)\b", ["I won"], True)
        }

        for topic, (reg, anc, use_ai_flag) in bonuses.items():
            if check_topic_robust(reg, anc, use_ai=use_ai_flag):
                scores += 2; feedback.append(f"[+] {topic}")

        return min(30, scores), ", ".join(feedback)

    def score_flow(self):
        anchors = {
            "salutation": ["Hello everyone", "Good morning", "Hi", "Greetings"],
            "intro": ["My name is", "I am", "I'm", "I’m", "Myself", "This is"],
            "closing": ["Thank you", "Thanks", "That is all", "The end"],
            "body": ["family", "mother", "school", "class", "hobby", "playing", "dream", "goal"]
        }
        
        if not self.sentences: return 0, "No text"
        
        text_emb = sbert_model.encode(self.sentences, convert_to_tensor=True)
        
        def get_idx(key, thresh=0.25):
            anc = sbert_model.encode(anchors[key], convert_to_tensor=True)
            sims = util.cos_sim(text_emb, anc).max(dim=1).values
            best_idx = int(sims.argmax())
            best_score = float(sims.max())
            return best_idx, best_score > thresh

        idx_s, has_s = get_idx("salutation", 0.25)
        idx_i, has_i = get_idx("intro", 0.25)
        idx_c, has_c = get_idx("closing", 0.30) 
        
        # Check if there is "meat" between intro and closing
        has_body = False
        if has_i and has_c and idx_c > idx_i: 
             if idx_c - idx_i >= 1:
                 mid_sents = self.sentences[idx_i+1 : idx_c]
                 if mid_sents:
                     mid_emb = sbert_model.encode(mid_sents, convert_to_tensor=True)
                     bod_emb = sbert_model.encode(anchors["body"], convert_to_tensor=True)
                     if util.cos_sim(mid_emb, bod_emb).max() > 0.25: has_body = True

        debug_info = f"(Indices: Sal={idx_s if has_s else 'X'}, Intro={idx_i if has_i else 'X'}, End={idx_c if has_c else 'X'})"

        if has_s and has_c:
            if has_i:
                if idx_s <= idx_i < idx_c:
                    return (5, "Perfect Flow") if has_body else (5, "Good Flow (Short body)")
                if idx_i == idx_c:
                    return 0, f"Disordered: Introduction and Closing are detected in same sentence. {debug_info}"
            
            elif idx_s < idx_c:
                return (5, "Good Flow") if has_body else (5, "Acceptable Flow")
                
        return 0, f"Flow disordered. {debug_info}"

    def score_speech_rate(self):
        if not self.provided_duration:
            return 10, "Duration not provided (Assumed Ideal)"
            
        wpm = self.total_words / self.duration_min if self.duration_min > 0 else 0
        
        if 111 <= wpm <= 140: return 10, f"Ideal ({int(wpm)} WPM)"
        if 81 <= wpm <= 160: return 6, f"Acceptable ({int(wpm)} WPM)"
        if wpm > 140: return 2, f"Too Fast ({int(wpm)} WPM)"
        if wpm < 81: return 2, f"Too Slow ({int(wpm)} WPM)"
        
        return 2, f"Poor Pacing ({int(wpm)} WPM)"

    def score_grammar(self):
        try:
            matches = grammar_tool.check(self.text)
            scoring_errors = []
            ignored_issues = []
            
            # --- Intelligent Filtering of Errors ---
            for m in matches:
                rid = getattr(m, 'ruleId', '').upper()
                msg = getattr(m, 'message', '').lower()
                replacements = getattr(m, 'replacements', [])
                
                offset = getattr(m, 'offset', 0)
                length = getattr(m, 'errorLength', getattr(m, 'length', 5))
                error_text = self.text[offset : offset + length]
                
                is_ignored = False
                
                # Ignore hyphenation suggestions if only one hyphen is missing
                if replacements:
                    top_rep = replacements[0]
                    if "-" in top_rep and top_rep.replace("-", "") == error_text.replace(" ", ""):
                        is_ignored = True
                
                # Ignore stylistic choices often flagged by strict grammar tools
                ignore_keywords = [
                    "hyphen", "compound", "joined", "whitespace", "comma", "punctuation",
                    "spelling", "typo", "morfologik", "uppercase", "capitalization",
                    "repetition", "consecutive", "successive", "same word", 
                    "style", "wordiness", "sentence start", "rewording", "thesaurus"
                ]
                
                if any(k in msg or k in rid.lower() for k in ignore_keywords):
                    is_ignored = True

                if is_ignored: ignored_issues.append(m)
                else: scoring_errors.append(m)
            
            # --- Scoring Calculation ---
            err_count = len(scoring_errors)
            errors_per_100 = (err_count / self.total_words) * 100 if self.total_words > 0 else 0
            
            # Conservative penalty
            grammar_metric = 1 - min(errors_per_100 / 5, 1)
            
            if grammar_metric > 0.9: s=10; g="Flawless"
            elif grammar_metric >= 0.7: s=8; g="Good"
            elif grammar_metric >= 0.5: s=6; g="Average"
            elif grammar_metric >= 0.3: s=4; g="Needs Improvement"
            else: s=2; g="Poor"
            
            # --- Feedback Formatting ---
            fb_lines = []
            fb_lines.append(f"{g} (Score: {s}/10)")
            fb_lines.append("NOTE: Spelling, hyphens, punctuation, and style ignored.")
            
            if scoring_errors:
                fb_lines.append(f"\n[CRITICAL GRAMMAR ERRORS] ({len(scoring_errors)} found):")
                for m in scoring_errors[:3]: # Limit to top 3
                    off = getattr(m, 'offset', 0)
                    ln = getattr(m, 'errorLength', getattr(m, 'length', 5))
                    ctx = self.text[off : off+ln+10].replace('\n', ' ')
                    fb_lines.append(f"   - {m.message} (Context: '...{ctx}...')")
            else:
                fb_lines.append("\n[CRITICAL GRAMMAR ERRORS]: None.")

            if ignored_issues:
                fb_lines.append(f"\n[IGNORED ISSUES] ({len(ignored_issues)} found):")
                for m in ignored_issues[:3]:
                    msg = getattr(m, 'message', 'Issue')
                    off = getattr(m, 'offset', 0)
                    ln = getattr(m, 'errorLength', getattr(m, 'length', 5))
                    ctx = self.text[off : off+ln+10].replace('\n', ' ')
                    fb_lines.append(f"   - {msg} (Context: '...{ctx}...')")
            
            return s, "\n".join(fb_lines)
            
        except Exception as e:
            return 5, f"Error during grammar check: {str(e)}"

    def score_vocabulary(self):
        distinct_words = len(set(self.words))
        ttr = distinct_words / self.total_words if self.total_words > 0 else 0
        
        if ttr >= 0.9: return 10, f"Excellent variety (TTR: {ttr:.2f})"
        elif ttr >= 0.7: return 8, f"Good variety (TTR: {ttr:.2f})"
        elif ttr >= 0.5: return 6, f"Average variety (TTR: {ttr:.2f})"
        elif ttr >= 0.3: return 4, f"Repetitive (TTR: {ttr:.2f})"
        else: return 2, f"Very repetitive (TTR: {ttr:.2f})"

    def score_clarity(self):
        filler_count = 0
        for word in self.words:
            if word in RUBRIC["fillers"]:
                filler_count += 1
        
        filler_rate = (filler_count / self.total_words) * 100 if self.total_words > 0 else 0
        
        if filler_rate <= 3: return 15, f"Clear speech ({filler_count} fillers)"
        elif filler_rate <= 6: return 12, f"Mostly clear ({filler_count} fillers)"
        elif filler_rate <= 9: return 9, f"Some hesitation ({filler_count} fillers)"
        elif filler_rate <= 12: return 6, f"Hesitant ({filler_count} fillers)"
        else: return 3, f"Distracted by fillers ({filler_count} fillers)"

    def score_engagement(self):
        vs = sentiment_analyzer.polarity_scores(self.text)
        
        # Normalize compound score (-1 to 1) to (0 to 1)
        prob = (vs['compound'] + 1) / 2
        
        high_energy_kws = [
            "excited", "thrilled", "passionate", "delighted", "honor", 
            "love", "amazing", "wonderful", "fantastic", "energetic",
            "grateful", "confident", "pleasure"
        ]
        
        has_enthusiasm = any(w in self.text.lower() for w in high_energy_kws)
        
        # Cap sentiment if it's high but lacks enthusiastic vocabulary
        if prob >= 0.9 and not has_enthusiasm:
            prob = 0.88 
            
        if prob >= 0.9:
            return 15, f"Very Engaging (Sentiment: {prob:.2f})"
        elif prob >= 0.7:
            return 12, f"Positive (Sentiment: {prob:.2f})"
        elif prob >= 0.5:
            return 9, f"Neutral (Sentiment: {prob:.2f})"
        elif prob >= 0.3:
            return 6, f"Slightly Negative (Sentiment: {prob:.2f})"
        else:
            return 3, f"Negative (Sentiment: {prob:.2f})"

    def calculate_overall_score(self):
        s_salutation, f_salutation = self.score_salutation()
        s_content, f_content = self.score_content()
        s_flow, f_flow = self.score_flow()
        s_rate, f_rate = self.score_speech_rate()
        s_grammar, f_grammar = self.score_grammar()
        s_vocab, f_vocab = self.score_vocabulary()
        s_clarity, f_clarity = self.score_clarity()
        s_engage, f_engage = self.score_engagement()
        
        total_score = (
            s_salutation + s_content + s_flow + s_rate + 
            s_grammar + s_vocab + s_clarity + s_engage
        )
        
        return {
            "Total Score": total_score,
            "Breakdown": {
                "Salutation": {"score": s_salutation, "max": 5, "feedback": f_salutation},
                "Content & Structure": {"score": s_content, "max": 30, "feedback": f_content},
                "Flow": {"score": s_flow, "max": 5, "feedback": f_flow},
                "Speech Rate": {"score": s_rate, "max": 10, "feedback": f_rate},
                "Grammar": {"score": s_grammar, "max": 10, "feedback": f_grammar},
                "Vocabulary": {"score": s_vocab, "max": 10, "feedback": f_vocab},
                "Clarity (Fillers)": {"score": s_clarity, "max": 15, "feedback": f_clarity},
                "Engagement": {"score": s_engage, "max": 15, "feedback": f_engage},
            }
        }