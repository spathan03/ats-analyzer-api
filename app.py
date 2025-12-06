from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import json
import traceback
import uuid
from datetime import datetime
import sys
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
from docx import Document

app = Flask(__name__)
CORS(app)

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk_stopwords = set(stopwords.words('english'))
except:
    nltk_stopwords = set()

# Configuration
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

class DocumentReader:
    """Simple document reader without heavy dependencies"""
    
    @staticmethod
    def read_pdf(filepath: str) -> str:
        """Read PDF file using PyPDF2"""
        try:
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""
    
    @staticmethod
    def read_docx(filepath: str) -> str:
        """Read DOCX file"""
        try:
            doc = Document(filepath)
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
            return text.strip()
        except Exception as e:
            print(f"Error reading DOCX: {e}")
            return ""
    
    @staticmethod
    def read_txt(filepath: str) -> str:
        """Read TXT file"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error reading TXT: {e}")
            return ""
    
    @staticmethod
    def read_document(filepath: str) -> str:
        """Auto-detect and read document"""
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.pdf':
            return DocumentReader.read_pdf(filepath)
        elif ext == '.docx':
            return DocumentReader.read_docx(filepath)
        elif ext == '.txt':
            return DocumentReader.read_txt(filepath)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

class ResumeAnalyzer:
    """Lightweight resume analyzer without spaCy"""
    
    def __init__(self):
        self.technical_skills = {
            'python', 'java', 'javascript', 'c++', 'c#', 'sql', 'aws', 'azure', 'gcp',
            'docker', 'kubernetes', 'react', 'angular', 'vue', 'nodejs', 'django', 'flask',
            'fastapi', 'spring', 'microservices', 'machine learning', 'deep learning', 'ai',
            'data science', 'analytics', 'pandas', 'numpy', 'scikit-learn', 'pytorch',
            'tensorflow', 'spark', 'hadoop', 'kafka', 'airflow', 'tableau', 'power bi',
            'git', 'ci/cd', 'devops', 'rest', 'graphql', 'api', 'linux', 'bash',
            'postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch', 'jira'
        }
        
        self.soft_skills = {
            'leadership', 'communication', 'teamwork', 'problem solving',
            'analytical', 'creative', 'adaptable', 'organized', 'detail-oriented',
            'collaborative', 'strategic', 'innovative', 'motivated', 'ownership',
            'stakeholder management', 'mentoring', 'presentation', 'negotiation'
        }
    
    def extract_keywords(self, text: str, top_n: int = 30) -> list:
        """Extract keywords using NLTK"""
        # Convert to lowercase and tokenize
        words = word_tokenize(text.lower())
        
        # Remove stopwords and non-alphabetic tokens
        words = [word for word in words 
                if word.isalpha() and word not in nltk_stopwords and len(word) > 2]
        
        # Get frequency distribution
        freq_dist = Counter(words)
        
        # Return top keywords
        return [word for word, _ in freq_dist.most_common(top_n)]
    
    def extract_skills(self, text: str) -> dict:
        """Extract skills from text"""
        text_lower = text.lower()
        
        found_technical = []
        found_soft = []
        
        # Check for technical skills
        for skill in self.technical_skills:
            if skill in text_lower:
                found_technical.append(skill)
        
        # Check for soft skills
        for skill in self.soft_skills:
            if skill in text_lower:
                found_soft.append(skill)
        
        return {
            'technical': found_technical,
            'soft': found_soft
        }
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using TF-IDF"""
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(max(0.0, min(1.0, similarity)))
        except:
            return 0.0
    
    def analyze_resume(self, resume_text: str, jd_text: str) -> dict:
        """Main analysis function"""
        # Extract keywords
        resume_keywords = self.extract_keywords(resume_text)
        jd_keywords = self.extract_keywords(jd_text)
        
        # Find matches
        matched_keywords = list(set(resume_keywords) & set(jd_keywords))
        missing_keywords = list(set(jd_keywords) - set(resume_keywords))
        
        # Extract skills
        resume_skills = self.extract_skills(resume_text)
        jd_skills = self.extract_skills(jd_text)
        
        # Find skill matches
        matched_technical = list(set(resume_skills['technical']) & set(jd_skills['technical']))
        missing_technical = list(set(jd_skills['technical']) - set(resume_skills['technical']))
        
        matched_soft = list(set(resume_skills['soft']) & set(jd_skills['soft']))
        missing_soft = list(set(jd_skills['soft']) - set(resume_skills['soft']))
        
        # Calculate similarity
        similarity = self.calculate_similarity(resume_text, jd_text)
        
        # Calculate scores
        keyword_score = min(3.0, (len(matched_keywords) / max(1, len(jd_keywords))) * 3.0)
        skills_score = min(2.0, ((len(matched_technical) + len(matched_soft)) / 
                                max(1, len(jd_skills['technical']) + len(jd_skills['soft']))) * 2.0)
        similarity_score = similarity * 1.0
        
        # Check structure
        structure_score = self._check_structure(resume_text)
        
        # Calculate total score (out of 10)
        total_score = round(keyword_score + skills_score + similarity_score + structure_score, 2)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            matched_keywords, missing_keywords,
            matched_technical, missing_technical,
            matched_soft, missing_soft,
            similarity
        )
        
        return {
            'total_score': total_score,
            'scores': {
                'keyword_matching': round(keyword_score, 2),
                'skills_match': round(skills_score, 2),
                'experience_relevance': round(similarity_score, 2),
                'structure': round(structure_score, 2)
            },
            'keywords': {
                'resume': resume_keywords[:20],
                'jd': jd_keywords[:20],
                'matched': matched_keywords[:15],
                'missing': missing_keywords[:15]
            },
            'skills': {
                'resume_technical': resume_skills['technical'],
                'resume_soft': resume_skills['soft'],
                'jd_technical': jd_skills['technical'],
                'jd_soft': jd_skills['soft'],
                'matched_technical': matched_technical,
                'missing_technical': missing_technical,
                'matched_soft': matched_soft,
                'missing_soft': missing_soft
            },
            'similarity': round(similarity, 3),
            'recommendations': recommendations,
            'structure_analysis': self._analyze_structure(resume_text)
        }
    
    def _check_structure(self, text: str) -> float:
        """Check resume structure"""
        score = 0.0
        
        # Check for sections
        text_lower = text.lower()
        sections = ['experience', 'education', 'skills', 'work', 'employment']
        found_sections = sum(1 for section in sections if section in text_lower)
        score += min(0.5, found_sections / 3 * 0.5)
        
        # Check for contact info
        has_email = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
        has_phone = bool(re.search(r'\b(?:\+?\d{1,3}[-.\s]?)?(?:\d{3}[-.\s]?\d{3}[-.\s]?\d{4})\b', text))
        contact_score = (1 if has_email else 0) + (1 if has_phone else 0)
        score += min(0.3, contact_score / 2 * 0.3)
        
        # Check for bullet points
        bullet_patterns = ['•', '- ', '* ', '· ']
        has_bullets = any(pattern in text for pattern in bullet_patterns)
        score += 0.2 if has_bullets else 0.0
        
        return min(1.5, score)
    
    def _analyze_structure(self, text: str) -> dict:
        """Analyze resume structure"""
        text_lower = text.lower()
        
        sections_found = []
        for section in ['experience', 'education', 'skills']:
            if section in text_lower:
                sections_found.append(section)
        
        has_email = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
        has_phone = bool(re.search(r'\b(?:\+?\d{1,3}[-.\s]?)?(?:\d{3}[-.\s]?\d{3}[-.\s]?\d{4})\b', text))
        
        return {
            'sections_found': sections_found,
            'sections_missing': [s for s in ['experience', 'education', 'skills'] if s not in text_lower],
            'has_contact': {
                'email': has_email,
                'phone': has_phone
            },
            'has_bullets': any(pattern in text for pattern in ['•', '- ', '* ', '· '])
        }
    
    def _generate_recommendations(self, matched_keywords, missing_keywords,
                                 matched_technical, missing_technical,
                                 matched_soft, missing_soft, similarity) -> list:
        """Generate recommendations"""
        recommendations = []
        
        if len(missing_keywords) > 0:
            recommendations.append(f"Add these keywords from the job description: {', '.join(missing_keywords[:5])}")
        
        if len(missing_technical) > 0:
            recommendations.append(f"Consider highlighting these technical skills: {', '.join(missing_technical[:3])}")
        
        if len(missing_soft) > 0:
            recommendations.append(f"Emphasize these soft skills: {', '.join(missing_soft[:3])}")
        
        if similarity < 0.5:
            recommendations.append("Tailor your resume more closely to the job description.")
        
        return recommendations

# Initialize analyzer
analyzer = ResumeAnalyzer()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'ATS Resume Analyzer API',
        'version': '1.0'
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_resume():
    """Analyze resume from file upload"""
    try:
        if 'resume' not in request.files or 'job_description' not in request.files:
            return jsonify({'error': 'Both resume and job description files are required'}), 400
        
        resume_file = request.files['resume']
        jd_file = request.files['job_description']
        
        if resume_file.filename == '' or jd_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not (allowed_file(resume_file.filename) and allowed_file(jd_file.filename)):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save files temporarily
        resume_filename = f"resume_{uuid.uuid4().hex}{os.path.splitext(resume_file.filename)[1]}"
        jd_filename = f"jd_{uuid.uuid4().hex}{os.path.splitext(jd_file.filename)[1]}"
        
        resume_path = os.path.join(UPLOAD_FOLDER, resume_filename)
        jd_path = os.path.join(UPLOAD_FOLDER, jd_filename)
        
        resume_file.save(resume_path)
        jd_file.save(jd_path)
        
        # Read documents
        doc_reader = DocumentReader()
        resume_text = doc_reader.read_document(resume_path)
        jd_text = doc_reader.read_document(jd_path)
        
        # Clean up
        try:
            os.remove(resume_path)
            os.remove(jd_path)
        except:
            pass
        
        # Analyze
        analysis = analyzer.analyze_resume(resume_text, jd_text)
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/analyze-text', methods=['POST'])
def analyze_text():
    """Analyze resume from text input"""
    try:
        data = request.json
        resume_text = data.get('resume_text', '')
        jd_text = data.get('jd_text', '')
        
        if not resume_text or not jd_text:
            return jsonify({'error': 'Both resume_text and jd_text are required'}), 400
        
        # Analyze
        analysis = analyzer.analyze_resume(resume_text, jd_text)
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Test endpoint"""
    test_resume = """
    John Doe
    Senior Software Engineer
    Email: john.doe@email.com
    Phone: (123) 456-7890
    
    EXPERIENCE
    Senior Software Engineer - Tech Company (2020-Present)
    • Developed scalable microservices using Python and FastAPI
    • Implemented CI/CD pipelines with Docker and Kubernetes
    • Led a team of 5 developers on key projects
    
    EDUCATION
    BS Computer Science - University of Technology (2016-2020)
    
    SKILLS
    Python, JavaScript, AWS, Docker, Kubernetes, React, SQL
    """
    
    test_jd = """
    Software Engineer
    We are looking for a skilled Software Engineer with experience in:
    - Python development
    - Microservices architecture
    - Docker and Kubernetes
    - AWS cloud services
    - REST APIs
    
    Required Skills:
    • Python programming (3+ years)
    • Docker containerization
    • Cloud experience (AWS preferred)
    • CI/CD pipelines
    
    Nice to have:
    • React or Angular
    • Database design
    • Team leadership
    """
    
    analysis = analyzer.analyze_resume(test_resume, test_jd)
    
    return jsonify({
        'success': True,
        'analysis': analysis,
        'message': 'Test analysis completed successfully'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)