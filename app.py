from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import json
import re
import uuid
from datetime import datetime
import PyPDF2

app = Flask(__name__)
CORS(app)

# Simple document reader
class DocumentReader:
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
        elif ext == '.txt':
            return DocumentReader.read_txt(filepath)
        else:
            # For DOCX, just return empty string or implement simple text extraction
            return ""

class SimpleATSAnalyzer:
    """Simple ATS analyzer without heavy dependencies"""
    
    def __init__(self):
        self.technical_skills = [
            'python', 'java', 'javascript', 'sql', 'aws', 'azure', 'docker',
            'kubernetes', 'react', 'angular', 'nodejs', 'django', 'flask',
            'machine learning', 'data science', 'pandas', 'numpy', 'git',
            'linux', 'postgresql', 'mysql', 'mongodb', 'rest api'
        ]
        
        self.soft_skills = [
            'leadership', 'communication', 'teamwork', 'problem solving',
            'analytical', 'creative', 'adaptable', 'organized', 'collaborative'
        ]
    
    def analyze(self, resume_text: str, jd_text: str) -> dict:
        """Simple analysis function"""
        resume_lower = resume_text.lower()
        jd_lower = jd_text.lower()
        
        # Calculate keyword matches
        resume_words = set(re.findall(r'\b\w+\b', resume_lower))
        jd_words = set(re.findall(r'\b\w+\b', jd_lower))
        
        common_words = resume_words.intersection(jd_words)
        
        # Check for skills
        found_tech_skills = []
        missing_tech_skills = []
        
        for skill in self.technical_skills:
            if skill in jd_lower:
                if skill in resume_lower:
                    found_tech_skills.append(skill)
                else:
                    missing_tech_skills.append(skill)
        
        # Calculate simple score (0-10)
        keyword_match_score = min(3.0, (len(common_words) / max(1, len(jd_words))) * 3)
        skill_match_score = min(2.0, (len(found_tech_skills) / max(1, len([s for s in self.technical_skills if s in jd_lower]))) * 2)
        
        # Check structure
        structure_score = self._check_structure(resume_text)
        
        # Similarity (simple word overlap)
        similarity = len(common_words) / max(1, len(jd_words))
        relevance_score = min(1.0, similarity)
        
        # Total score
        total_score = round(keyword_match_score + skill_match_score + structure_score + relevance_score, 2)
        
        # Recommendations
        recommendations = []
        if missing_tech_skills:
            recommendations.append(f"Add missing technical skills: {', '.join(missing_tech_skills[:3])}")
        if keyword_match_score < 2:
            recommendations.append("Add more keywords from the job description to your resume")
        if structure_score < 1:
            recommendations.append("Improve resume structure: include clear Experience, Education, and Skills sections")
        
        return {
            'total_score': total_score,
            'breakdown': {
                'keyword_matching': round(keyword_match_score, 2),
                'skills_match': round(skill_match_score, 2),
                'structure': round(structure_score, 2),
                'relevance': round(relevance_score, 2)
            },
            'skills_analysis': {
                'found_technical': found_tech_skills,
                'missing_technical': missing_tech_skills,
                'common_keywords': list(common_words)[:20]
            },
            'recommendations': recommendations,
            'metrics': {
                'resume_length': len(resume_text),
                'jd_length': len(jd_text),
                'word_overlap': len(common_words),
                'similarity_percentage': round(similarity * 100, 1)
            }
        }
    
    def _check_structure(self, text: str) -> float:
        """Check basic resume structure"""
        score = 0.0
        text_lower = text.lower()
        
        # Check for common sections
        sections = ['experience', 'education', 'skills']
        found_sections = [s for s in sections if s in text_lower]
        
        if len(found_sections) >= 2:
            score += 0.5
        if len(found_sections) == 3:
            score += 0.3
        
        # Check for contact info
        has_email = bool(re.search(r'\S+@\S+\.\S+', text))
        has_phone = bool(re.search(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text))
        
        if has_email:
            score += 0.1
        if has_phone:
            score += 0.1
        
        # Check for bullet points
        if any(c in text for c in ['•', '- ', '* ']):
            score += 0.3
        
        return min(1.5, score)

# Initialize analyzer
analyzer = SimpleATSAnalyzer()
doc_reader = DocumentReader()

@app.route('/')
def home():
    return jsonify({
        'service': 'ATS Resume Analyzer API',
        'status': 'running',
        'endpoints': {
            '/api/health': 'GET - Health check',
            '/api/analyze': 'POST - Analyze resume (files)',
            '/api/analyze-text': 'POST - Analyze resume (text)',
            '/api/test': 'GET - Test analysis'
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0'
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_resume():
    try:
        if 'resume' not in request.files:
            return jsonify({'error': 'Resume file is required'}), 400
        
        resume_file = request.files['resume']
        
        if resume_file.filename == '':
            return jsonify({'error': 'No resume file selected'}), 400
        
        # Save file temporarily
        temp_filename = f"temp_{uuid.uuid4().hex}.pdf"
        temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
        resume_file.save(temp_path)
        
        # Read resume
        resume_text = doc_reader.read_document(temp_path)
        
        # Clean up
        try:
            os.remove(temp_path)
        except:
            pass
        
        # Get job description text
        jd_text = ""
        if 'job_description' in request.files:
            jd_file = request.files['job_description']
            if jd_file.filename != '':
                jd_temp = f"jd_{uuid.uuid4().hex}.txt"
                jd_path = os.path.join(tempfile.gettempdir(), jd_temp)
                jd_file.save(jd_path)
                jd_text = doc_reader.read_document(jd_path)
                try:
                    os.remove(jd_path)
                except:
                    pass
        
        # If no JD file, check for text
        if not jd_text and request.form.get('jd_text'):
            jd_text = request.form.get('jd_text')
        
        if not jd_text:
            return jsonify({'error': 'Job description is required'}), 400
        
        # Analyze
        analysis = analyzer.analyze(resume_text, jd_text)
        
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
    try:
        data = request.json or request.form
        
        resume_text = data.get('resume_text', '')
        jd_text = data.get('jd_text', '')
        
        if not resume_text or not jd_text:
            return jsonify({'error': 'Both resume_text and jd_text are required'}), 400
        
        # Analyze
        analysis = analyzer.analyze(resume_text, jd_text)
        
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
    """Test endpoint with sample data"""
    test_resume = """
    John Doe - Software Engineer
    Email: john@example.com | Phone: (123) 456-7890
    
    EXPERIENCE
    Senior Developer at TechCorp (2020-Present)
    • Built web applications using Python and JavaScript
    • Managed AWS infrastructure
    • Implemented CI/CD pipelines
    
    EDUCATION
    BS Computer Science - State University (2016-2020)
    
    SKILLS
    Python, JavaScript, AWS, Docker, Git, SQL
    """
    
    test_jd = """
    Software Developer
    
    Looking for a developer with experience in:
    - Python programming
    - Web development
    - AWS cloud services
    - Docker containers
    
    Requirements:
    • 2+ years Python experience
    • Knowledge of web frameworks
    • Cloud computing experience
    • Version control with Git
    """
    
    analysis = analyzer.analyze(test_resume, test_jd)
    
    return jsonify({
        'success': True,
        'analysis': analysis,
        'message': 'Test analysis completed successfully'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)