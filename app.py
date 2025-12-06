from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import json
import traceback
import uuid
from datetime import datetime
import sys

app = Flask(__name__)
CORS(app)

# Configure upload settings
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Try to import ATS modules with fallback
try:
    # Add current directory to path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Try to import the main module
    try:
        from ats_core import ATSScorer, DocumentReader
        HAS_ATS_MODULES = True
        print("✅ ATS modules loaded successfully")
    except ImportError as e:
        print(f"⚠ Could not import ATS modules: {e}")
        print("⚠ Using stub implementation")
        from ats_stub import ATSScorer, DocumentReader
        HAS_ATS_MODULES = False
except Exception as e:
    print(f"❌ Error importing modules: {e}")
    # Define stub classes if everything fails
    class DocumentReader:
        @staticmethod
        def read_document(filepath: str) -> str:
            return "File reading not available"
    
    class ATSScorer:
        def analyze_resume(self, resume_text: str, jd_text: str) -> dict:
            return {"error": "ATS analyzer not available"}
    
    HAS_ATS_MODULES = False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'ATS Resume Analyzer API',
        'has_ats_modules': HAS_ATS_MODULES,
        'version': '2.0'
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_resume():
    """Main analysis endpoint for file uploads"""
    try:
        # Check if files are present
        if 'resume' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Resume file is required'
            }), 400
        
        if 'job_description' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Job description file is required'
            }), 400
        
        resume_file = request.files['resume']
        jd_file = request.files['job_description']
        
        if resume_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No resume file selected'
            }), 400
        
        if jd_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No job description file selected'
            }), 400
        
        # Validate file types
        if not allowed_file(resume_file.filename):
            return jsonify({
                'success': False,
                'error': f'Invalid resume file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        if not allowed_file(jd_file.filename):
            return jsonify({
                'success': False,
                'error': f'Invalid job description file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Generate unique filenames
        resume_filename = f"resume_{uuid.uuid4().hex}{os.path.splitext(resume_file.filename)[1]}"
        jd_filename = f"jd_{uuid.uuid4().hex}{os.path.splitext(jd_file.filename)[1]}"
        
        resume_path = os.path.join(UPLOAD_FOLDER, resume_filename)
        jd_path = os.path.join(UPLOAD_FOLDER, jd_filename)
        
        # Save files temporarily
        resume_file.save(resume_path)
        jd_file.save(jd_path)
        
        # Read documents
        doc_reader = DocumentReader()
        resume_text = doc_reader.read_document(resume_path)
        jd_text = doc_reader.read_document(jd_path)
        
        # Clean up temporary files
        try:
            os.remove(resume_path)
            os.remove(jd_path)
        except:
            pass
        
        # Validate text content
        if len(resume_text.strip()) < 50:
            return jsonify({
                'success': False,
                'error': 'Resume text is too short or could not be read properly'
            }), 400
        
        if len(jd_text.strip()) < 50:
            return jsonify({
                'success': False,
                'error': 'Job description text is too short or could not be read properly'
            }), 400
        
        # Perform analysis
        scorer = ATSScorer()
        analysis = scorer.analyze_resume(resume_text, jd_text)
        
        # Convert sets to lists for JSON serialization
        def convert_sets(obj):
            if isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: convert_sets(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_sets(i) for i in obj]
            return obj
        
        analysis = convert_sets(analysis)
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat(),
            'file_info': {
                'resume_filename': resume_file.filename,
                'jd_filename': jd_file.filename,
                'resume_length': len(resume_text),
                'jd_length': len(jd_text)
            }
        })
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in analyze endpoint: {error_trace}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': error_trace if app.debug else None
        }), 500

@app.route('/api/analyze-text', methods=['POST'])
def analyze_text():
    """Alternative endpoint for text input"""
    try:
        data = request.json
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        resume_text = data.get('resume_text', '')
        jd_text = data.get('jd_text', '')
        
        if not resume_text or resume_text.strip() == '':
            return jsonify({
                'success': False,
                'error': 'Resume text is required'
            }), 400
        
        if not jd_text or jd_text.strip() == '':
            return jsonify({
                'success': False,
                'error': 'Job description text is required'
            }), 400
        
        if len(resume_text.strip()) < 100:
            return jsonify({
                'success': False,
                'error': 'Resume text is too short (minimum 100 characters)'
            }), 400
        
        if len(jd_text.strip()) < 50:
            return jsonify({
                'success': False,
                'error': 'Job description text is too short (minimum 50 characters)'
            }), 400
        
        # Perform analysis
        scorer = ATSScorer()
        analysis = scorer.analyze_resume(resume_text.strip(), jd_text.strip())
        
        # Convert sets to lists
        def convert_sets(obj):
            if isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: convert_sets(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_sets(i) for i in obj]
            return obj
        
        analysis = convert_sets(analysis)
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in analyze-text endpoint: {error_trace}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': error_trace if app.debug else None
        }), 500

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Test endpoint to verify functionality"""
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
    
    try:
        scorer = ATSScorer()
        analysis = scorer.analyze_resume(test_resume, test_jd)
        
        # Convert sets to lists
        def convert_sets(obj):
            if isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: convert_sets(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_sets(i) for i in obj]
            return obj
        
        analysis = convert_sets(analysis)
        
        return jsonify({
            'success': True,
            'message': 'Test analysis completed successfully',
            'analysis': analysis,
            'has_ats_modules': HAS_ATS_MODULES
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'has_ats_modules': HAS_ATS_MODULES
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)