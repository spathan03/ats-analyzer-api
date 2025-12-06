from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import json
import traceback
import uuid
from datetime import datetime
import sys

# Import your ATS analyzer modules
sys.path.append('.')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Import the ATS analyzer components
try:
    # Your existing modules will be imported here
    # We'll restructure them for web use
    from ats_core import ATSScorer, DocumentReader
except ImportError:
    print("ATS modules not found, using stub implementation")
    from ats_stub import ATSScorer, DocumentReader

# Configuration
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'ATS Resume Analyzer API'
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_resume():
    """Main analysis endpoint"""
    try:
        # Check if files are present
        if 'resume' not in request.files or 'job_description' not in request.files:
            return jsonify({
                'error': 'Both resume and job description files are required'
            }), 400
        
        resume_file = request.files['resume']
        jd_file = request.files['job_description']
        
        if resume_file.filename == '' or jd_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not (allowed_file(resume_file.filename) and allowed_file(jd_file.filename)):
            return jsonify({
                'error': 'Invalid file type. Allowed: PDF, DOCX, TXT'
            }), 400
        
        # Generate unique filenames
        resume_filename = f"resume_{uuid.uuid4().hex}{os.path.splitext(resume_file.filename)[1]}"
        jd_filename = f"jd_{uuid.uuid4().hex}{os.path.splitext(jd_file.filename)[1]}"
        
        resume_path = os.path.join(UPLOAD_FOLDER, resume_filename)
        jd_path = os.path.join(UPLOAD_FOLDER, jd_filename)
        
        # Save files
        resume_file.save(resume_path)
        jd_file.save(jd_path)
        
        # Read documents
        doc_reader = DocumentReader()
        resume_text = doc_reader.read_document(resume_path)
        jd_text = doc_reader.read_document(jd_path)
        
        # Clean up files
        os.remove(resume_path)
        os.remove(jd_path)
        
        # Analyze
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
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/analyze-text', methods=['POST'])
def analyze_text():
    """Alternative endpoint for text input"""
    try:
        data = request.json
        resume_text = data.get('resume_text', '')
        jd_text = data.get('jd_text', '')
        
        if not resume_text or not jd_text:
            return jsonify({'error': 'Both resume_text and jd_text are required'}), 400
        
        scorer = ATSScorer()
        analysis = scorer.analyze_resume(resume_text, jd_text)
        
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
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)