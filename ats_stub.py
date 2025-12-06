"""
Stub implementation for when main dependencies fail
"""
import re
from typing import Dict, List, Tuple, Set
from collections import Counter

class DocumentReader:
    @staticmethod
    def read_document(filepath: str) -> str:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except:
            return ""

class NLPProcessor:
    def __init__(self):
        self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    def extract_keywords(self, text: str, top_n: int = 50) -> List[Tuple[str, float]]:
        words = [w.lower() for w in re.findall(r'\w+', text) 
                if w.lower() not in self.stop_words and len(w) > 3]
        return Counter(words).most_common(top_n)
    
    def extract_skills(self, text: str) -> Dict[str, Set[str]]:
        return {'technical': set(), 'soft': set()}
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        return 0.5

class GrammarChecker:
    def check(self, text: str) -> List[Dict]:
        return []

class ResumeStructureAnalyzer:
    def analyze(self, text: str) -> Dict:
        return {
            'sections_found': [],
            'sections_missing': [],
            'has_contact': {'email': False, 'phone': False, 'linkedin': False},
            'has_bullets': False,
            'bullet_count': 0,
            'dates_found': 0,
            'has_chronology': False,
            'line_count': 0
        }

class ATSScorer:
    def __init__(self):
        self.nlp_processor = NLPProcessor()
        self.grammar_checker = GrammarChecker()
        self.structure_analyzer = ResumeStructureAnalyzer()
    
    def analyze_resume(self, resume_text: str, jd_text: str) -> Dict:
        return {
            'keyword_matching': {'score': 0, 'percentage': 0},
            'skills_match': {'score': 0, 'percentage': 0},
            'structure': {'score': 0, 'percentage': 0},
            'grammar': {'score': 0, 'error_count': 0},
            'experience_relevance': {'score': 0, 'similarity': 0},
            'missing_requirements': {'score': 0, 'missing_count': 0},
            'total_score': 0,
            'recommendations': ['Please install all dependencies for full functionality']
        }