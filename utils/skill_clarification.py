# FILE: utils/skill_clarification.py
from __future__ import annotations
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class ClarificationQuestion:
    """Simple data class for clarification questions"""
    id: str
    text: str
    type: str = "text_input"
    options: List[str] = None

def generate_clarification_questions(structured_json: Dict[str, Any], skill_gaps: List[str], min_required_skill_coverage: float = 0.5) -> List[ClarificationQuestion]:
    """Generate targeted questions to clarify missing skills"""
    
    if not skill_gaps:
        return []
    
    questions = []
    current_skills = structured_json.get("technical_skills", [])
    
    # Always generate questions if there are skill gaps, regardless of coverage
    # The coverage threshold was preventing useful clarification questions
    
    # Generate questions for top skill gaps
    for i, gap in enumerate(skill_gaps[:10]):  # Limit to top 10 gaps
        gap_lower = gap.lower()
        
        # Programming languages
        if gap_lower in ['python', 'java', 'javascript', 'c++', 'c#', 'go', 'rust', 'php', 'ruby', 'scala']:
            questions.append(ClarificationQuestion(
                id=f"skill_{gap_lower}",
                text=f"Do you have experience with {gap}?",
                type="multiselect",
                options=["Yes, professional experience", "Yes, personal projects", "Basic knowledge", "No experience"]
            ))
        
        # Frameworks and technologies
        elif any(keyword in gap_lower for keyword in ['framework', 'library', 'tool']):
            questions.append(ClarificationQuestion(
                id=f"tech_{gap_lower}",
                text=f"What is your experience level with {gap}?",
                type="multiselect",
                options=["Expert", "Intermediate", "Beginner", "No experience"]
            ))
        
        # Cloud platforms
        elif gap_lower in ['aws', 'azure', 'gcp', 'google cloud']:
            questions.append(ClarificationQuestion(
                id=f"cloud_{gap_lower}",
                text=f"Which {gap} services have you used?",
                type="text_input"
            ))
        
        # Databases
        elif gap_lower in ['sql', 'mysql', 'postgresql', 'mongodb', 'redis']:
            questions.append(ClarificationQuestion(
                id=f"db_{gap_lower}",
                text=f"What is your experience with {gap} databases?",
                type="multiselect",
                options=["Advanced queries", "Basic CRUD operations", "Database design", "No experience"]
            ))
        
        # Generic skill question
        else:
            questions.append(ClarificationQuestion(
                id=f"generic_{gap_lower}",
                text=f"Do you have any experience with {gap}?",
                type="text_input"
            ))
    
    return questions

def incorporate_clarification_answers(answers: Dict[str, Any], structured_json: Dict[str, Any]) -> Dict[str, Any]:
    """Apply user answers to update resume data structure"""
    
    updated_json = structured_json.copy()
    current_skills = updated_json.get("technical_skills", [])
    
    for answer_id, answer_value in answers.items():
        if not answer_value:  # Skip empty answers
            continue
            
        # Extract skill from answer ID
        if answer_id.startswith('skill_'):
            skill = answer_id.replace('skill_', '')
            # Add skill if user has any level of experience
            if isinstance(answer_value, list):
                if any(exp in str(answer_value).lower() for exp in ['yes', 'professional', 'personal', 'basic']):
                    if skill not in [s.lower() for s in current_skills]:
                        current_skills.append(skill)
            elif isinstance(answer_value, str) and answer_value.strip():
                if skill not in [s.lower() for s in current_skills]:
                    current_skills.append(skill)
        
        elif answer_id.startswith('tech_'):
            tech = answer_id.replace('tech_', '')
            # Add technology if user has experience
            if isinstance(answer_value, list):
                if any(level in str(answer_value).lower() for level in ['expert', 'intermediate', 'beginner']):
                    if tech not in [s.lower() for s in current_skills]:
                        current_skills.append(tech)
        
        elif answer_id.startswith('cloud_'):
            cloud = answer_id.replace('cloud_', '')
            # Add cloud platform if user mentioned services
            if isinstance(answer_value, str) and answer_value.strip():
                if cloud not in [s.lower() for s in current_skills]:
                    current_skills.append(cloud)
                # Also add mentioned services as separate skills
                services = [s.strip() for s in answer_value.split(',') if s.strip()]
                for service in services[:3]:  # Limit to 3 services
                    if len(service) > 2 and service not in current_skills:
                        current_skills.append(service)
        
        elif answer_id.startswith('db_'):
            db = answer_id.replace('db_', '')
            # Add database if user has experience
            if isinstance(answer_value, list):
                if any(level in str(answer_value).lower() for level in ['advanced', 'basic', 'design']):
                    if db not in [s.lower() for s in current_skills]:
                        current_skills.append(db)
        
        elif answer_id.startswith('generic_'):
            generic_skill = answer_id.replace('generic_', '')
            # Add generic skill if user provided any answer
            if isinstance(answer_value, str) and answer_value.strip():
                if generic_skill not in [s.lower() for s in current_skills]:
                    current_skills.append(generic_skill)
    
    # Update the structured JSON
    updated_json["technical_skills"] = list(set(current_skills))  # Remove duplicates
    
    return updated_json
