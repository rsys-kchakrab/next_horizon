# NextHorizon - Your Personalized Career Guide

## Overview
NextHorizon is a comprehensive AI-powered career guidance platform that helps professionals identify skill gaps, discover relevant roles, and get personalized course recommendations. Built with a clean, modular architecture using Streamlit and OpenAI's GPT-4o-mini model.

## Project Structure

```
nexthorizon/
├── .gitignore                      # Git ignore rules
├── app.py                          # 🚀 MAIN APPLICATION ENTRY POINT
├── config/
│   ├── __init__.py                 # Session and configuration management
│   └── session_config.py           # Streamlit session state initialization
├── ui/
│   ├── __init__.py                 # User interface components
│   ├── sidebar.py                  # Database upload sidebar
│   └── tabs/
│       ├── __init__.py             
│       ├── resume_parsing.py       # Resume upload & extraction
│       ├── role_recommendations.py # AI-powered role matching
│       ├── skill_gaps.py           # Skill gap analysis & recommendations
│       └── course_recommendations.py # Personalized course suggestions
├── agents/
│   ├── __init__.py                 # AI agents for enhanced functionality
│   └── clarifier_agent.py          # Interactive question generation
├── processing/
│   ├── __init__.py                 # Data processing and enhancement
│   └── data_enhancer.py            # Resume data enhancement and completion
├── engines/
│   └── __init__.py                 # Future recommendation engines
├── ai/
│   ├── __init__.py                 # AI integration layer
│   └── openai_client.py            # GPT-4o-mini API integration
├── utils/
│   ├── __init__.py                 # Core utility functions
│   ├── compute_metrics.py          # Quality metrics and scoring
│   ├── evaluation_metrics.py       # Analytics and performance tracking
│   ├── util_models.py              # Data models and structures
│   └── resume_text_builder.py      # Resume text construction
├── security/
│   ├── __init__.py                 # Security and validation
│   └── guardrails.py               # Input validation and safety checks
├── build_jd_dataset/              # Job description database tools
│   ├── build_jd_database_v4.py     # JD database builder
│   ├── jd_database.csv             # Job descriptions dataset
│   └── role_list.csv               # Available roles list
├── build_training_dataset/        # Training dataset creation tools
│   ├── build_training_database.py  # Training data builder
│   ├── skill_list.csv              # Skills dataset
│   └── training_database.csv       # Training dataset
├── resume_processing_crew.py      # Advanced resume processing pipeline
├── ml_training.py                 # Machine learning model training
└── PROJECT_OVERVIEW.md            # 📋 This documentation file
```

## Core Features

### 🎯 Resume Analysis & Parsing
- **Smart PDF/TXT Upload**: Seamless resume document processing
- **Intelligent Text Extraction**: Automated parsing of resume content
- **Skills Identification**: AI-powered skill extraction and categorization
- **Experience Calculation**: Automatic total years of experience computation
- **Data Enhancement**: Missing information completion and validation

### 🔍 Career Role Recommendations
- **AI-Powered Matching**: GPT-4o-mini model analyzes skills vs job requirements
- **Comprehensive Job Database**: 400+ curated job descriptions across industries
- **Similarity Scoring**: Advanced matching algorithms with percentage compatibility
- **Role Insights**: Detailed job descriptions with required skills and qualifications
- **Personalized Suggestions**: Tailored recommendations based on individual profile

### 📊 Skill Gap Analysis
- **Gap Identification**: Precise analysis of missing skills for target roles
- **Priority Ranking**: Skills ranked by importance and market demand
- **Learning Pathways**: Clear roadmap for skill development
- **Progress Tracking**: Monitor skill acquisition over time
- **Industry Alignment**: Skills aligned with current market trends

### 📚 Course Recommendations
- **Intelligent Course Matching**: AI-driven course selection based on skill gaps
- **Multi-Platform Integration**: Courses from various learning platforms
- **Personalized Learning Paths**: Customized curriculum based on career goals
- **Quality Scoring**: Course recommendations with relevance ratings
- **Learning Optimization**: Efficient skill development strategies

## Technical Architecture

### 🤖 AI Integration
- **GPT-4o-mini Model**: Latest OpenAI model for enhanced accuracy and transparency
- **TF-IDF Fallback**: Robust backup system for consistent performance
- **Smart Ranking**: Advanced algorithms for role and course recommendations
- **Natural Language Processing**: Sophisticated text analysis and understanding

### 🔧 Data Processing Pipeline
- **Multi-format Support**: PDF, TXT, and direct text input processing
- **Data Validation**: Comprehensive input validation and error handling
- **Quality Metrics**: Automated scoring and quality assessment
- **Performance Analytics**: Real-time performance tracking and optimization

### 🛡️ Security & Reliability
- **Input Validation**: Robust security checks and data sanitization
- **Error Handling**: Graceful error management and user feedback
- **Session Management**: Secure session state and data persistence
- **Privacy Protection**: Safe handling of personal and professional data

## Quick Start Guide

### Installation & Setup
```bash
# Clone the repository
git clone <repository-url>
cd nexthorizon

# Install dependencies (create requirements.txt if needed)
pip install streamlit openai pandas numpy scikit-learn

# Run the application
streamlit run app.py
```

### Application Usage
1. **Upload Resume**: Use the sidebar to upload PDF/TXT resume files
2. **Parse Resume**: Extract skills, experience, and professional information
3. **Get Role Recommendations**: AI analyzes your profile for matching career opportunities
4. **Identify Skill Gaps**: Discover missing skills for your target roles
5. **Find Courses**: Get personalized course recommendations to bridge skill gaps

## Development Guidelines

### 🏗️ Modular Architecture
- **Separation of Concerns**: Clean boundaries between UI, AI, processing, and utilities
- **Scalable Design**: Easy to extend and modify individual components
- **Python Best Practices**: Proper package structure with `__init__.py` files
- **Import Clarity**: Logical import paths that reflect functionality

### 📝 Code Standards
- **Type Hints**: Enhanced code documentation and IDE support
- **Error Handling**: Comprehensive error management throughout the application
- **Documentation**: Clear docstrings and inline comments
- **Consistent Naming**: Descriptive variable and function names

### 🔄 Import Examples
```python
# Main application components
from config.session_config import initialize_session_state
from ui.sidebar import render_sidebar  
from ai.openai_client import openai_rank_roles
from utils.compute_metrics import parse_quality
from security.guardrails import validate_input
from agents.clarifier_agent import ClarifierAgent
from processing.data_enhancer import backfill_from_text
```

## Key Benefits

### 🚀 Production Ready
- **Professional Architecture**: Enterprise-level code organization
- **Maintainable Codebase**: Easy to understand, modify, and extend
- **Scalable Foundation**: Ready for team development and feature expansion
- **Clean Dependencies**: Minimal external dependencies with clear purposes

### 👥 Developer Experience
- **Fast Navigation**: Logical file locations make finding code effortless
- **IDE Support**: Proper package structure enables full autocomplete and navigation
- **Easier Debugging**: Modular design isolates issues to specific components
- **Quick Onboarding**: Self-documenting structure for new developers

### 🎯 User Experience
- **Fast Performance**: Optimized code structure and efficient algorithms
- **Reliable Results**: Robust error handling and fallback mechanisms
- **Intuitive Interface**: Clean, focused UI with clear navigation
- **Consistent Behavior**: Predictable responses and smooth user flow

## Contributing

### 🛠️ Development Setup
1. **Environment**: Python 3.8+ recommended
2. **Dependencies**: Install via `pip install -r requirements.txt` (create as needed)
3. **IDE**: VS Code with Python extension recommended
4. **Git**: Follow conventional commit messages

### 📋 Future Enhancements
- **Advanced ML Models**: Enhanced recommendation algorithms
- **Multi-language Support**: Internationalization capabilities
- **API Integration**: REST API for external system integration
- **Mobile Responsiveness**: Enhanced mobile device support
- **Real-time Updates**: Live skill gap monitoring and notifications

---

## Summary

NextHorizon is a sophisticated, AI-powered career guidance platform built with modern software engineering principles. The clean, modular architecture ensures excellent maintainability while delivering powerful features for career development and skill enhancement. Whether you're a job seeker looking for your next opportunity or a professional planning your career growth, NextHorizon provides intelligent, personalized guidance to help you succeed.
