# NextHorizon - AI-Powered Career Development Platform

## Overview
NextHorizon is a streamlined AI-powered career guidance platform that helps professionals identify skill gaps, discover relevant roles, and get personalized course recommendations. Built with a clean, simplified architecture using Streamlit and OpenAI's latest models for maximum reliability and performance.

## Project Structure

```
nexthorizon/
├── app.py                          # 🚀 MAIN APPLICATION ENTRY POINT
├── config/
│   ├── __init__.py                 
│   └── session_config.py           # Streamlit session state management
├── ai/
│   ├── __init__.py                 
│   └── openai_client.py            # OpenAI API integration (GPT-4o-mini + embeddings)
├── ui/                             # 🎨 STREAMLIT UI COMPONENTS (Flattened Structure)
│   ├── __init__.py                 
│   ├── course_recommendations.py   # Course recommendation interface
│   ├── resume_parsing.py           # Resume upload & parsing interface
│   ├── role_recommendations.py     # Role recommendation interface
│   ├── sidebar.py                  # Database upload sidebar
│   └── skill_gaps.py               # Skill gap analysis interface
├── utils/                          # 🛠️ CORE UTILITY FUNCTIONS
│   ├── __init__.py                 
│   ├── data_enhancer.py            # Resume data enhancement and cleanup
│   ├── resume_processor.py         # Complete resume processing pipeline (includes text building)
│   ├── session_helpers.py          # Session state management and validation
│   ├── skill_analysis.py           # Skill extraction and gap analysis
│   └── skill_clarification.py      # Skill clarification workflow
├── build_jd_dataset/              # 📊 JOB DESCRIPTION DATABASE
│   ├── build_jd_database.py        # Database builder script
│   ├── jd_database.csv             # 400+ curated job descriptions
│   └── role_list.csv               # Available career roles
├── build_training_dataset/        # 📚 TRAINING DATA UTILITIES
│   ├── build_training_database.py  # Training dataset builder
│   ├── skill_list.csv              # Comprehensive skills database
│   └── training_database.csv       # Training dataset for courses
└── PROJECT_OVERVIEW.md            # 📋 This documentation
```

## Core Features

### 🎯 Resume Analysis & Parsing
- **Multi-Format Upload**: Seamless PDF and TXT resume processing
- **AI-Powered Extraction**: OpenAI-based intelligent content parsing
- **Skills Identification**: Automated skill extraction and categorization
- **Experience Calculation**: Automatic total years of experience computation
- **Data Enhancement**: Missing information completion using AI analysis

### 🔍 Role Recommendations
- **Vector Search Matching**: OpenAI embeddings (text-embedding-3-small) for precise role matching
- **Comprehensive Database**: 400+ curated job descriptions across industries
- **Similarity Scoring**: Advanced semantic matching with percentage compatibility
- **Detailed Insights**: Complete job descriptions with requirements and qualifications
- **Personalized Results**: AI-driven recommendations based on individual profile

### 📊 Skill Gap Analysis
- **Intelligent Gap Detection**: AI-powered analysis of missing skills for target roles
- **Priority Ranking**: Skills ranked by importance for career advancement
- **Clear Learning Paths**: Structured roadmap for skill development
- **Interactive Clarification**: Dynamic Q&A to refine skill assessments
- **Market Alignment**: Skills aligned with current industry demands

### 📚 Course Recommendations
- **AI-Driven Selection**: GPT-4o-mini powered course matching based on skill gaps
- **Curated Learning Content**: High-quality courses from established platforms
- **Personalized Pathways**: Custom learning sequences for career goals
- **Relevance Scoring**: AI-evaluated course recommendations with confidence ratings
- **Efficient Learning**: Optimized skill development strategies

## Technical Architecture

### 🤖 AI-First Design
- **OpenAI Integration**: GPT-4o-mini for text generation and analysis
- **Vector Embeddings**: text-embedding-3-small for semantic similarity matching
- **No ML Dependencies**: Eliminated complex trained models in favor of reliable APIs
- **Real-time Processing**: Fast, consistent AI-powered recommendations

### 🔧 Simplified Data Flow
- **Linear Workflow**: Upload → Parse → Select Role → Find Gaps → Get Courses
- **Session Management**: Clean Streamlit session state handling
- **Utility-Based**: Simple functions instead of complex class hierarchies
- **Error Resilience**: Graceful error handling throughout the pipeline

### 🛡️ Clean Architecture
- **Minimal Dependencies**: Streamlit, OpenAI, Pandas - core libraries only
- **Modular Design**: Independent components with clear responsibilities
- **Maintainable Code**: Straightforward functions and logical organization
- **Production Ready**: Simplified structure suitable for deployment
- **Version-Free**: Clean codebase without development artifacts or version numbers

## Quick Start Guide

### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation & Setup
```bash
# Install dependencies
pip install streamlit openai pandas numpy

# Set OpenAI API key (Windows PowerShell)
$env:OPENAI_API_KEY = "your-api-key-here"

# Run the application
streamlit run app.py
```

### Usage Workflow
1. **Database Setup**: Upload JD database (jd_database.csv) via sidebar
2. **Resume Upload**: Upload PDF/TXT resume or paste text directly
3. **Parse Resume**: AI extracts skills, experience, and key information
4. **Select Role**: Choose target role from AI-recommended options
5. **Analyze Gaps**: View missing skills with clarification questions
6. **Get Courses**: Receive personalized course recommendations

## Development Guidelines

### 🏗️ Simplified Architecture Principles
- **Utility-First**: Simple functions over complex class hierarchies
- **OpenAI-Powered**: Leverage API capabilities instead of maintaining ML models
- **Clean Separation**: UI components in `ui/`, core logic in `utils/`
- **Session-Based**: Streamlit session state for user data persistence
- **Error-Resilient**: Graceful handling of API failures and edge cases

### � Code Organization
```python
# Main application flow
from config.session_config import initialize_session_state
from ui.sidebar import render_sidebar
from ui.resume_parsing import render_resume_parsing
from ui.role_recommendations import render_role_recommendations
from ui.skill_gaps import render_skill_gaps
from ui.course_recommendations import render_course_recommendations

# Core utilities
from utils.resume_processor import process_resume
from utils.skill_analysis import extract_skills_from_jd_text, calculate_skill_gaps
from utils.skill_clarification import generate_clarification_questions
from utils.session_helpers import validate_role_selected
from utils.data_enhancer import backfill_from_text

# AI integration
from ai.openai_client import (
    openai_rank_roles,
    openai_recommend_courses,
    openai_parse_resume
)
```

## Key Benefits

### 🚀 Production Ready
- **Simplified Architecture**: Clean, maintainable code structure
- **API-First Approach**: Reliable OpenAI integration without local ML complexity
- **Minimal Dependencies**: Core libraries only for reduced deployment overhead
- **Fast Performance**: Vector search and AI APIs for responsive user experience

### 👥 Developer Experience
- **Intuitive Structure**: Logical file organization with clear purpose
- **Easy Navigation**: Flattened UI structure and centralized utilities
- **Quick Setup**: Minimal configuration required to get started
- **Clear Responsibilities**: Each module has a single, well-defined purpose

### 🎯 User Experience
- **Streamlined Workflow**: Linear progression from resume to recommendations
- **Intelligent Interactions**: AI-powered clarification questions for better accuracy
- **Real-time Results**: Fast AI processing with immediate feedback
- **Professional Output**: Comprehensive career guidance and actionable insights

## Architecture Highlights

### 🔄 Data Flow
```
1. Resume Upload → utils/resume_processor.py
2. AI Parsing → ai/openai_client.py (GPT-4o-mini)
3. Role Matching → Vector embeddings + similarity search
4. Skill Analysis → utils/skill_analysis.py
5. Gap Identification → AI-powered comparison
6. Course Recommendations → Training dataset + AI ranking
```

### 🗂️ File Responsibilities
- **`app.py`**: Main Streamlit application with tab navigation
- **`ui/`**: Individual UI components for each application step
- **`utils/`**: Core processing functions and session management (optimized structure)
  - `resume_processor.py`: Complete resume processing pipeline including text building
  - `skill_analysis.py`: Skill extraction and gap analysis (renamed for clarity)
  - `session_helpers.py`: Session state management and validation (renamed for clarity)
  - `skill_clarification.py`: Interactive skill clarification workflow
  - `data_enhancer.py`: Resume data enhancement and cleanup
- **`ai/`**: OpenAI API integration and response handling
- **`config/`**: Application configuration and session initialization
- **`build_*_dataset/`**: Data preparation and database management

### 🧹 Simplified Design
- **Removed**: Complex agent frameworks, TF-IDF fallbacks, ML training pipelines, version artifacts
- **Consolidated**: Duplicate functions, over-engineered abstractions, redundant utilities
- **Streamlined**: Directory structure, import paths, code organization, file naming
- **Preserved**: Core functionality, data integrity, user experience
- **Optimized**: Utils folder organization with semantic naming and function consolidation

## Recent Architecture Improvements

### 🔧 Code Organization Optimization
- **Utils Folder Restructured**: Consolidated from 6 files to 5 clean, semantically named utilities
- **Function Consolidation**: Merged resume text building functions into `resume_processor.py`
- **Semantic Naming**: Renamed files for better clarity:
  - `skill_extraction.py` → `skill_analysis.py` (better reflects analysis capabilities)
  - `session_validators.py` → `session_helpers.py` (reflects mixed validation/getter responsibilities)
- **Version Cleanup**: Removed all version artifacts (v1, v2, v3, v4, v5, v6) from codebase
- **Duplicate Elimination**: Consolidated redundant OpenAI functions across modules

### 🎯 Production Readiness
- **Clean File Structure**: Optimal organization for maintainability and developer experience
- **No Technical Debt**: All legacy code, unused imports, and development artifacts removed
- **Streamlined Processing**: Simplified data flow with consolidated utility functions
- **Professional Codebase**: Version-free, artifact-free production-ready code

## Contributing

### 🛠️ Development Setup
1. **Environment**: Python 3.8+ with virtual environment recommended
2. **API Access**: OpenAI API key required for all AI features
3. **Database**: Upload `jd_database.csv` through the application UI
4. **Testing**: Manual testing through Streamlit interface

### 📋 Extension Points
- **New UI Components**: Add files to `ui/` directory with `render_*()` functions
- **Additional Utilities**: Extend `utils/` with new processing functions
- **Enhanced AI Features**: Modify `ai/openai_client.py` for new capabilities
- **Database Integration**: Expand dataset builders for additional data sources

### 🎯 Future Enhancements
- **API Endpoints**: REST API layer for external integrations
- **Database Backends**: PostgreSQL/MongoDB for production data storage
- **Authentication**: User accounts and personalized recommendations
- **Analytics**: Usage tracking and recommendation effectiveness metrics
- **Mobile Support**: Responsive design for mobile devices

---

*Built with ❤️ using Streamlit and OpenAI - Simplified for maintainability and performance*

## Summary

NextHorizon is a sophisticated, AI-powered career guidance platform built with modern software engineering principles. The clean, modular architecture ensures excellent maintainability while delivering powerful features for career development and skill enhancement. Whether you're a job seeker looking for your next opportunity or a professional planning your career growth, NextHorizon provides intelligent, personalized guidance to help you succeed.
