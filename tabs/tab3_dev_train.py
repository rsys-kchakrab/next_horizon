
# FILE: tabs/tab3_dev_train.py
from __future__ import annotations
from pathlib import Path
import os
import streamlit as st
import pandas as pd
from ml_training import train_role_model, train_training_model
from util_models import save_model, load_model

def render(models_dir: Path):
    st.subheader("Developer: Train & Manage Models")
    st.caption("Upload data, configure hyperparameters, and train models here. Users will choose Trained Model vs OpenAI at runtime.")

    role_path = models_dir / "role_model.joblib"
    course_path = models_dir / "training_model.joblib"

    # ============ Role Classifier Section ============
    st.markdown("## 🎯 Role Classifier")
    st.markdown("Train a model to predict roles from job descriptions")
    
    # JD Dataset Builder
    st.markdown("### 🌐 Build JD Dataset")
    with st.expander("Web Scrape Job Descriptions", expanded=False):
        st.markdown("""
        **Build a comprehensive job description database by scraping job platforms:**
        - LinkedIn Jobs, Indeed, Glassdoor, etc.
        - Extracts job descriptions and role titles
        - Uses role-based search queries
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            per_role = st.number_input("Jobs per Role", 1, 50, 10, 1, key="scrape_per_role")
            jd_workers = st.number_input("Parallel Workers", 1, 8, 3, 1, key="scrape_jd_workers")
        
        with col2:
            jd_engine = st.selectbox("Search Engine", ["auto", "serpapi", "ddg"], 
                                index=0, key="scrape_jd_engine",
                                help="auto: use SerpAPI if available, else DuckDuckGo")
            jd_timeout = st.number_input("Request Timeout (s)", 5, 30, 10, 1, key="scrape_jd_timeout")
        
        col_build_jd, col_status_jd = st.columns([1, 1])
        
        with col_build_jd:
            if st.button("🔄 Build JD Dataset", type="primary", key="build_jd_btn"):
                import subprocess
                import sys
                
                with st.spinner("Building JD dataset... This may take several minutes."):
                    try:
                        # Check if build script exists
                        script_path = "build_jd_dataset/build_jd_database_v4.py"
                        role_path = "build_jd_dataset/role_list.csv"
                        
                        if not os.path.exists(script_path):
                            st.error(f"Build script not found: {script_path}")
                            st.info("💡 Create build_jd_dataset/ directory with build_jd_database_v4.py script")
                        elif not os.path.exists(role_path):
                            st.error(f"Role list not found: {role_path}")
                            st.info("💡 Create role_list.csv with job role names")
                        else:
                            # Run the scraper
                            cmd = [
                                sys.executable, script_path,
                                "--roles", role_path,
                                "--out", "jd_database.csv",
                                "--per_role", str(per_role),
                                "--engine", jd_engine,
                                "--workers", str(jd_workers),
                                "--timeout", str(jd_timeout)
                            ]
                            
                            # Show command being run
                            st.code(" ".join(cmd))
                            
                            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 min timeout
                            
                            if result.returncode == 0:
                                # Load the generated dataset
                                try:
                                    new_df = pd.read_csv("jd_database.csv")
                                    st.session_state.jd_df = new_df
                                    st.success(f"✅ Successfully built JD database: {len(new_df)} job descriptions")
                                    
                                    # Show summary
                                    st.write("**Dataset Summary:**")
                                    st.write(f"- Total job descriptions: {len(new_df)}")
                                    if 'role_title' in new_df.columns:
                                        st.write(f"- Role types: {new_df['role_title'].nunique()}")
                                        
                                        # Role breakdown
                                        role_counts = new_df['role_title'].value_counts()
                                        st.write("**Top Roles:**")
                                        st.bar_chart(role_counts.head(8))
                                    
                                except Exception as e:
                                    st.error(f"Error loading generated JD dataset: {e}")
                            else:
                                st.error(f"Build failed with return code {result.returncode}")
                                st.error("STDOUT:")
                                st.code(result.stdout)
                                st.error("STDERR:")
                                st.code(result.stderr)
                                
                    except subprocess.TimeoutExpired:
                        st.error("Build process timed out (10 minutes). Try reducing --per_role or --workers.")
                    except Exception as e:
                        st.error(f"Build error: {str(e)}")
        
        with col_status_jd:
            # Show current JD scraping status
            if os.path.exists("jd_database.csv"):
                try:
                    scraped_jd_df = pd.read_csv("jd_database.csv")
                    st.info(f"📁 JD database available:\n{len(scraped_jd_df)} job descriptions")
                    if st.button("Load JD Database", key="load_scraped_jd_btn"):
                        st.session_state.jd_df = scraped_jd_df
                        st.success("✅ Loaded JD database")
                        st.rerun()
                except:
                    pass
    
    # Data upload for JD CSV
    st.markdown("### 📊 Job Description Database Upload")
    st.info("Upload job descriptions database for role analysis (used in Aspirations tab)")
    jd_csv = st.file_uploader("Upload JD Database (CSV)", type=["csv"], key="jd_upload",
                              help="CSV should contain 'jd_text' and 'role_title' columns for job descriptions")
    
    # Handle JD CSV upload
    if jd_csv is not None:
        try:
            df = pd.read_csv(jd_csv)
            st.session_state.jd_df = df
            st.success(f"✅ Loaded JD database: {len(df)} records")
            
            # Show data preview
            with st.expander("Preview Data", expanded=False):
                st.write("**Columns:**", list(df.columns))
                st.write("**Shape:**", df.shape)
                if 'role_title' in df.columns:
                    role_counts = df['role_title'].value_counts()
                    st.write("**Role Distribution:**")
                    st.bar_chart(role_counts.head(10))
                st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error loading CSV: {str(e)}")
    
    # Check current JD data status
    if st.session_state.jd_df.empty:
        st.warning("⚠️ No JD database loaded. Upload a CSV file above or ensure 'jd_database.csv' exists in the project directory.")
        # Try to load sample data
        try:
            sample_df = pd.read_csv("jd_database.csv")
            if st.button("Load Sample JD Database", key="load_sample_jd"):
                st.session_state.jd_df = sample_df
                st.success(f"✅ Loaded sample data: {len(sample_df)} records")
                st.rerun()
        except:
            pass
    else:
        st.info(f"📈 Current JD database: {len(st.session_state.jd_df)} records")
        
        # ===== Hyperparameters Section =====
        st.markdown("### ⚙️ Model Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Algorithm Settings**")
            model_name = st.selectbox("Model Type", ["Logistic Regression", "LinearSVC"], 
                                    index=0, key="role_model_type",
                                    help="Logistic Regression: Probabilistic, good for balanced datasets\nLinearSVC: Faster, good for large datasets")
            
            test_size = st.slider("Test Split Ratio", 0.1, 0.5, 0.2, 0.05, key="role_test_size",
                                help="Proportion of data used for testing (rest for training)")
            
            random_state = st.number_input("Random Seed", 0, 9999, 42, 1, key="role_random_state",
                                         help="For reproducible results")
            
            min_per_class = st.number_input("Min Samples per Role", 1, 10, 2, 1, key="role_min_per_class",
                                          help="Roles with fewer samples will be filtered out")
        
        with col2:
            st.markdown("**Text Processing Settings**")
            
            # N-gram range
            ngram_min = st.selectbox("N-gram Min", [1, 2], index=0, key="role_ngram_min",
                                   help="Minimum n-gram size (1=words, 2=bigrams)")
            ngram_max = st.selectbox("N-gram Max", [1, 2, 3], index=1, key="role_ngram_max",
                                   help="Maximum n-gram size")
            
            max_features = st.selectbox("Max Features", [5000, 10000, 25000, 50000, 100000], 
                                      index=3, key="role_max_features",
                                      help="Maximum number of TF-IDF features")
            
            # Advanced TF-IDF settings
            with st.expander("Advanced TF-IDF Settings", expanded=False):
                min_df = st.number_input("Min Document Frequency", 1, 10, 1, 1, key="role_min_df",
                                       help="Ignore terms that appear in fewer documents")
                max_df = st.slider("Max Document Frequency", 0.5, 1.0, 0.95, 0.05, key="role_max_df",
                                 help="Ignore terms that appear in more than this fraction of documents")
                use_sublinear_tf = st.checkbox("Use Sublinear TF", value=True, key="role_sublinear_tf",
                                             help="Apply sublinear tf scaling (1 + log(tf))")
        
        # Training button and results
        st.markdown("### 🚀 Training")
        col_train, col_load = st.columns([1, 1])
        
        with col_train:
            if st.button("🎯 Train Role Model", type="primary", key="train_role_btn"):
                with st.spinner("Training role classifier..."):
                    try:
                        # Prepare hyperparameters
                        ngram_range = (ngram_min, ngram_max)
                        
                        # Call training function with hyperparameters
                        res = train_role_model(
                            st.session_state.jd_df, 
                            text_col="jd_text", 
                            label_col="role_title",
                            test_size=test_size, 
                            random_state=random_state, 
                            model_type=model_name.lower().replace(" ", ""),
                            min_per_class=min_per_class,
                            ngram_range=ngram_range,
                            max_features=max_features,
                            min_df=min_df,
                            max_df=max_df,
                            use_sublinear_tf=use_sublinear_tf
                        )
                        
                        # Save results
                        st.session_state.role_model = res.model
                        st.session_state.role_model_metrics = res.metrics
                        save_model(res.model, role_path)
                        
                        st.success(f"✅ Model trained and saved → {role_path.name}")
                        
                        # Display comprehensive metrics
                        metrics = res.metrics
                        
                        # Key Performance Indicators
                        st.markdown("#### 📊 Model Performance")
                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        
                        val_metrics = metrics.get('validation', {}).get('summary', {})
                        with col_m1:
                            st.metric("Accuracy", f"{val_metrics.get('accuracy', 0):.3f}")
                        with col_m2:
                            st.metric("F1-Score (Macro)", f"{val_metrics.get('f1_macro', 0):.3f}")
                        with col_m3:
                            topk = metrics.get('validation', {}).get('topk', {})
                            st.metric("Top-3 Accuracy", f"{topk.get('top3_acc', 0):.3f}")
                        with col_m4:
                            balance = metrics.get('data_quality', {}).get('class_balance_score', 0)
                            st.metric("Class Balance", f"{balance:.3f}", 
                                    help="1.0 = perfectly balanced, 0.0 = highly imbalanced")
                        
                        # Data Quality Insights
                        st.markdown("#### 📈 Data Quality Analysis")
                        data_quality = metrics.get('data_quality', {})
                        col_d1, col_d2 = st.columns(2)
                        
                        with col_d1:
                            st.write("**Training Set:**")
                            st.write(f"• Samples: {data_quality.get('n_train', 0):,}")
                            st.write(f"• Test Samples: {data_quality.get('n_test', 0):,}")
                            
                            val_data = metrics.get('validation', {})
                            n_classes = val_data.get('model_complexity', {}).get('n_classes', 0)
                            st.write(f"• Number of Roles: {n_classes}")
                            
                        with col_d2:
                            st.write("**Quality Indicators:**")
                            if balance >= 0.8:
                                st.success(f"✅ Well-balanced dataset ({balance:.2f})")
                            elif balance >= 0.6:
                                st.warning(f"⚠️ Moderately imbalanced ({balance:.2f})")
                            else:
                                st.error(f"❌ Highly imbalanced dataset ({balance:.2f})")
                                
                            classes_with_pred = val_data.get('model_complexity', {}).get('classes_with_predictions', 0)
                            if classes_with_pred == n_classes:
                                st.success("✅ Model predicts all classes")
                            else:
                                st.warning(f"⚠️ Model only predicts {classes_with_pred}/{n_classes} classes")
                        
                        # Performance Breakdown
                        with st.expander("📋 Detailed Performance Metrics", expanded=False):
                            col_p1, col_p2 = st.columns(2)
                            
                            with col_p1:
                                st.markdown("**Classification Metrics:**")
                                for metric, value in val_metrics.items():
                                    if isinstance(value, float):
                                        st.write(f"• {metric.replace('_', ' ').title()}: {value:.4f}")
                                
                                st.markdown("**Top-K Accuracy:**")
                                topk_metrics = metrics.get('validation', {}).get('topk', {})
                                for k, acc in topk_metrics.items():
                                    if acc is not None:
                                        st.write(f"• {k.replace('_', '-').title()}: {acc:.4f}")
                            
                            with col_p2:
                                st.markdown("**Model Configuration:**")
                                hyperparams = metrics.get('hyperparameters', {})
                                for param, value in hyperparams.items():
                                    if param != 'model_type':
                                        st.write(f"• {param.replace('_', ' ').title()}: {value}")
                        
                        # Confusion Matrix Visualization
                        conf_matrix = val_data.get('confusion_matrix', [])
                        class_labels = val_data.get('class_labels', [])
                        
                        if conf_matrix and class_labels and len(class_labels) <= 20:  # Only show for reasonable number of classes
                            with st.expander("🎯 Confusion Matrix", expanded=False):
                                try:
                                    import numpy as np
                                    
                                    conf_df = pd.DataFrame(conf_matrix, 
                                                         index=class_labels, 
                                                         columns=class_labels)
                                    st.write("**Confusion Matrix (Actual vs Predicted):**")
                                    st.dataframe(conf_df, use_container_width=True)
                                    
                                    # Calculate per-class accuracy
                                    conf_array = np.array(conf_matrix)
                                    per_class_acc = np.diag(conf_array) / np.sum(conf_array, axis=1)
                                    
                                    st.write("**Per-Class Accuracy:**")
                                    for label, acc in zip(class_labels, per_class_acc):
                                        st.write(f"• {label}: {acc:.3f}")
                                        
                                except Exception as e:
                                    st.write("Could not display confusion matrix")
                        
                        # Recommendations
                        st.markdown("#### 💡 Recommendations")
                        recommendations = []
                        
                        if balance < 0.6:
                            recommendations.append("🔄 **Data Imbalance**: Consider collecting more data for underrepresented roles or using class balancing techniques")
                        
                        if val_metrics.get('accuracy', 0) < 0.7:
                            recommendations.append("📈 **Low Accuracy**: Try increasing max_features, adjusting n-gram range, or collecting more training data")
                        
                        if topk.get('top3_acc', 0) - val_metrics.get('accuracy', 0) > 0.2:
                            recommendations.append("🎯 **Good Top-K Performance**: Model shows promise for recommendation tasks even with moderate exact accuracy")
                        
                        if val_metrics.get('f1_macro', 0) < val_metrics.get('accuracy', 0) - 0.1:
                            recommendations.append("⚖️ **Class Imbalance Impact**: F1-score significantly lower than accuracy suggests class imbalance issues")
                        
                        if recommendations:
                            for rec in recommendations:
                                st.info(rec)
                        else:
                            st.success("🎉 Model performance looks good! No immediate issues detected.")
                            
                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")
                        st.error("Please check your data format and try again.")
        
        with col_load:
            if st.button("📂 Load Saved Model", key="load_role_btn"):
                model = load_model(role_path)
                if model is not None:
                    st.session_state.role_model = model
                    st.success(f"✅ Loaded {role_path.name}")
                else:
                    st.error("❌ No saved role model found")
        
        # Show current model status
        if st.session_state.role_model is not None:
            st.success("🎯 Role model is loaded and ready")
            if st.session_state.role_model_metrics:
                with st.expander("Current Model Performance Summary", expanded=False):
                    metrics = st.session_state.role_model_metrics
                    
                    # Display key metrics if available
                    if 'validation' in metrics:
                        val_metrics = metrics.get('validation', {}).get('summary', {})
                        col_s1, col_s2, col_s3 = st.columns(3)
                        with col_s1:
                            st.metric("Accuracy", f"{val_metrics.get('accuracy', 0):.3f}")
                        with col_s2:
                            st.metric("F1-Score", f"{val_metrics.get('f1_macro', 0):.3f}")
                        with col_s3:
                            balance = metrics.get('data_quality', {}).get('class_balance_score', 0)
                            st.metric("Balance Score", f"{balance:.3f}")
                    
                    # Show full metrics
                    with st.expander("Full Metrics", expanded=False):
                        st.text("Metrics computed successfully. Check individual metric displays above.")

    # ============ Training Content Classifier Section ============
    st.markdown("---")
    st.markdown("## 📚 Course Recommendation System")
    st.markdown("Build and train models for course recommendations")
    
    # Training Dataset Builder
    st.markdown("### 🌐 Build Training Dataset")
    with st.expander("Web Scrape Training Content", expanded=False):
        st.markdown("""
        **Build a comprehensive training dataset by scraping course platforms:**
        - Coursera, Udemy, edX, Udacity, Pluralsight, etc.
        - Extracts metadata: title, description, duration, price, rating
        - Uses skill-based search queries
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            per_skill = st.number_input("Courses per Skill", 1, 20, 5, 1, key="scrape_per_skill")
            workers = st.number_input("Parallel Workers", 1, 8, 3, 1, key="scrape_workers")
        
        with col2:
            engine = st.selectbox("Search Engine", ["auto", "serpapi", "ddg"], 
                                index=0, key="scrape_engine",
                                help="auto: use SerpAPI if available, else DuckDuckGo")
            timeout = st.number_input("Request Timeout (s)", 5, 30, 10, 1, key="scrape_timeout")
        
        col_build, col_status = st.columns([1, 1])
        
        with col_build:
            if st.button("🔄 Build Training Dataset", type="primary", key="build_training_btn"):
                import subprocess
                import sys
                
                with st.spinner("Building training dataset... This may take several minutes."):
                    try:
                        # Check if build script exists
                        script_path = "build_training_dataset/build_training_database.py"
                        skill_path = "build_training_dataset/skill_list.csv"
                        
                        if not os.path.exists(script_path):
                            st.error(f"Build script not found: {script_path}")
                        elif not os.path.exists(skill_path):
                            st.error(f"Skill list not found: {skill_path}")
                        else:
                            # Run the scraper
                            cmd = [
                                sys.executable, script_path,
                                "--skills", skill_path,
                                "--out", "training_database.csv",
                                "--per_skill", str(per_skill),
                                "--engine", engine,
                                "--workers", str(workers),
                                "--timeout", str(timeout)
                            ]
                            
                            # Show command being run
                            st.code(" ".join(cmd))
                            
                            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                            
                            if result.returncode == 0:
                                # Load the generated dataset
                                try:
                                    new_df = pd.read_csv("training_database.csv")
                                    st.session_state.training_df = new_df
                                    st.success(f"✅ Successfully built training database: {len(new_df)} courses")
                                    
                                    # Show summary
                                    st.write("**Dataset Summary:**")
                                    st.write(f"- Total courses: {len(new_df)}")
                                    st.write(f"- Skills covered: {new_df['skill'].nunique()}")
                                    st.write(f"- Providers: {new_df['provider'].nunique()}")
                                    
                                    # Provider breakdown
                                    provider_counts = new_df['provider'].value_counts()
                                    st.write("**Top Providers:**")
                                    st.bar_chart(provider_counts.head(8))
                                    
                                except Exception as e:
                                    st.error(f"Error loading generated dataset: {e}")
                            else:
                                st.error(f"Build failed with return code {result.returncode}")
                                st.error("STDOUT:")
                                st.code(result.stdout)
                                st.error("STDERR:")
                                st.code(result.stderr)
                                
                    except subprocess.TimeoutExpired:
                        st.error("Build process timed out (5 minutes). Try reducing --per_skill or --workers.")
                    except Exception as e:
                        st.error(f"Build error: {str(e)}")
        
        with col_status:
            # Show current scraping status
            if os.path.exists("training_database.csv"):
                try:
                    scraped_df = pd.read_csv("training_database.csv")
                    st.info(f"📁 Training database available:\n{len(scraped_df)} courses")
                    if st.button("Load Training Database", key="load_scraped_btn"):
                        st.session_state.training_df = scraped_df
                        st.success("✅ Loaded training database")
                        st.rerun()
                except:
                    pass
    
    # Data upload for Training Database CSV
    st.markdown("### 📊 Course Database Upload")
    st.info("Upload course database for recommendations (used in Guidance & Courses tab)")
    train_csv = st.file_uploader("Upload Training Database (CSV)", type=["csv"], key="train_upload",
                                help="CSV should contain course data: 'title', 'description', 'provider', 'skill', 'link', etc.")
    
    # Handle Training Database CSV upload
    if train_csv is not None:
        try:
            df = pd.read_csv(train_csv)
            st.session_state.training_df = df  # Load into training_df for course recommendations
            st.success(f"✅ Loaded training database: {len(df)} records")
            
            # Show data preview
            with st.expander("Preview Training Database", expanded=False):
                st.write("**Columns:**", list(df.columns))
                st.write("**Shape:**", df.shape)
                if 'skill' in df.columns:
                    skill_counts = df['skill'].value_counts()
                    st.write("**Skill Distribution:**")
                    st.bar_chart(skill_counts.head(10))
                if 'provider' in df.columns:
                    provider_counts = df['provider'].value_counts()
                    st.write("**Provider Distribution:**")
                    st.bar_chart(provider_counts.head(10))
                st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error loading training database CSV: {str(e)}")
    
    # Check current training database status
    if st.session_state.training_df.empty:
        st.warning("⚠️ No training database loaded. Build a dataset above, upload a CSV file, or ensure 'training_database.csv' exists in the project directory.")
        # Try to load sample data
        try:
            sample_df = pd.read_csv("training_database.csv")
            if st.button("Load Sample Training Database", key="load_sample_train"):
                st.session_state.training_df = sample_df
                st.success(f"✅ Loaded sample data: {len(sample_df)} records")
                st.rerun()
        except:
            pass
    else:
        st.info(f"📈 Current training database: {len(st.session_state.training_df)} records")
        
        # ===== Model Training Section =====
        st.markdown("### ⚙️ Model Training (Optional)")
        st.caption("Train a model on the uploaded training database for enhanced course recommendations")
        
        # Check if we have data suitable for training
        has_description = 'description' in st.session_state.training_df.columns
        has_skill = 'skill' in st.session_state.training_df.columns
        
        if not has_description or not has_skill:
            st.warning("⚠️ Training database needs 'description' and 'skill' columns for model training.")
            st.info("💡 You can still use the database for OpenAI recommendations without training a model.")
        else:
            # Use training_df for model training (copy from training_df)
            st.session_state.train_df = st.session_state.training_df.copy()
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Algorithm Settings**")
                model_name_c = st.selectbox("Model Type", ["Logistic Regression", "LinearSVC"], 
                                          index=0, key="course_model_type")
                
                test_size_c = st.slider("Test Split Ratio", 0.1, 0.5, 0.2, 0.05, key="course_test_size")
                
                random_state_c = st.number_input("Random Seed", 0, 9999, 42, 1, key="course_random_state")
                
                min_per_class_c = st.number_input("Min Samples per Skill", 1, 10, 2, 1, key="course_min_per_class")
            
            with col2:
                st.markdown("**Text Processing Settings**")
                
                # N-gram range
                ngram_min_c = st.selectbox("N-gram Min", [1, 2], index=0, key="course_ngram_min")
                ngram_max_c = st.selectbox("N-gram Max", [1, 2, 3], index=1, key="course_ngram_max")
                
                max_features_c = st.selectbox("Max Features", [5000, 10000, 25000, 50000], 
                                            index=2, key="course_max_features")
                
                # Advanced settings
                with st.expander("Advanced TF-IDF Settings", expanded=False):
                    min_df_c = st.number_input("Min Document Frequency", 1, 10, 1, 1, key="course_min_df")
                    max_df_c = st.slider("Max Document Frequency", 0.5, 1.0, 0.95, 0.05, key="course_max_df")
                    use_sublinear_tf_c = st.checkbox("Use Sublinear TF", value=True, key="course_sublinear_tf")
        
        # Training button and results
        st.markdown("### 🚀 Training")
        col_train_c, col_load_c = st.columns([1, 1])
        
        with col_train_c:
            if st.button("📚 Train Course Model", type="primary", key="train_course_btn"):
                with st.spinner("Training course classifier on training database..."):
                    try:
                        # Prepare hyperparameters
                        ngram_range_c = (ngram_min_c, ngram_max_c)
                        
                        # Call training function using train_df (copy of training_df)
                        res = train_training_model(
                            st.session_state.train_df, 
                            text_col="description", 
                            label_col="skill",
                            test_size=test_size_c, 
                            random_state=random_state_c, 
                            model_type=model_name_c.lower().replace(" ", ""),
                            min_per_class=min_per_class_c,
                            ngram_range=ngram_range_c,
                            max_features=max_features_c,
                            min_df=min_df_c,
                            max_df=max_df_c,
                            use_sublinear_tf=use_sublinear_tf_c
                        )
                        
                        # Save results
                        st.session_state.training_model = res.model
                        st.session_state.training_model_metrics = res.metrics
                        save_model(res.model, course_path)
                        
                        st.success(f"✅ Model trained and saved → {course_path.name}")
                        
                        # Display comprehensive metrics
                        metrics = res.metrics
                        
                        # Check if it's supervised or unsupervised
                        if 'validation' in metrics:
                            # Supervised model metrics
                            st.markdown("#### 📊 Model Performance")
                            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                            
                            val_metrics = metrics.get('validation', {}).get('summary', {})
                            with col_m1:
                                st.metric("Accuracy", f"{val_metrics.get('accuracy', 0):.3f}")
                            with col_m2:
                                st.metric("F1-Score (Macro)", f"{val_metrics.get('f1_macro', 0):.3f}")
                            with col_m3:
                                topk = metrics.get('validation', {}).get('topk', {})
                                st.metric("Top-3 Accuracy", f"{topk.get('top3_acc', 0):.3f}")
                            with col_m4:
                                balance = metrics.get('data_quality', {}).get('class_balance_score', 0)
                                st.metric("Skill Balance", f"{balance:.3f}")
                            
                            # Data Quality for supervised
                            st.markdown("#### 📈 Training Data Analysis")
                            data_quality = metrics.get('data_quality', {})
                            col_d1, col_d2 = st.columns(2)
                            
                            with col_d1:
                                st.write("**Dataset Size:**")
                                st.write(f"• Training: {data_quality.get('n_train', 0):,} courses")
                                st.write(f"• Testing: {data_quality.get('n_test', 0):,} courses")
                                
                                val_data = metrics.get('validation', {})
                                n_skills = val_data.get('model_complexity', {}).get('n_classes', 0)
                                st.write(f"• Unique Skills: {n_skills}")
                                
                            with col_d2:
                                st.write("**Quality Assessment:**")
                                if balance >= 0.8:
                                    st.success(f"✅ Well-balanced skills ({balance:.2f})")
                                elif balance >= 0.6:
                                    st.warning(f"⚠️ Some skill imbalance ({balance:.2f})")
                                else:
                                    st.error(f"❌ Highly imbalanced skills ({balance:.2f})")
                            
                            # Recommendations for supervised model
                            st.markdown("#### 💡 Model Insights")
                            if val_metrics.get('accuracy', 0) < 0.6:
                                st.warning("📊 Consider more training data or feature engineering for better skill classification")
                            elif val_metrics.get('accuracy', 0) > 0.8:
                                st.success("🎯 Excellent skill classification performance!")
                            else:
                                st.info("📈 Good performance - model should work well for course recommendations")
                                
                        else:
                            # Unsupervised model metrics
                            st.markdown("#### 📊 Model Information")
                            col_m1, col_m2 = st.columns(2)
                            
                            with col_m1:
                                st.metric("Training Items", f"{metrics.get('n_items', 0):,}")
                                st.write("**Mode:** Unsupervised (Similarity-based)")
                                
                            with col_m2:
                                hyperparams = metrics.get('hyperparameters', {})
                                st.write("**Configuration:**")
                                st.write(f"• Features: {hyperparams.get('max_features', 'Unknown'):,}")
                                st.write(f"• N-grams: {hyperparams.get('ngram_range', 'Unknown')}")
                            
                            st.info("📝 This model uses TF-IDF similarity for course recommendations without explicit skill classification.")
                        
                        # Show detailed metrics
                        with st.expander("📋 Detailed Technical Metrics", expanded=False):
                            st.text("Training metrics computed successfully. Check metric displays above.")
                            
                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")
                        st.error("Please check your data format and ensure 'description' column exists.")
        
        with col_load_c:
            if st.button("📂 Load Saved Model", key="load_course_btn"):
                model = load_model(course_path)
                if model is not None:
                    st.session_state.training_model = model
                    st.success(f"✅ Loaded {course_path.name}")
                else:
                    st.error("❌ No saved training model found")
        
        # Show current model status
        if st.session_state.training_model is not None:
            st.success("📚 Training model is loaded and ready")
            if st.session_state.training_model_metrics:
                with st.expander("Current Model Performance Summary", expanded=False):
                    metrics = st.session_state.training_model_metrics
                    
                    # Display key metrics if available
                    if 'validation' in metrics:
                        val_metrics = metrics.get('validation', {}).get('summary', {})
                        col_s1, col_s2 = st.columns(2)
                        with col_s1:
                            st.metric("Accuracy", f"{val_metrics.get('accuracy', 0):.3f}")
                        with col_s2:
                            st.metric("F1-Score", f"{val_metrics.get('f1_macro', 0):.3f}")
                    else:
                        st.write(f"**Mode:** {metrics.get('mode', 'Unknown')}")
                        st.write(f"**Items:** {metrics.get('n_items', 0):,}")
                    
                    # Show full metrics
                    with st.expander("Full Metrics", expanded=False):
                        st.text("Model metrics computed successfully. Check individual metric displays above.")
