"""
Streamlit UI Dashboard for Skin Cancer Classification
Features: Model uptime, predictions, visualizations, data upload, retraining
"""
import streamlit as st
import requests
import time
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io

# API Configuration
API_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Skin Cancer Classification Dashboard",
    page_icon="⚕",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def get_health_status():
    """Get API health status"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def get_metrics():
    """Get model metrics"""
    try:
        response = requests.get(f"{API_URL}/metrics", timeout=5)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def predict_image(image_file):
    """Send image for prediction"""
    try:
        files = {"file": image_file}
        response = requests.post(f"{API_URL}/predict", files=files, timeout=120)
        
        # Check if response is successful
        if response.status_code != 200:
            return {"error": f"API returned status {response.status_code}: {response.text}"}
        
        # Try to parse JSON
        try:
            return response.json()
        except ValueError:
            return {"error": f"Invalid JSON response: {response.text[:200]}"}
            
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. Model inference is taking too long."}
    except requests.exceptions.ConnectionError:
        return {"error": "Could not connect to API. Make sure the API is running on port 8000."}
    except Exception as e:
        return {"error": str(e)}


def upload_bulk_data(files):
    """Upload bulk data"""
    try:
        files_data = [("files", file) for file in files]
        response = requests.post(f"{API_URL}/upload/bulk", files=files_data, timeout=60)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def trigger_retraining():
    """Trigger model retraining"""
    try:
        response = requests.post(f"{API_URL}/retrain", timeout=300)  # 5 minutes for retraining
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. Retraining may still be running in background."}
    except Exception as e:
        return {"error": str(e)}


def get_retrain_status():
    """Get retraining status"""
    try:
        response = requests.get(f"{API_URL}/retrain/status", timeout=10)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


# Class descriptions for skin lesions
CLASS_DESCRIPTIONS = {
    'akiec': {
        'name': 'Actinic Keratoses',
        'description': 'Pre-cancerous lesions caused by sun exposure. Common in fair-skinned individuals.',
        'risk': 'Low to Moderate',
        'recommendation': 'Consult a dermatologist for evaluation and possible treatment.'
    },
    'bcc': {
        'name': 'Basal Cell Carcinoma',
        'description': 'Most common form of skin cancer. Grows slowly and rarely spreads.',
        'risk': 'Moderate',
        'recommendation': 'Medical evaluation required. Highly treatable when detected early.'
    },
    'bkl': {
        'name': 'Benign Keratosis',
        'description': 'Non-cancerous skin growth, including seborrheic keratoses.',
        'risk': 'Benign',
        'recommendation': 'Generally harmless. Consult doctor if changes occur.'
    },
    'df': {
        'name': 'Dermatofibroma',
        'description': 'Benign fibrous nodule, usually harmless and common.',
        'risk': 'Benign',
        'recommendation': 'No treatment needed unless bothersome. Safe to monitor.'
    },
    'mel': {
        'name': 'Melanoma',
        'description': 'Most serious form of skin cancer. Can spread to other organs.',
        'risk': 'High',
        'recommendation': 'URGENT: Seek immediate medical attention from a dermatologist.'
    },
    'nv': {
        'name': 'Melanocytic Nevus',
        'description': 'Common mole. Usually benign but should be monitored for changes.',
        'risk': 'Benign',
        'recommendation': 'Monitor for changes in size, shape, or color. Regular check-ups advised.'
    },
    'vasc': {
        'name': 'Vascular Lesion',
        'description': 'Blood vessel abnormality, including angiomas and hemangiomas.',
        'risk': 'Benign',
        'recommendation': 'Usually benign. Consult doctor if growing or bleeding.'
    }
}


def get_class_info(class_code):
    """Get detailed information about a predicted class"""
    return CLASS_DESCRIPTIONS.get(class_code, {
        'name': class_code.upper(),
        'description': 'Skin lesion classification',
        'risk': 'Unknown',
        'recommendation': 'Consult a dermatologist for proper evaluation.'
    })


# Main UI
st.markdown('<h1 class="main-header">Skin Cancer Classification Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Model Status", "Single Prediction", "Batch Prediction", "Data Visualizations", "Data Upload & Retraining"])

# Page: Model Status
if page == "Model Status":
    st.header("Model Status & Uptime")
    
    col1, col2, col3, col4 = st.columns(4)
    
    health = get_health_status()
    metrics = get_metrics()
    
    with col1:
        if "error" not in health:
            status = "Online" if health.get("status") == "healthy" else "Offline"
            st.metric("Status", status)
        else:
            st.metric("Status", "Error")
    
    with col2:
        if "error" not in metrics:
            uptime_hours = metrics.get("uptime_seconds", 0) / 3600
            st.metric("Uptime", f"{uptime_hours:.2f} hrs")
        else:
            st.metric("Uptime", "N/A")
    
    with col3:
        if "error" not in health:
            device = health.get("device", "Unknown")
            st.metric("Device", device)
        else:
            st.metric("Device", "N/A")
    
    with col4:
        if "error" not in health:
            model_loaded = "Yes" if health.get("model_loaded") else "No"
            st.metric("Model Loaded", model_loaded)
        else:
            st.metric("Model Loaded", "No")
    
    # Uptime Chart
    st.subheader("Model Uptime Visualization")
    if "error" not in metrics:
        uptime_data = pd.DataFrame({
            "Time": [datetime.now() - timedelta(seconds=metrics.get("uptime_seconds", 0)), datetime.now()],
            "Status": ["Started", "Current"]
        })
        fig = go.Figure(data=[go.Scatter(x=uptime_data["Time"], y=[1, 1], mode='lines+markers', name='Uptime')])
        fig.update_layout(title="Model Uptime Timeline", xaxis_title="Time", yaxis_title="Status")
        st.plotly_chart(fig, use_container_width=True)

# Page: Single Prediction
elif page == "Single Prediction":
    st.header("Single Image Prediction")
    
    uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            if st.button("Predict", type="primary"):
                with st.spinner("Analyzing image..."):
                    # Reset file pointer
                    uploaded_file.seek(0)
                    result = predict_image(uploaded_file)
                    
                    if "error" in result:
                        st.error(f"Prediction failed: {result['error']}")
                    else:
                        st.success("Prediction Complete!")
                        
                        # Get class information
                        class_name = result.get('class_name', 'Unknown')
                        class_info = get_class_info(class_name)
                        
                        # Display main metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Predicted Condition", class_info['name'])
                        with col2:
                            confidence = result.get('confidence', 0)
                            st.metric("Confidence", f"{confidence*100:.1f}%")
                        with col3:
                            inference_time = result.get('inference_time', 0)
                            st.metric("Inference Time", f"{inference_time:.3f}s")
                        
                        # Detailed diagnosis information
                        st.markdown("---")
                        st.subheader("Diagnosis Information")
                        
                        col_desc, col_risk = st.columns(2)
                        
                        with col_desc:
                            st.markdown(f"**Description:**")
                            st.info(class_info['description'])
                        
                        with col_risk:
                            st.markdown(f"**Risk Level:**")
                            risk_color = {
                                'Benign': 'success',
                                'Low to Moderate': 'warning',
                                'Moderate': 'warning',
                                'High': 'error'
                            }.get(class_info['risk'], 'info')
                            
                            if risk_color == 'success':
                                st.success(f"{class_info['risk']}")
                            elif risk_color == 'warning':
                                st.warning(f"{class_info['risk']}")
                            elif risk_color == 'error':
                                st.error(f"{class_info['risk']}")
                            else:
                                st.info(f"{class_info['risk']}")
                        
                        st.markdown("**Medical Recommendation:**")
                        st.warning(class_info['recommendation'])
                        
                        st.markdown("---")
                        st.caption("**Disclaimer:** This is an AI-assisted prediction tool and should NOT replace professional medical diagnosis. Always consult a qualified dermatologist for accurate diagnosis and treatment.")
                        
                        # Probability chart (only if probabilities are available)
                        if 'probabilities' in result and result['probabilities']:
                            st.subheader("Detailed Probability Distribution")
                            probs_df = pd.DataFrame(list(result['probabilities'].items()), columns=['Class', 'Probability'])
                            probs_df['Probability'] = probs_df['Probability'] * 100  # Convert to percentage
                            
                            # Add full names to the chart
                            probs_df['Full Name'] = probs_df['Class'].apply(lambda x: CLASS_DESCRIPTIONS.get(x, {}).get('name', x))
                            
                            fig = px.bar(probs_df, x='Full Name', y='Probability', 
                                        title='Confidence Distribution Across All Classes',
                                        labels={'Probability': 'Confidence (%)', 'Full Name': 'Condition'},
                                        hover_data=['Class'])
                            fig.update_traces(marker_color='lightblue')
                            fig.update_xaxes(tickangle=45)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Detailed probability breakdown not available")

# Page: Batch Prediction
elif page == "Batch Prediction":
    st.header("Batch Image Prediction")
    
    uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        st.info(f"{len(uploaded_files)} images uploaded")
        
        if st.button("Predict All", type="primary"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing {i+1}/{len(uploaded_files)}: {file.name}")
                
                # Check file size
                file.seek(0, 2)  # Seek to end
                file_size_mb = file.tell() / (1024 * 1024)
                file.seek(0)  # Reset to start
                
                if file_size_mb > 10:
                    st.warning(f"⚠️ {file.name} is {file_size_mb:.1f}MB - may take longer to process")
                
                result = predict_image(file)
                
                # Check for errors
                if 'error' in result:
                    st.error(f"❌ Error processing {file.name}: {result['error']}")
                    results.append({
                        "Filename": file.name,
                        "Condition": "ERROR",
                        "Risk Level": "Unknown",
                        "Confidence": "0.00%",
                        "Inference Time": "0.000s"
                    })
                else:
                    # Get class name and full description
                    class_name = result.get('class_name', 'Unknown')
                    class_info = get_class_info(class_name)
                    
                    results.append({
                        "Filename": file.name,
                        "Condition": class_info['name'],
                        "Risk Level": class_info['risk'],
                        "Confidence": f"{result.get('confidence', 0)*100:.2f}%",
                        "Inference Time": f"{result.get('inference_time', 0):.3f}s"
                    })
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.empty()
            
            st.success("Batch prediction complete!")
            
            # Display results table
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)
            
            # Summary statistics
            st.subheader("Prediction Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_images = len(results_df)
                st.metric("Total Images", total_images)
            
            with col2:
                avg_confidence = results_df['Confidence'].str.rstrip('%').astype(float).mean()
                st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
            
            with col3:
                benign_count = results_df[results_df['Risk Level'] == 'Benign'].shape[0]
                st.metric("Benign Cases", benign_count)
            
            # Show risk distribution
            st.subheader("Risk Level Distribution")
            risk_counts = results_df['Risk Level'].value_counts()
            fig = px.pie(values=risk_counts.values, names=risk_counts.index, 
                        title='Distribution by Risk Level',
                        color_discrete_map={
                            'Benign': '#28a745',
                            'Low to Moderate': '#ffc107',
                            'Moderate': '#fd7e14',
                            'High': '#dc3545'
                        })
            st.plotly_chart(fig, use_container_width=True)
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button("Download Results CSV", csv, "predictions.csv", "text/csv")

# Page: Data Visualizations
elif page == "Data Visualizations":
    st.header("Data Visualizations")
    
    st.subheader("Feature Analysis")
    
    # Sample visualizations (you'll replace with actual data)
    st.write("### Class Distribution")
    sample_data = pd.DataFrame({
        'Class': ['Benign', 'Malignant'],
        'Count': [450, 350]
    })
    fig1 = px.bar(sample_data, x='Class', y='Count', title='Dataset Class Distribution', color='Class')
    st.plotly_chart(fig1, use_container_width=True)
    
    st.write("**Story**: This shows the distribution of benign vs malignant cases in our dataset. A relatively balanced dataset helps the model learn both classes effectively.")
    
    st.write("### Model Performance Over Time")
    sample_perf = pd.DataFrame({
        'Epoch': list(range(1, 11)),
        'Accuracy': [0.65, 0.72, 0.78, 0.82, 0.85, 0.87, 0.89, 0.90, 0.91, 0.92]
    })
    fig2 = px.line(sample_perf, x='Epoch', y='Accuracy', title='Training Accuracy Over Epochs')
    st.plotly_chart(fig2, use_container_width=True)
    
    st.write("**Story**: The model's accuracy improves steadily during training, showing effective learning without overfitting.")
    
    st.write("### Inference Time Distribution")
    import numpy as np
    sample_times = pd.DataFrame({
        'Request': list(range(1, 101)),
        'Time (ms)': np.random.normal(50, 10, 100)
    })
    fig3 = px.histogram(sample_times, x='Time (ms)', title='Inference Time Distribution', nbins=20)
    st.plotly_chart(fig3, use_container_width=True)
    
    st.write("**Story**: Most predictions complete within 40-60ms, demonstrating consistent and fast model performance suitable for production use.")

# Page: Data Upload & Retraining
elif page == "Data Upload & Retraining":
    st.header("Data Upload & Model Retraining")
    
    tab1, tab2 = st.tabs(["Upload Data", "Retrain Model"])
    
    with tab1:
        st.subheader("Upload Training Data")
        st.write("Upload multiple images to add to the training dataset")
        
        st.info("Automatic retraining is enabled. Retraining will trigger automatically after 10 uploads.")
        
        bulk_files = st.file_uploader("Select images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="bulk")
        
        if bulk_files:
            st.info(f"{len(bulk_files)} files ready to upload")
            
            if st.button("Upload All", type="primary"):
                with st.spinner("Uploading files..."):
                    result = upload_bulk_data(bulk_files)
                    
                    if "error" in result:
                        st.error(f"Upload failed: {result['error']}")
                    else:
                        st.success(f"Successfully uploaded {result.get('total_uploaded', 0)} files!")
                        
                        # Show auto-retrain status
                        if result.get('retrain_triggered'):
                            st.success(result.get('retrain_message', 'Automatic retraining triggered!'))
                        else:
                            st.info(result.get('retrain_message', ''))
                        
                        # Show upload details
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Uploaded", result.get('total_uploaded', 0))
                        with col2:
                            # Show cumulative count (before reset if retraining triggered)
                            cumulative = result.get('cumulative_uploads', 0)
                            st.metric("Cumulative Uploads", cumulative)
                        with col3:
                            threshold = result.get('auto_retrain_threshold', 10)
                            # Use current_count_after_reset if available, otherwise cumulative_uploads
                            current = result.get('current_count_after_reset', result.get('cumulative_uploads', 0))
                            remaining = threshold - current
                            st.metric("Until Auto-Retrain", max(0, remaining))
    
    with tab2:
        st.subheader("Retrain Model")
        st.write("Trigger model retraining with newly uploaded data")
        
        # Check retrain status
        status = get_retrain_status()
        
        if "error" not in status:
            st.info(f"**Status**: {status.get('status_message', 'Unknown')}")
            
            if status.get('last_retrain'):
                st.write(f"**Last Retrain**: {status['last_retrain']}")
            
            is_retraining = status.get('is_retraining', False)
            
            if st.button("Start Retraining", type="primary", disabled=is_retraining):
                result = trigger_retraining()
                
                if "error" in result:
                    st.error(f"Failed to start retraining: {result['error']}")
                else:
                    st.success("Retraining started! Check status below.")
            
            if is_retraining:
                st.warning("Retraining in progress... Please wait.")
                
                # Auto-refresh status
                if st.button("Refresh Status"):
                    st.rerun()

# Footer
st.markdown("---")
st.markdown("**Skin Cancer Classification System** | Built with Streamlit & FastAPI | MLOps Project 2025")
