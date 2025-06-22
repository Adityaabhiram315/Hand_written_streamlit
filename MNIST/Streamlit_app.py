import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import os
from io import BytesIO
import base64

# Page configuration
st.set_page_config(
    page_title="MNIST CNN Model Dashboard",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stSelectbox > div > div > select {
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data_and_model():
    """Load the model, test data, and training history"""
    try:
        # Get the current directory path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'mnist_cnn_model.h5')
        history_path = os.path.join(current_dir, 'training_history.pkl')
        
        # Load model
        if os.path.exists(model_path):
            model = load_model(model_path)
        else:
            st.error(f"Model file not found at: {model_path}")
            st.info("Please run 'python train_model.py' to generate the model file.")
            return None, None, None, None
        
        # Load MNIST test data
        (_, _), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        y_test = tf.keras.utils.to_categorical(y_test, 10)
        
        # Load training history if available
        history = None
        if os.path.exists(history_path):
            with open(history_path, 'rb') as f:
                history = pickle.load(f)
        else:
            # Create dummy history data for demonstration
            st.warning(f"Training history not found at: {history_path}")
            st.info("Using dummy data for training history. Run 'python train_model.py' for actual training history.")
            history = {
                'loss': [0.5, 0.3, 0.2, 0.15, 0.12, 0.1, 0.08, 0.07, 0.06, 0.05],
                'accuracy': [0.85, 0.92, 0.95, 0.96, 0.97, 0.975, 0.98, 0.982, 0.985, 0.987],
                'val_loss': [0.6, 0.35, 0.25, 0.2, 0.18, 0.16, 0.15, 0.14, 0.13, 0.12],
                'val_accuracy': [0.82, 0.90, 0.93, 0.94, 0.95, 0.955, 0.96, 0.965, 0.97, 0.972]
            }
        
        return model, X_test, y_test, history
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.error(f"Current working directory: {os.getcwd()}")
        st.error(f"Files in current directory: {os.listdir('.')}")
        return None, None, None, None

def plot_training_metrics(history):
    """Create interactive training metrics plots using Plotly"""
    epochs = list(range(1, len(history['loss']) + 1))
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Training & Validation Loss', 'Training & Validation Accuracy'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Loss plot
    fig.add_trace(
        go.Scatter(x=epochs, y=history['loss'], 
                  mode='lines+markers', name='Training Loss',
                  line=dict(color='#ff6b6b', width=3),
                  marker=dict(size=8)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=epochs, y=history['val_loss'], 
                  mode='lines+markers', name='Validation Loss',
                  line=dict(color='#4ecdc4', width=3),
                  marker=dict(size=8)),
        row=1, col=1
    )
    
    # Accuracy plot
    fig.add_trace(
        go.Scatter(x=epochs, y=history['accuracy'], 
                  mode='lines+markers', name='Training Accuracy',
                  line=dict(color='#45b7d1', width=3),
                  marker=dict(size=8)),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=epochs, y=history['val_accuracy'], 
                  mode='lines+markers', name='Validation Accuracy',
                  line=dict(color='#f9ca24', width=3),
                  marker=dict(size=8)),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="Training and Validation Metrics",
            x=0.5,
            font=dict(size=24, color="#2c3e50")
        ),
        showlegend=True,
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Update x-axis
    fig.update_xaxes(title_text="Epoch", showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    # Update y-axis
    fig.update_yaxes(title_text="Loss", row=1, col=1, showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(title_text="Accuracy", row=1, col=2, showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

def create_confusion_matrix(model, X_test, y_test):
    """Create confusion matrix using Plotly"""
    # Get predictions
    predictions = model.predict(X_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create heatmap
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=[str(i) for i in range(10)],
        y=[str(i) for i in range(10)],
        color_continuous_scale="Blues",
        title="Confusion Matrix"
    )
    
    # Add text annotations
    for i in range(10):
        for j in range(10):
            fig.add_annotation(
                x=j, y=i,
                text=str(cm[i, j]),
                showarrow=False,
                font=dict(color="white" if cm[i, j] > cm.max()/2 else "black", size=12)
            )
    
    fig.update_layout(
        title=dict(x=0.5, font=dict(size=20)),
        width=600,
        height=500
    )
    
    return fig

def show_digit_samples(digit, model, X_test, y_test):
    """Generate and display 5 samples of a specific digit"""
    true_labels = np.argmax(y_test, axis=1)
      # Find digit examples
    digit_indices = np.where(true_labels == digit)[0]
    if len(digit_indices) == 0:
        st.error(f"No examples of digit {digit} found!")
        return
    
    # Get best 5 samples based on model confidence
    digit_images = X_test[digit_indices]
    predictions = model.predict(digit_images, verbose=0)
    confidence_scores = predictions[:, digit]
    top_5_indices = np.argsort(confidence_scores)[-5:][::-1]
    selected_indices = digit_indices[top_5_indices]
    
    # Create subplot figure with increased spacing and size
    fig = make_subplots(
        rows=1, cols=5,        subplot_titles=[f'Sample {i+1}<br>Confidence: {confidence_scores[top_5_indices[i]]:.3f}' 
                       for i in range(5)],
        specs=[[{"type": "xy"}] * 5],
        horizontal_spacing=0.01,  # Minimal spacing between subplots for maximum image space
        vertical_spacing=0.1
    )
    
    for i, idx in enumerate(selected_indices):
        img = X_test[idx].reshape(28, 28)
        
        fig.add_trace(
            go.Heatmap(
                z=img,
                colorscale='gray',                showscale=False,
                hovertemplate='Pixel value: %{z:.3f}<br>Row: %{y}<br>Col: %{x}<extra></extra>',
                zmin=0,
                zmax=1
            ),
            row=1, col=i+1
        )
    
    fig.update_layout(
        title=dict(
            text=f'Top 5 Confident Predictions for Digit {digit}',
            x=0.5,
            font=dict(size=24, color="#2c3e50")
        ),
        height=700,  # Increased height for better visibility
        width=1400,  # Increased width to make images larger
        showlegend=False,
        margin=dict(l=20, r=20, t=120, b=40),  # Adjusted margins for better spacing
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Remove axis labels and ticks, ensure square aspect ratio
    for i in range(1, 6):
        fig.update_xaxes(
            showticklabels=False, 
            showgrid=False, 
            zeroline=False,
            scaleanchor=f"y{i}",  # Make aspect ratio square
            scaleratio=1,
            row=1, col=i
        )
        fig.update_yaxes(
            showticklabels=False, 
            showgrid=False, 
            zeroline=False,
            autorange="reversed",  # Flip y-axis to match image orientation
            row=1, col=i
        )
    
    return fig

def create_accuracy_by_digit(model, X_test, y_test):
    """Create accuracy by digit chart"""
    predictions = model.predict(X_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    accuracies = []
    for digit in range(10):
        digit_mask = y_true == digit
        if np.sum(digit_mask) > 0:
            digit_accuracy = np.mean(y_pred[digit_mask] == y_true[digit_mask])
            accuracies.append(digit_accuracy)
        else:
            accuracies.append(0)
    
    fig = px.bar(
        x=list(range(10)),
        y=accuracies,
        labels={'x': 'Digit', 'y': 'Accuracy'},
        title='Accuracy by Digit',
        color=accuracies,
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        title=dict(x=0.5, font=dict(size=20)),
        showlegend=False,
        height=400
    )
    
    return fig

# Main app
def main():
    st.markdown('<h1 class="main-header">🔢 MNIST CNN Model Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    model, X_test, y_test, history = load_data_and_model()
    
    if model is None:
        st.error("Failed to load model or data. Please ensure 'mnist_cnn_model.h5' exists.")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🎯 Digit Generator", "📊 Model Metrics & Analytics"]    )
    
    if page == "🎯 Digit Generator":
        st.header("🎯 Digit Sample Generator")
        st.markdown("Enter a digit (0-9) to see the top 5 most confident predictions from the model:")
        
        # Use full width for better image display
        digit = st.selectbox(
            "Select digit:",
            options=list(range(10)),
            index=0,
            help="Choose a digit from 0 to 9"
        )
        
        generate_button = st.button("Generate Samples", type="primary")
        
        # Display images in full width when button is clicked
        if generate_button:
            with st.spinner("Generating samples..."):
                fig = show_digit_samples(digit, model, X_test, y_test)
                if fig:
                    # Use full container width and make it responsive
                    st.plotly_chart(fig, use_container_width=True, config={
                        'displayModeBar': True,
                        'displaylogo': False,
                        'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
                    })
    
    elif page == "📊 Model Metrics & Analytics":
        st.header("📊 Model Performance Analytics")
        
        # Training metrics
        st.subheader("📈 Training History")
        if history:
            fig_metrics = plot_training_metrics(history)
            st.plotly_chart(fig_metrics, use_container_width=True)
            
            # Display final metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric(
                    label="Final Training Loss",
                    value=f"{history['loss'][-1]:.4f}",
                    delta=f"{history['loss'][-1] - history['loss'][0]:.4f}"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric(
                    label="Final Training Accuracy",
                    value=f"{history['accuracy'][-1]:.1%}",
                    delta=f"{(history['accuracy'][-1] - history['accuracy'][0]):.1%}"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric(
                    label="Final Validation Loss",
                    value=f"{history['val_loss'][-1]:.4f}",
                    delta=f"{history['val_loss'][-1] - history['val_loss'][0]:.4f}"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric(
                    label="Final Validation Accuracy",
                    value=f"{history['val_accuracy'][-1]:.1%}",
                    delta=f"{(history['val_accuracy'][-1] - history['val_accuracy'][0]):.1%}"
                )
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Additional analytics
        st.subheader("🔍 Detailed Analysis")
        
        tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Accuracy by Digit", "Model Summary"])
        
        with tab1:
            with st.spinner("Generating confusion matrix..."):
                cm_fig = create_confusion_matrix(model, X_test, y_test)
                st.plotly_chart(cm_fig, use_container_width=True)
        
        with tab2:
            with st.spinner("Calculating accuracy by digit..."):
                acc_fig = create_accuracy_by_digit(model, X_test, y_test)
                st.plotly_chart(acc_fig, use_container_width=True)
        
        with tab3:
            st.subheader("Model Architecture")
            
            # Get model summary
            summary_str = []
            model.summary(print_fn=summary_str.append)
            summary_text = '\n'.join(summary_str)
            
            st.code(summary_text, language='text')
            
            # Model info
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Model Parameters:**")
                total_params = model.count_params()
                st.write(f"Total parameters: {total_params:,}")
            
            with col2:
                st.markdown("**Input Shape:**")
                st.write(f"Input shape: {model.input_shape}")

if __name__ == "__main__":
    main()
