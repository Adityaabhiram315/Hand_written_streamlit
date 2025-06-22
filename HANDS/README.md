# MNIST CNN Model Dashboard

A comprehensive Streamlit web application for visualizing MNIST CNN model performance and generating digit samples.

## Features

### 🎯 Digit Generator Page
- **Interactive Digit Selection**: Choose any digit (0-9) from a dropdown menu
- **Top 5 Confident Predictions**: View the 5 test samples where the model is most confident
- **Confidence Scores**: Each sample displays the model's confidence score
- **Beautiful Visualizations**: Grayscale heatmaps with hover information

### 📊 Model Metrics & Analytics Page
- **Training History**: Interactive line plots showing:
  - Training and Validation Loss over epochs
  - Training and Validation Accuracy over epochs
- **Performance Metrics**: Key performance indicators with delta changes
- **Confusion Matrix**: Interactive heatmap showing prediction accuracy
- **Accuracy by Digit**: Bar chart showing model performance for each digit
- **Model Summary**: Detailed architecture and parameter information

## Setup and Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model** (if needed):
   ```bash
   python train_model.py
   ```
   This will create:
   - `mnist_cnn_model.h5` - The trained model
   - `training_history.pkl` - Training metrics history

3. **Run the Streamlit App**:
   ```bash
   streamlit run Streamlit_app.py
   ```

## File Structure

```
MNIST/
├── Streamlit_app.py          # Main Streamlit application
├── train_model.py            # Training script with history tracking
├── mnist_cnn_model.h5        # Trained CNN model
├── training_history.pkl      # Training metrics history
├── Mnist_training.ipynb      # Jupyter notebook for model training
├── Visualizing_predDigits.ipynb  # Jupyter notebook for digit visualization
└── requirements.txt          # Python dependencies
```

## How to Use

### Digit Generator
1. Navigate to the "🎯 Digit Generator" page using the sidebar
2. Select a digit (0-9) from the dropdown menu
3. Click "Generate Samples" to view the top 5 confident predictions
4. Hover over the images to see pixel values

### Model Analytics
1. Navigate to the "📊 Model Metrics & Analytics" page
2. View the training history charts showing loss and accuracy progression
3. Check the confusion matrix to see prediction accuracy for each digit
4. Analyze accuracy by digit to identify which digits the model handles best
5. Review the model architecture in the Model Summary tab

## Model Architecture

The CNN model consists of:
- 2 Convolutional layers (32 and 64 filters)
- 2 MaxPooling layers
- Dropout layers for regularization (0.25 and 0.5)
- 1 Dense hidden layer (128 units)
- Output layer with 10 units (softmax activation)

## Technologies Used

- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **TensorFlow/Keras**: Deep learning model
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning utilities

## Performance

The model typically achieves:
- Training Accuracy: ~98.7%
- Validation Accuracy: ~97.2%
- Test Accuracy: ~97%+

## Customization

You can easily customize the application by:
- Modifying the model architecture in `train_model.py`
- Adjusting visualization colors and styles in `Streamlit_app.py`
- Adding new metrics or charts to the analytics page
- Changing the number of samples shown in the digit generator

## Troubleshooting

1. **Model not found**: Ensure `mnist_cnn_model.h5` exists. Run `python train_model.py` if needed.
2. **No training history**: The app will create dummy data if `training_history.pkl` doesn't exist.
3. **Import errors**: Make sure all dependencies are installed with `pip install -r requirements.txt`.

## License

This project is open source and available under the MIT License.
