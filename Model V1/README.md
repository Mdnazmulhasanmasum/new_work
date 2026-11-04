# TinyBrainNet-MT Streamlit Web Application

This repository contains the deployment code for **TinyBrainNet-MT**, an edge-ready multi-task deep learning diagnostic application for brain MRI classification, tumor segmentation, and attention map visualization.

## Files included:
- `app.py`: Main Streamlit application code.
- `requirements.txt`: Python package dependencies.
- `tinybrainnet_v1.onnx` *(Place your trained ONNX model weights file here)*

## Local Setup:
1. Extract the ZIP package.
2. Place your `tinybrainnet_v1.onnx` model weights inside this root folder.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```
