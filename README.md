# Unsupervised Industrial Anomaly Detection

A production-grade PyTorch pipeline that detects manufacturing defects without ever being trained on defective data. It utilizes PatchCore principles, extracting feature maps from a pre-trained WideResNet-50 and using Linear Algebra (K-Nearest Neighbors) to calculate spatial divergence.

## Prerequisites
1. **Python 3.11** or higher

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Industrial_Anomaly_Detection
   ```

2. **Set up the Python environment:**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Usage

1. Activate your virtual environment:
   ```bash
   source venv/bin/activate
   ```
2. Run the application:
   ```bash
   streamlit run app.py
   ```
3. Upload a "Normal" (defect-free) image of a manufactured part. The model extracts its ResNet features and establishes a mathematical baseline.
4. Upload a "Test" image (potentially defective). The system computes the K-Nearest Neighbor matrix distance and outputs a heatmap highlighting exact defect locations.
