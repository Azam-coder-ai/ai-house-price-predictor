# 🏠 AI House Price Predictor

An end-to-end Machine Learning and Data Engineering pipeline that predicts residential property values using classical regression algorithms. This project features a clean data preprocessing workflow and an interactive web interface.

## 📊 Project Architecture & Workflow
1. **Data Cleaning:** Handled missing inputs, managed boolean string encoding, and expanded categorical variables via One-Hot Encoding.
2. **Feature Engineering & Outlier Removal:** Applied the Interquartile Range (IQR) mathematical rule to filter out distorted pricing metrics, boosting model accuracy to **66.84%**.
3. **Feature Scaling:** Implemented Scikit-Learn's `MinMaxScaler` to balance structural dimensions safely between 0 and 1.
4. **Model Training:** Split data into an 80/20 Train-Test structure and optimized patterns using a `LinearRegression` engine.
5. **Interactive UI:** Serialized core assets into `.pkl` file binaries and constructed a web interface using the `Streamlit` framework.

## 🛠️ Technology Stack
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Web UI Framework:** Streamlit / Gradio

## 🚀 How to Run Locally

1. Clone this repository to your machine:
   ```bash
   git clone https://github.com/Azam-coder-ai/ai-house-price-predictor.git
   ```

2. Install the necessary developer libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the backend pipeline script to train the model architecture:
   ```bash
   python main.py
   ```

4. Launch the web application:
   ```bash
   streamlit run app.py
   ```
