# 📰 Fake News Detection Using Machine Learning

A machine learning based web application that classifies news articles as **Real** or **Fake**. The project uses Natural Language Processing (NLP) techniques to clean and vectorize text data, and a trained classification model to predict the authenticity of a given news article or headline.

## 🚀 Live Demo

Try the app here: **[Fake News Detection - Live App](https://fake-news-detection-using-machine-learning-madexrunjkpu2rimkzc.streamlit.app/)**

Simply paste a news headline or article text into the input box and click **Predict** to see whether the news is classified as Real or Fake.

## 📌 Features

- Clean and simple web interface built with **Streamlit**
- Text preprocessing (lowercasing, punctuation removal, stopword removal, stemming/lemmatization)
- Feature extraction using **TF-IDF Vectorization**
- Trained ML model for binary classification (Real vs Fake)
- Instant prediction with a single click
- Lightweight and easy to deploy

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** scikit-learn, pandas, numpy, nltk / spaCy
- **Web Framework:** Streamlit
- **Model:** Logistic Regression / Naive Bayes / Passive Aggressive Classifier (update as per your final model)
- **Vectorization:** TF-IDF

## 📂 Project Structure

```
fake-news-detection/
│
├── app.py                  # Streamlit application entry point
├── model.pkl                # Trained ML model
├── vectorizer.pkl            # Saved TF-IDF vectorizer
├── requirements.txt          # Project dependencies
├── dataset/                  # Training/testing dataset
├── notebooks/                 # Jupyter notebooks for EDA & model training
└── README.md                  # Project documentation
```

## ⚙️ Installation & Setup

1. Clone the repository
   ```bash
   git clone https://github.com/<your-username>/fake-news-detection.git
   cd fake-news-detection
   ```

2. Create a virtual environment (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install the dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app
   ```bash
   streamlit run app.py
   ```

5. Open the local URL shown in the terminal (usually `http://localhost:8501`) in your browser.

## 🧠 How It Works

1. **Data Collection** – A labeled dataset of real and fake news articles is used for training.
2. **Preprocessing** – Text is cleaned by removing stopwords, punctuation, and applying stemming/lemmatization.
3. **Feature Extraction** – The cleaned text is converted into numerical features using TF-IDF vectorization.
4. **Model Training** – A classification algorithm is trained on the vectorized data to learn patterns distinguishing fake news from real news.
5. **Prediction** – The trained model and vectorizer are loaded in the Streamlit app to classify new, unseen text in real time.

## 📊 Results

- Model Accuracy: *(add your accuracy score here)*
- Evaluation Metrics: Precision, Recall, F1-Score *(add values as applicable)*

## 🔮 Future Improvements

- Add support for detecting fake news from URLs/articles directly
- Use deep learning models (LSTM, BERT) for improved accuracy
- Multi-language support
- Confidence score display along with prediction

## 👩‍💻 Author

**Aarushi Tyagi**

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
