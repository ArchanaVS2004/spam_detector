# 🛡️ Spam Message & Link Detector

An interactive web dashboard that detects **spam messages** and **malicious or phishing links** in real time. It combines a machine-learning text classifier with a rule-based link analyzer and presents the results in a clean, visual dashboard built with Plotly Dash.

**🔗 Live demo:** [https://spam-detector-vuwf.onrender.com](https://spam-detector-vuwf.onrender.com)

## ✨ Features

- **Spam classification:** TF-IDF + Logistic Regression model that gives a spam probability for any message.
- **Link analysis:** automatically extracts URLs and scores each one for phishing risk, with a plain-English explanation of every red flag.
- **Spam risk gauge:** a 0–100% dial combining text and link risk into one score (green / amber / red zones).
- **Explainable results:** highlights the words and phrases that pushed the model toward "spam".
- **Live statistics:** summary cards (messages scanned, spam detected, spam rate, links found), a spam vs not-spam doughnut chart and a per-message risk timeline.
- **History tools:** session history table, one-click clear, and CSV export.
- **Quick demo buttons:** built-in spam, phishing and normal example messages.

## 🧠 How It Works

### 1. Text classifier
Messages are converted to numeric features using **TF-IDF** (unigrams and bigrams, English stop words removed) and classified by a **Logistic Regression** model with balanced class weights. The output is a spam probability from 0–100%.

### 2. Link analyzer
Every URL found in a message is scored using heuristics:

| Red flag | Points |
|---|---|
| Raw IP address instead of a domain | +3 |
| `@` symbol in the address | +3 |
| Punycode / look-alike characters (`xn--`) | +3 |
| URL shortener (bit.ly, tinyurl, etc.) | +2 |
| Suspicious domain ending (`.xyz`, `.top`, `.click`, ...) | +2 |
| Phishing keywords (login, verify, bank, password, ...) | +1 to +2 |
| Not using HTTPS | +1 |
| Many hyphens or subdomains | +1 each |
| Very long URL | +1 |

A score of **4 or more** is labelled *Dangerous*, **2–3** *Suspicious*, and below 2 *Looks OK*.

### 3. Final risk score
```
risk = max(text_spam_probability, highest_link_score × 15)   (capped at 100)
verdict = SPAM if risk ≥ 50 else NOT SPAM
```
A single dangerous link can therefore mark a message as spam even if the wording looks harmless.

## 🧰 Tech Stack

- **Python 3.9+**
- **Plotly Dash:** dashboard and interactivity
- **scikit-learn:** TF-IDF vectorizer and Logistic Regression
- **pandas / joblib:** data handling and model persistence
- **Gunicorn:** production server
- **Render:** hosting

## 📁 Project Structure

```
spam-detector/
├── app.py             # Dash dashboard, link analyzer and callbacks
├── train_model.py     # Trains the classifier and saves model.joblib
├── requirements.txt   # Python dependencies
├── Procfile           # Process definition for hosting platforms
├── render.yaml        # Render deployment blueprint
├── spam.csv           # (optional) training dataset, see below
└── README.md
```

## 🚀 Run Locally

**Prerequisites:** Python 3.9 or newer and `pip`.

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/spam-detector.git
cd spam-detector

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Recommended) add the dataset, see "Training Data" below

# 5. Train the model (optional, app.py trains automatically on first run)
python train_model.py

# 6. Start the dashboard
python app.py
```


## 📊 Training Data

For best accuracy, download the **SMS Spam Collection** dataset (available on [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) and save it as `spam.csv` in the project root. The file should contain two columns:

| Column | Meaning |
|---|---|
| `v1` | Label: `ham` or `spam` |
| `v2` | Message text |

If `spam.csv` is not present, the app falls back to a small built-in demo dataset, which is fine for testing but much less accurate. When the real dataset is used, `train_model.py` holds out 20% of the data and prints accuracy and a classification report.

## ⚠️ Limitations

- The link analyzer is **heuristic-based**; it does not check live reputation databases, so it can miss new phishing domains or flag harmless ones.
- The text model is only as good as its training data. It is trained on English SMS messages and may perform worse on other languages, emails or social media text.
- Treat results as a **decision aid**, not a guarantee of safety.
- Messages are processed on the server for analysis and are not stored; history exists only in your browser session.


## 🤝 Contributing

Contributions, issues and feature requests are welcome. Fork the repo, create a feature branch, and open a pull request.

## 📄 License

This project is licensed under the MIT License. Add a `LICENSE` file to the repository to make this official.

## 👤 Author

**Archana V S**
GitHub: [@ArchanaVS2004][https://github.com/ArchanaVS2004]
