"""Train the spam classifier and save it to model.joblib.
Uses spam.csv (UCI/Kaggle SMS Spam Collection: columns v1=label, v2=text)
if present, otherwise a small built-in demo dataset."""
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

MODEL_PATH = "model.joblib"

DEMO_SPAM = [
    "WINNER!! You have won a $1000 gift card. Click http://bit.ly/claim-now to claim",
    "Congratulations! You've been selected for a free iPhone. Reply YES now",
    "URGENT: Your bank account is locked. Verify at http://secure-login.xyz now",
    "Free entry in our weekly prize draw. Text WIN to 80086",
    "You have won a lottery of 5,000,000. Send your bank details to claim",
    "Claim your free vacation today! Limited time offer, click here",
    "Your PayPal account is suspended. Confirm your password immediately",
    "Get cheap loans approved instantly, no credit check. Apply now",
    "Hot singles in your area waiting to chat. Click now",
    "Earn $5000 per week working from home. No experience needed",
    "Final notice: your package could not be delivered. Pay fee at http://track-parcel.top",
    "Dear customer, your KYC expired. Update now or account will be blocked",
    "Buy cheap meds online without prescription. Huge discount today",
    "You are our lucky winner! Call 09061701461 to collect your prize",
    "Act now! 100% free casino bonus. Register today and win big",
    "Your Netflix payment failed. Update billing info here http://netflix-billing.click",
    "Double your bitcoin in 24 hours. Guaranteed returns, invest now",
    "Exclusive offer: 90% off luxury watches. Order now before stock ends",
    "Your OTP is compromised. Share the code to secure your account",
    "Urgent!! Claim your tax refund now at http://192.168.4.7/refund",
]
DEMO_HAM = [
    "Hey, are we still meeting for lunch tomorrow?",
    "Can you send me the notes from today's class?",
    "I'll be home by 7, do you want me to pick up dinner?",
    "Happy birthday! Hope you have a wonderful day",
    "Meeting moved to 3 pm in the conference room",
    "Don't forget to bring your laptop charger tomorrow",
    "Thanks for helping me with the project, really appreciate it",
    "What time does the movie start tonight?",
    "Mom called, she wants you to call her back",
    "I reached the station, waiting near the main gate",
    "Let's catch up this weekend, it's been ages",
    "Please review the attached report and share your feedback",
    "Running late, traffic is terrible. Start without me",
    "Did you finish the assignment? Submission is on Friday",
    "Good morning! Have a great day at work",
    "Can you transfer the rent share to me when you get a chance?",
    "The doctor said the results look fine, no need to worry",
    "I'm at the grocery store, need anything?",
    "See you at the game tonight, I'll save you a seat",
    "Your appointment is confirmed for Monday at 10 am",
]

def load_data():
    if os.path.exists("spam.csv"):
        df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
        df.columns = ["label", "text"]
        df["y"] = (df["label"].str.lower() == "spam").astype(int)
        print(f"Loaded spam.csv with {len(df)} messages")
        return df
    print("spam.csv not found - using small demo dataset (download spam.csv for better accuracy)")
    return pd.DataFrame({"text": DEMO_SPAM + DEMO_HAM,
                         "y": [1] * len(DEMO_SPAM) + [0] * len(DEMO_HAM)})

def train():
    df = load_data()
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    if len(df) > 200:
        X_tr, X_te, y_tr, y_te = train_test_split(
            df["text"], df["y"], test_size=0.2, random_state=42, stratify=df["y"])
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        print(f"Test accuracy: {accuracy_score(y_te, pred):.3f}")
        print(classification_report(y_te, pred, target_names=["ham", "spam"]))
    pipe.fit(df["text"], df["y"])
    joblib.dump(pipe, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return pipe

if __name__ == "__main__":
    train()
