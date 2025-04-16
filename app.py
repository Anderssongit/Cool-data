from flask import Flask, jsonify, render_template
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import time
import random

# Selenium imports for scraping
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

app = Flask(__name__)
# Global variable to store sentiment summary (this gets updated on startup)
sentiment_result = {}

def scrape_posts():
    """Step 1: Use Selenium to scrape posts and save them to an Excel file."""
    driver = webdriver.Firefox()
    driver.get('https://truthsocial.com/@realDonaldTrump')
    driver.maximize_window()

    collected_posts = set()
    scroll_attempts = 15
    posts_per_scroll = 20

    def generate_xpaths(post_num):
        # Three sets of XPath variations
        return [
            f'/html/body/div/div[1]/div/div[2]/div[1]/div/div[2]/main/div[{i}]/div/div[2]/div/div[2]/div/div[2]/div[{post_num}]/div[1]/div/div/div[2]/div[1]/div/div/p/p'
            for i in range(1,5)
        ] + [
            f'/html/body/div/div[1]/div/div[2]/div[1]/div/div[2]/main/div/div/div[{i}]/div/div[2]/div/div[2]/div[{post_num}]/div[1]/div/div/div[2]/div[1]/div/div/p/p'
            for i in range(1,5)
        ] + [
            f'/html/body/div/div[1]/div/div[2]/div[{i}]/div/div[2]/main/div/div/div[2]/div/div[2]/div/div[2]/div[{post_num}]/div[1]/div/div/div[2]/div[1]/div/div/p/p'
            for i in range(1,3)
        ]

    def smart_scroll():
        """Scroll gradually to simulate human behavior."""
        scroll_amount = random.randint(300, 300)
        for _ in range(3):  # perform 3 mini scrolls
            driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
            time.sleep(random.uniform(0.5, 1.2))

    # Try clicking an “Accept” cookie button if present
    try:
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Accept')]"))
        ).click()
    except:
        pass

    # Perform scrolling and try to scrape posts using multiple XPath patterns
    for scroll in range(scroll_attempts):
        for post_num in range(1, posts_per_scroll + 1):
            for xpath in generate_xpaths(post_num):
                try:
                    element = driver.find_element(By.XPATH, xpath)
                    if element.is_displayed():
                        text = element.text.strip()
                        if text and text not in collected_posts:
                            collected_posts.add(text)
                except Exception:
                    continue
        smart_scroll()
        time.sleep(2 + (scroll * 0.3))

    driver.quit()

    # Save the scraped posts to an Excel file
    df = pd.DataFrame({"Text": list(collected_posts)})
    df["Post #"] = df.index + 1
    df = df[["Post #", "Text"]]
    excel_path = "trump_posts_aggressive.xlsx"
    df.to_excel(excel_path, index=False)
    return excel_path

def run_sentiment_analysis(excel_file):
    """Step 2: Load scraped posts, run sentiment analysis, and compute average sentiment."""
    # Load and filter data
    df = pd.read_excel(excel_file)
    df_filtered = df[~df["Text"].str.contains("youtube", case=False)]
    top_posts = df_filtered.head(5)["Text"].tolist()
    
    # Load sentiment analysis model (using SST-2; it has two classes: negative and positive)
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def analyze_sentiment(text):
        encoded_input = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = model(**encoded_input)
        probs = F.softmax(outputs.logits, dim=1)[0]
        negative_prob = probs[0].item()
        positive_prob = probs[1].item()
        all_probs = {"negative": negative_prob, "positive": positive_prob}
        predicted_label = max(all_probs, key=all_probs.get)
        return predicted_label, all_probs

    results = []
    total_negative = 0
    total_positive = 0
    for i, post in enumerate(top_posts, start=1):
        sentiment_label, probabilities = analyze_sentiment(post)
        results.append({
            "Post #": i,
            "Text": post,
            "Predicted Sentiment": sentiment_label,
            "Negative Prob": probabilities["negative"],
            "Positive Prob": probabilities["positive"]
        })
        total_negative += probabilities["negative"]
        total_positive += probabilities["positive"]

    # Calculate averages across the top posts
    avg_negative = total_negative / len(top_posts) if top_posts else 0
    avg_positive = total_positive / len(top_posts) if top_posts else 0
    overall = "positive" if avg_positive >= avg_negative else "negative"
    summary = {
        "avg_negative": avg_negative,
        "avg_positive": avg_positive,
        "overall_sentiment": overall
    }
    
    # Optionally save detailed results to Excel
    results_df = pd.DataFrame(results)
    results_df.to_excel("trump_posts_finbert_sentiment_clean.xlsx", index=False)
    return summary

@app.route('/')
def index():
    """Serve the main web page."""
    return render_template("index.html")

@app.route('/api/sentiment')
def api_sentiment():
    """Endpoint that returns the sentiment analysis summary as JSON."""
    return jsonify(sentiment_result)

def update_results():
    """Run the complete pipeline and return the sentiment summary."""
    excel_file = scrape_posts()
    summary = run_sentiment_analysis(excel_file)
    return summary

if __name__ == '__main__':
    # Run the pipeline on startup and store the result globally
    sentiment_result = update_results()
    # Start the Flask server
    app.run(debug=True)
