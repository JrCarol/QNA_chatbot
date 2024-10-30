# src/qna_bot.py

# Import necessary libraries
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer  # Import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained Question Answering model
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def scrape_content(url):
    ## Scrapes text content from a URL and returns it as a single string.
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text_content = ' '.join([p.get_text() for p in paragraphs])
    return text_content

def preprocess_text(text):
    ## Tokenizes text into sentences for easier Q&A and returns a list.
    sentences = sent_tokenize(text)  # This should return a list of sentences
    return sentences

def generate_qna(url):
    ## Scrapes content from the URL and prepares it for the Q&A model.
    text_content = scrape_content(url)
    sentences = preprocess_text(text_content)
    return sentences

def find_best_answer(question, sentences, top_n=3, similarity_threshold=0.2):

    ## Finds the top N sentences that are most similar to the user's question.
    ## Filters sentences based on the similarity threshold and concatenates them.
    # Fit TF-IDF on question and sentences
    vectorizer = TfidfVectorizer().fit_transform([question] + sentences)
    similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()

    # Get sorted indices of sentences by similarity score in descending order
    sorted_indices = similarity.argsort()[::-1]
    
    # Select the top N sentences that meet the similarity threshold
    selected_sentences = [
        sentences[i] for i in sorted_indices if similarity[i] > similarity_threshold][:top_n]

    # Combine selected sentences or return a fallback message
    if selected_sentences:
        return " ".join(selected_sentences)
    else:
        return "I'm sorry, I couldn't find a relevant answer in the content."


def chatbot_interaction(url):
    sentences = generate_qna(url)
    print("Welcome to the URL-based Q&A chatbot! Ask me anything about the content at the URL provided.")
    print("Type 'exit' to end the conversation.")
    
    while True:
        question = input("You: ")
        if question.lower() == "exit":
            print("Chatbot: Thank you for chatting. Goodbye!")
            break
        answer = find_best_answer(question, sentences, top_n=3, similarity_threshold=0.2)
        print("Chatbot:", answer)

# Example usage
if __name__ == "__main__":
    example_url = "https://en.wikipedia.org/wiki/Seventeen_(South_Korean_band)"  # Replace with the URL you want to scrape
    chatbot_interaction(example_url)
