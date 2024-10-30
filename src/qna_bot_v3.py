import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np

# Download required NLTK data
nltk.download('punkt', quiet=True)

# Initialize models
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

def scrape_content(url):
    """Scrapes text content from a URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        soup = BeautifulSoup(response.text, 'html.parser')
        # Get text from paragraphs
        paragraphs = soup.find_all('p')
        text_content = ' '.join([p.get_text() for p in paragraphs])
        return text_content
    except Exception as e:
        raise Exception(f"Failed to scrape URL: {str(e)}")

def preprocess_text(text):
    """Preprocess the text into manageable chunks."""
    # Clean the text
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[[0-9]*\]', '', text)
    
    # Split into sentences and create chunks
    sentences = sent_tokenize(text)
    chunks = []
    chunk_size = 3
    
    for i in range(0, len(sentences), 2):
        chunk = ' '.join(sentences[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    return chunks

def get_answer(question, text_chunks):
    """Get the best answer from the text chunks."""
    try:
        # Encode question and chunks
        question_embedding = sentence_model.encode([question])
        chunk_embeddings = sentence_model.encode(text_chunks)
        
        # Calculate similarities
        similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
        
        # Get most relevant chunk
        best_chunk_idx = np.argmax(similarities)
        best_chunk = text_chunks[best_chunk_idx]
        
        # Get answer using QA model
        answer = qa_pipeline(
            question=question,
            context=best_chunk,
            max_answer_len=100
        )
        
        return {
            'text': answer['answer'],
            'context': best_chunk,
            'confidence': answer['score']
        }
    except Exception as e:
        return {
            'text': f"Sorry, I encountered an error: {str(e)}",
            'context': None,
            'confidence': 0
        }

def main():
    # Main function to run the chatbot.
    print("Welcome to the URL-based Q&A Chatbot!")
    
    # Get URL from user
    url = input("Please enter a URL to analyze: ").strip()
    
    try:
        # Load and process content
        print("\nLoading content from URL...")
        content = scrape_content(url)
        text_chunks = preprocess_text(content)
        print("Content loaded successfully!")
        
        # Start conversation loop
        print("\nYou can now ask questions about the content.")
        print("Type 'exit' to end the conversation")
        print("Type 'source' to see the context of the last answer")
        
        last_answer = None
        
        while True:
            # Get user input
            user_input = input("\nYou: ").strip().lower()
            
            # Check for exit command
            if user_input == 'exit':
                print("Goodbye!")
                break
            
            # Check for source command
            if user_input == 'source':
                if last_answer and last_answer['context']:
                    print("\nSource context:")
                    print(last_answer['context'])
                else:
                    print("No previous answer to show context for.")
                continue
            
            # Get and show answer
            last_answer = get_answer(user_input, text_chunks)
            print(f"\nChatbot: {last_answer['text']}")

            # Show confidence score
            if last_answer['confidence']:
                print(f"Confidence: {last_answer['confidence']:.2f}")
    
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Please try again with a different URL.")

if __name__ == "__main__":
    main()