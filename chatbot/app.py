from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import pandas as pd
import praw
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

socketio = SocketIO(app)

# Load your pre-trained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.Client()
collection = client.get_or_create_collection("sex-ed-documents")

# Load your GPT-2 model
generator = pipeline('text-generation', model='gpt2')

# Step 2: Function to scrape websites and extract text
def scrape_website(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            text = ' '.join([para.get_text() for para in paragraphs])
            return text
        else:
            return None
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

# List of URLs to scrape
urls = [
    "https://scarleteen.com",
    "https://www.plannedparenthood.org/learn/teens",
    "https://www.sexandu.ca/",
    "https://www.cdc.gov/sexualhealth/"
]

# Step 3: Scrape and process data from websites
for idx, url in enumerate(urls):
    print(f"Scraping {url}")
    text = scrape_website(url)
    if text:
        embedding = model.encode(text)
        # Store document in ChromaDB collection
        collection.add(
            documents=[text],
            metadatas=[{"url": url}],
            ids=[f"doc{idx}"]
        )
        print(f"Stored data from {url}")
    else:
        print(f"Failed to scrape {url}")

# Step 4: Read and process data from Excel (if you have relevant sex ed data)
# Uncomment and modify the following lines if you have an Excel file with relevant data
excel_path = os.path.join("data", "gurus_reddif.json.xlsx")  # Replace with the path to your Excel file
df = pd.read_excel(excel_path)
df.columns = [col.strip() for col in df.columns]
df = df.fillna("")
for idx, row in df.iterrows():
  question = row['Concept']
  text = row['Description']
  url = 'csv_source'
  embedding = model.encode(text)
  collection.add(
    documents=[text],
    metadatas=[{"url": url, "question": question}],
    ids=[f"csv_doc{idx}"]
  )
  print(f"Stored CSV data from {url}")

# Step 5: Fetch data from Reddit using PRAW
reddit = praw.Reddit(client_id='UwHt2a04eD1Jhn85Pgo6Tw',
                     client_secret='bnQIKg7VglHt-fp7WB9y5NGLDO54og',
                     user_agent='Scraper 1.0 by /u/Kooky-Inspector-4523')

subreddits = ['sex', 'sexeducation', 'sexover30']
for subreddit in subreddits:
    for submission in reddit.subreddit(subreddit).hot(limit=10):  # Fetching top 10 hot posts
        text = submission.title + "\n" + submission.selftext
        url = submission.url
        embedding = model.encode(text)
        # Store document in ChromaDB collection
        collection.add(
            documents=[text],
            metadatas=[{"url": url}],
            ids=[f"reddit_{subreddit}_{submission.id}"]
        )
        print(f"Stored Reddit data from {url}")

# Step 6: Search function to find closest match in vector database
def search_query(query):
    results = collection.query(
        query_texts=[query],
        n_results=1
    )
    if results and results['documents']:
        best_match = results['documents'][0][0]
        return best_match
    else:
        return "Sorry, I don't know this."

# Step 7: Define routes for chatbot interaction and other pages
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/os')
def os_page():
    return render_template('os.html')

@socketio.on('message')
def handle_message(message):
    user_input = message
    response = chatbot_response(user_input)
    emit('response', {'message': response})

def chatbot_response(user_input):
    context = search_query(user_input)
    response = generator(user_input, max_length=100, context=context)[0]['generated_text']
    return response

if __name__ == '__main__':
    socketio.run(app, debug=True)
