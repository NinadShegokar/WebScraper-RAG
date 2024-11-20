import torch
from transformers import pipeline, AutoTokenizer, AutoModel
import requests
from bs4 import BeautifulSoup
import faiss
import nltk
import numpy as np
from langchain_ollama import OllamaLLM
from nltk.tokenize import sent_tokenize
from rich.console import Console
from rich.table import Table

# Create a Console instance
console = Console()

llm = OllamaLLM(model="mistral:latest")

def scrape_page(url):
    r = requests.get(url)
    if r.status_code == 200:
        soup = BeautifulSoup(r.content, 'html.parser')
        for script_or_style in soup(['script', 'style', 'footer', 'header', 'aside', 'nav', 'form', 'button', 'input']):
            script_or_style.decompose()
        content_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'span', 'div']
        all_text = [element.get_text(strip=True) for tag in content_tags for element in soup.find_all(tag) if element.get_text(strip=True)]
        return '\n'.join(all_text)
    return None

url = "https://timesofindia.indiatimes.com/sports/cricket/india-in-australia/morne-morkel-provides-a-big-update-on-injured-shubman-gill/articleshow/115474162.cms"
content = scrape_page(url)

console.print("[bold cyan]Scraped Content:[/bold cyan]")
console.print(content)

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")

def chunk_content(content, chunk_size=512):
    content_chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
    return content_chunks

content_chunks = chunk_content(content)

chunk_embeddings = []
for chunk in content_chunks:
    inputs = tokenizer(chunk, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        embedding = model(**inputs).last_hidden_state[:, 0, :].numpy()  # CLS token pooling
    chunk_embeddings.append(embedding)

embeddings_np = np.vstack(chunk_embeddings)

dimension = embeddings_np.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_np)

def generate_rag_response(query, model, tokenizer, index, content_chunks):
    query_inputs = tokenizer(query, return_tensors='pt')
    with torch.no_grad():
        query_embedding = model(**query_inputs).last_hidden_state[:, 0, :].detach().numpy()
    k = 5  
    distances, indices = index.search(query_embedding, k)
    relevant_contexts = [content_chunks[i] for i in indices[0]]
    combined_context = " ".join(relevant_contexts)

    input_text = (
        f"The following context may contain relevant information to answer the question:\n{combined_context}\n\n"
        f"Based on this information, answer the following question directly and concisely, using factual details:\n"
        f"Do not exceed 3 sentences while answering\n"
        f"Question: {query}\nAnswer:"
    )
    response = llm.invoke(input_text)
    return response.strip()

query = "What did Morkel say during the media interaction?"
response = generate_rag_response(query, model, tokenizer, index, content_chunks)

table = Table(title="Generated Response", title_style="bold magenta")
table.add_column("Query", style="cyan", justify="left")
table.add_column("Response", style="green", justify="left")
table.add_row(query, response)

console.print(table)
