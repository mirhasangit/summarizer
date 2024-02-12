from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
import os



load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

from langchain.document_loaders import PyPDFLoader

# Load the book
loader = PyPDFLoader("./sodl.pdf")
pages = loader.load()

# Cut out the open and closing parts
pages = pages[17:323]

# Combine the pages, and replace the tabs with spaces
text = ""

for page in pages:
    text += page.page_content
    
text = text.replace('\t', ' ')

num_tokens = llm.get_num_tokens(text)

print (f"This book has {num_tokens} tokens in it")

# Loaders
from langchain.schema import Document

# Splitters
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Model
from langchain.chat_models import ChatOpenAI

# Embedding Support
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Summarizer we'll use for Map Reduce
from langchain.chains.summarize import load_summarize_chain

# Data Science
import numpy as np
from sklearn.cluster import KMeans

text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=10000, chunk_overlap=3000)

docs = text_splitter.create_documents([text])

num_documents = len(docs)

print (f"Now our book is split up into {num_documents} documents")

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

vectors = embeddings.embed_documents([x.page_content for x in docs])

# Assuming 'embeddings' is a list or array of 1536-dimensional embeddings

# Choose the number of clusters, this can be adjusted based on the book's content.
# ~10 is about the best.
# Usually if you have 10 passages from a book you can tell what it's about
num_clusters = 11

# Perform K-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)

kmeans.labels_

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Taking out the warnings
import warnings
from warnings import simplefilter

# Filter out FutureWarnings
simplefilter(action='ignore', category=FutureWarning)

vectors_np = np.array(vectors) 

# Perform t-SNE and reduce to 2 dimensions
tsne = TSNE(n_components=2, random_state=42)
reduced_data_tsne = tsne.fit_transform(vectors_np)

# Plot the reduced data
plt.scatter(reduced_data_tsne[:, 0], reduced_data_tsne[:, 1], c=kmeans.labels_)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Book Embeddings Clustered')
plt.show()

# Find the closest embeddings to the centroids

# Create an empty list that will hold your closest points
closest_indices = []

# Loop through the number of clusters you have
for i in range(num_clusters):
    
    # Get the list of distances from that particular cluster center
    distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
    
    # Find the list position of the closest one (using argmin to find the smallest distance)
    closest_index = np.argmin(distances)
    
    # Append that position to your closest indices list
    closest_indices.append(closest_index)

    

selected_indices = sorted(closest_indices)
selected_indices

try:
    vectors = embeddings.embed_documents([x.page_content for x in docs])
except openai.RateLimitError as e:
    print("Rate limit exceeded. Please try again later.")
    # Handle the error appropriately, maybe by exiting or delaying further requests.


llm3 = ChatOpenAI(temperature=0,
                 openai_api_key=openai_api_key,
                 max_tokens=1000,
                 model='gpt-3.5-turbo'
                )





map_prompt = """
You will be given a single passage of a book. This section will be enclosed in triple backticks (```)
Your goal is to give a summary of this section so that a reader will have a full understanding of what happened.
Use direct statements and active voice. Avoid personal commentary and use the present tense. 
Your response should be the main idea as a header and at least three paragraphs that fully encompass what was said in the passage.

```{text}```
FULL SUMMARY:
"""
map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

map_chain = load_summarize_chain(llm=llm3,
                             chain_type="stuff",
                             prompt=map_prompt_template)

selected_docs = [docs[doc] for doc in selected_indices]

# Make an empty list to hold your summaries
summary_list = []

filename = "book_summaries.txt"

with open(filename, 'w', encoding='utf-8') as file:

    # Loop through a range of the lenght of your selected docs
    for i, doc in enumerate(selected_docs):
    
        # Go get a summary of the chunk
        chunk_summary = map_chain.run([doc])
    
        # Append that summary to your list
        summary_list.append(chunk_summary)

        file.write(f"Summary #{i} (chunk #{selected_indices[i]}):\n{chunk_summary}\n\n")

        # print (f"Summary #{i} (chunk #{selected_indices[i]}) - Preview: {chunk_summary[:250]} \n")

    print(f"Summaries have been written to {filename}.")