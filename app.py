from dotenv import load_dotenv
import os
from langchain_openai import OpenAI, OpenAIEmbeddings  
from langchain_community.document_loaders import PyPDFLoader  
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS  
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
import warnings
from warnings import simplefilter

# Environment Setup
load_dotenv()
file_path = "./SODL.pdf"

# API Key Configuration
openai_api_key = os.getenv('OPENAI_API_KEY')
llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

# Document Loading and Preprocessing
book_name = os.path.splitext(os.path.basename(file_path))[0]
loader = PyPDFLoader(file_path)
pages = loader.load()[17:323]  # Load the book and cut out the open and closing parts
text = "".join(page.page_content for page in pages).replace('\t', ' ')  # Combine pages and replace tabs with spaces
num_tokens = llm.get_num_tokens(text)
print (f"This book has {num_tokens} tokens in it")

# Document Splitting
text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=10000, chunk_overlap=3000)
docs = text_splitter.create_documents([text])
print(f"Now our book is split up into {len(docs)} documents")

# Embedding and Clustering
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectors = embeddings.embed_documents([x.page_content for x in docs])
num_clusters = 11
kmeans = KMeans(n_clusters=num_clusters, n_init= 15, random_state=42).fit(vectors)

# t-SNE Visualization
#simplefilter(action='ignore', category=FutureWarning)  # Filter out FutureWarnings
vectors_np = np.array(vectors)
tsne = TSNE(n_components=2, random_state=42)
reduced_data_tsne = tsne.fit_transform(vectors_np)
plt.scatter(reduced_data_tsne[:, 0], reduced_data_tsne[:, 1], c=kmeans.labels_)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Book Embeddings Clustered')
# 2D visualization of the clusters:
# plt.show()

# Summarization Setup
llm3 = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, max_tokens=1000, model='gpt-3.5-turbo')
map_prompt = """
You will be given a single passage of a book. This section will be enclosed in triple backticks (```)
Imagine you are the author of the book and you are explaining its main ideas directly to a reader in a personal conversation. 
Your goal is to give a summary of this section so that a reader will have a full understanding of what happened.
Your response should be at least three to seven paragraphs and fully encompass what was said in the passage. 
The summary should be standalone, meaning it should not refer to the text or the author in the third person. 
Craft it as if you are speaking in your own voice, using the first-person perspective where appropriate, 
and ensure that it is complete in itself without needing any additional context.
Never use statements such as "the passage" or "the author says" in your summary.
Lastly, suggest a title that encapsulates the essence of the summary.


```{text}```

Title:

Summary:
"""
map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
map_chain = load_summarize_chain(llm=llm3, chain_type="stuff", prompt=map_prompt_template)
selected_indices = [np.argmin(np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)) for i in range(num_clusters)] # Select the closest document to the centroid of each cluster
selected_docs = [docs[doc] for doc in sorted(selected_indices)]

# Summarization Execution
folder_name = "Complete_Book_Summary"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
for i, doc in enumerate(selected_docs):
    chunk_summary = map_chain.run([doc])
    filename = os.path.join(folder_name,  f"{book_name}_Page_{i + 1}.txt")
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(f"Summary (chunk #{sorted(selected_indices)[i]}):\n{chunk_summary}\n\n")
    print(f"Summary for chunk #{sorted(selected_indices)[i]} has been written to {filename}.")
