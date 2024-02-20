
# Comprehensive Book Analysis and Summarization Tool

This script is designed to perform an in-depth analysis and summarization of a text document, specifically targeting PDF files. It utilizes the LangChain and OpenAI libraries to process, cluster, and summarize text, leveraging machine learning and natural language processing technologies.

## Features

- **PDF Document Loading:** Load text content from a PDF file, excluding specified beginning and ending parts.
- **Token Counting:** Count the number of tokens in the document using OpenAI's API.
- **Document Splitting:** Split the document into manageable chunks for processing.
- **Embedding and Clustering:** Generate document embeddings and cluster them using KMeans to identify related sections.
- **t-SNE Visualization:** Visualize the clustering of document sections in a 2D space.
- **Summarization:** Summarize each cluster's centroid document to capture the essence of the text.

## Requirements

To run this script, you will need:

- Python 3.8 or later
- `dotenv`: To load environment variables
- `os`: For interacting with the operating system
- `langchain_openai` and `langchain_community`: For processing and analysis tools
- `sklearn`: For clustering and dimensionality reduction
- `matplotlib`: For plotting and visualization
- `numpy`: For numerical operations

Make sure you have an OpenAI API key set in your environment variables as `OPENAI_API_KEY`.

## Installation

Clone this repository and navigate into the project directory. Install the required dependencies with:

```bash
pip install python-dotenv langchain-openai langchain-community scikit-learn matplotlib numpy
```

## Usage

1. Ensure your PDF document is in the project directory or adjust the `file_path` variable accordingly.
2. Set your OpenAI API key in a `.env` file in the root directory of this project:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
3. Run the script:
   ```bash
   python script_name.py
   ```
   Replace `script_name.py` with the name of this script file.

4. The output will include token counts, document splits, a visualization (which needs to be manually displayed with `plt.show()`), and summaries for each document cluster.

## Customization

You can customize the script by adjusting:

- The PDF file path and the pages to be analyzed.
- The number of clusters in the KMeans algorithm.
- The dimensionality reduction technique settings for t-SNE visualization.
- The summarization prompt and parameters for the ChatOpenAI model.

## Contributing

Contributions to improve the script or extend its functionality are welcome. Please open an issue or pull request on the project's repository.


