# smart_hr_document_assistant

This repository contains a notebook-based proof-of-concept for building a simple HR document assistant using LangChain, Google Gemini embeddings, and a Chroma vector store.

## What it does

- Loads an HR policy PDF (`Sickness-And-Absence-Policy.pdf`)
- Splits the document into text chunks
- Creates embeddings using Gemini (`models/gemini-embedding-001`)
- Stores embeddings in a Chroma vector database
- Uses a RAG pipeline with Gemini (`gemini-2.5-flash`) to answer questions from the document

## Notebook workflow

1. Install required libraries:
   - `langchain`
   - `langchain-community`
   - `langchain-core`
   - `chromadb`
   - `pypdf`
   - `langchain-google-genai`
2. Load the PDF with `PyPDFLoader`.
3. Split the document into chunks using `RecursiveCharacterTextSplitter`.
4. Configure Gemini embeddings via `GoogleGenerativeAIEmbeddings`.
5. Build a Chroma vector store from the document chunks.
6. Create a retriever and prompt template for RAG.
7. Query the chain and print the answer.

## Usage

- Open `smart_hr_document_assistant.ipynb`.
- Ensure the PDF is available at `/content/smart_hr_document_assistant/Sickness-And-Absence-Policy.pdf` or update the path.
- Set your `GEMINI_API_KEY` in the Colab environment.
- Run the notebook cells in order.

## Example query

The notebook includes an example question:

> "Should I submit any documents if I take 3 sick leaves in a row?"
