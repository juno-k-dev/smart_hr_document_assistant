# smart_hr_document_assistant

This repository contains a notebook-based proof-of-concept for building a simple HR document assistant using LangChain, Google Gemini embeddings, and a Chroma vector store.

## What it does

- Loads multiple HR policy documents from PDF and DOCX files
- Normalizes metadata for each document/page, including source and file type
- Splits all documents into text chunks using `RecursiveCharacterTextSplitter`
- Creates embeddings using Gemini (`models/gemini-embedding-001`)
- Stores embeddings in a Chroma vector database
- Uses a RAG pipeline with Gemini (`gemini-2.5-flash`) to answer questions from the documents
- Prints the generated answer plus source citations for the retrieved chunks

## Notebook workflow

1. Install required libraries:
   - `langchain`
   - `langchain-community`
   - `langchain-core`
   - `chromadb`
   - `pypdf`
   - `langchain-google-genai`
   - `unstructured`
   - `python-docx`
2. Load PDF files with `PyPDFLoader` and DOCX files with `UnstructuredWordDocumentLoader`.
3. Add metadata fields for `source`, `type`, and normalized `page` values.
4. Split all loaded documents into chunks using `RecursiveCharacterTextSplitter` with `chunk_size=1000` and `chunk_overlap=200`.
5. Configure Gemini embeddings with `GoogleGenerativeAIEmbeddings`.
6. Build a Chroma vector store from the resulting document chunks.
7. Create a retriever and a RAG chain using `ChatGoogleGenerativeAI` and `ChatPromptTemplate`.
8. Query the chain and print both the answer and the source documents.

## Supported document files

- `/content/smart_hr_document_assistant/Sickness-And-Absence-Policy.pdf`
- `/content/smart_hr_document_assistant/Fixed-Term-Employment-Contract.pdf`
- `/content/smart_hr_document_assistant/Harassment-and-bullying-policy-3.docx`
- `/content/smart_hr_document_assistant/Employee-Expenses-Policy.docx`

## Usage

- Open `smart_hr_document_assistant.ipynb`.
- Ensure the document files exist in `/content/smart_hr_document_assistant/` or update the file paths in the notebook.
- Set your `GEMINI_API_KEY` in the Colab environment.
- Run the notebook cells in order.

## Example query

The notebook includes an example query:

> "What all expenses in UK can be reimbursed?"
