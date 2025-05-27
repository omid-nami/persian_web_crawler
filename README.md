# Persian Web Crawler and RAG System

This project provides a complete pipeline for crawling Persian websites, processing the extracted content, and building a Retrieval-Augmented Generation (RAG) system using Ollama LLM with multi-GPU support.

## Features

### Web Crawling
- Crawls up to 100 pages (configurable) starting from a base URL
- Extracts and cleans Persian text content
- Normalizes Persian text using Hazm
- Removes stopwords and tokenizes text
- Saves data in both JSON (raw) and Parquet (processed) formats
- Includes error handling and rate limiting

### Enhanced RAG System
- Advanced Persian text processing and cleaning
- Document chunking with configurable overlap
- FAISS vector store for efficient similarity search
- Integration with Ollama LLM for question answering
- Multi-GPU support for improved performance
- Support for multiple embedding models including multilingual models
- Interactive query interface with source citation
- Comprehensive logging and error handling

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- NVIDIA GPU with CUDA support (recommended)
- Ollama installed and running (for LLM inference)
- Git (for version control)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/persian_web_crawler.git
   cd persian_web_crawler
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Install Ollama (if not already installed):
   ```bash
   # On Linux (using Snap)
   sudo snap install ollama
   
   # Or using the install script
   # curl -fsSL https://ollama.com/install.sh | sh
   
   # Start Ollama service
   sudo systemctl start ollama
   
   # Install required model (e.g., qwen2.5)
   ollama pull qwen2.5
   
   # For multi-GPU support, set these environment variables before starting Ollama:
   export OLLAMA_GPUS=-1  # Use all available GPUs
   export OLLAMA_GPU_LAYERS=999  # Offload as many layers as possible to GPU
   ollama serve
   
   # Start the Ollama server
   ollama serve
   ```

5. Pull the desired model (e.g., qwen2.5):
   ```bash
   ollama pull qwen2.5
   ```

## GPU Configuration

### Multi-GPU Setup

To utilize multiple GPUs with Ollama:

1. Stop the Ollama service if it's running:
   ```bash
   sudo systemctl stop ollama
   ```

2. Configure GPU settings for Snap installation:
   ```bash
   sudo snap set ollama \
       OLLAMA_GPU_LAYERS=999 \
       OLLAMA_GPUS=-1
   ```

3. Restart the service:
   ```bash
   sudo systemctl restart ollama
   ```

4. Verify GPU usage:
   ```bash
   nvidia-smi
   ```

### Verifying GPU Support

Run the following to check if your GPUs are being utilized:

```bash
# In one terminal, monitor GPU usage
watch -n 1 nvidia-smi

# In another terminal, run a test query
ollama run qwen2.5 "Test GPU usage"
```

## Usage

### 1. Web Crawling

To crawl a website, run:

```bash
python persian_crawler.py https://example.com --max-pages 50
```

#### Crawler Arguments
- `url`: The starting URL to crawl (required)
- `--max-pages`: Maximum number of pages to crawl (default: 100)

### 2. RAG System

#### Interactive Mode

To start an interactive question-answering session:

```bash
python rag_ollama.py --data-dir processed_data --model qwen2.5
```

This will:
1. Load and process documents from the specified directory
2. Create a FAISS vector store
3. Start an interactive question-answering session

Available commands in the interactive session:
- `exit` or `quit`: Exit the program
- `clear`: Clear the chat history
- `help`: Show available commands

#### Exporting the Vector Store

To export the vector store for later use:

```bash
python export_vector_store.py --data-dir processed_data --output-dir my_vector_store
```

This creates a portable vector store that can be used with the RAG system.

#### Using the Exported Vector Store

```bash
python rag_with_ollama.py --vector-store my_vector_store --model qwen2.5
```

This loads the pre-built vector store instead of processing documents again. The system will retrieve relevant information and generate answers using the Ollama LLM.

Example questions:
- "هوش مصنوعی چیست؟"
- "کاربردهای یادگیری ماشین در پزشکی چیست؟"
- "تاریخچه هوش مصنوعی را توضیح دهید"

### 3. Testing

Run tests for different components:

```bash
# Test document processing
python test_rag.py --test processor

# Test vector store
python test_rag.py --test vector

# Test complete RAG system
python test_rag.py --test rag

# Run all tests
python test_rag.py --test all
```

## Output

The crawler creates two directories:

- `raw_data/`: Contains raw JSON files with the extracted content
- `processed_data/`: Contains processed data in Parquet format, ready for LLM training
- `faiss_index/`: Contains the FAISS vector store (created by the RAG system)
- `*.log`: Log files for debugging

## System Architecture

### 1. Web Crawling Pipeline
1. **Crawling**: The crawler visits each page and extracts links
2. **Content Extraction**: For each page, it extracts:
   - Main text content
   - Title
   - Publication date (if available)
   - URL
3. **Text Processing**:
   - Normalizes Persian text
   - Removes stopwords
   - Tokenizes text
4. **Data Storage**: Saves both raw and processed data

### 2. RAG System Pipeline
1. **Document Processing**:
   - Loads and cleans documents
   - Splits documents into chunks with overlap
   - Removes duplicates and low-quality text

2. **Vector Store**:
   - Generates embeddings using a pre-trained model
   - Builds a FAISS index for efficient similarity search
   - Supports cosine similarity for semantic search

3. **Question Answering**:
   - Processes user questions
   - Retrieves relevant document chunks
   - Generates answers using Ollama LLM
   - Provides sources for the answers

## Configuration

You can customize the RAG system by modifying these parameters in `rag_enhanced.py`:

- `chunk_size`: Size of document chunks (default: 800)
- `chunk_overlap`: Overlap between chunks (default: 150)
- `k`: Number of document chunks to retrieve (default: 4)
- `embedding_model`: Pre-trained embedding model (default: "sentence-transformers/parsbert-base-parsmpnet-uncased")

## Performance Considerations

- The system is optimized for Persian text but can be adapted for other languages
- Processing large documents may require significant memory
- For best results, use a GPU for embedding generation
- The quality of answers depends on the quality of the crawled content and the LLM model

## Note on Ethics and Legality

- Always check the website's `robots.txt` before crawling
- Respect `robots.txt` directives
- Add delays between requests to avoid overloading the server
- Only crawl websites where you have permission to do so
- Be mindful of copyright and intellectual property rights
- The system is for research and educational purposes only

## Troubleshooting

### Common Issues

1. **Ollama not running**
   - Make sure the Ollama server is running: `ollama serve`
   - Check if the model is downloaded: `ollama list`

2. **Memory issues**
   - Reduce `chunk_size` or `k` parameters
   - Process documents in smaller batches
   - Use a machine with more RAM

3. **Poor quality answers**
   - Ensure the crawled content is relevant to your questions
   - Try different embedding models
   - Adjust the `k` parameter to retrieve more or fewer documents

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
