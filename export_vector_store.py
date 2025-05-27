import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from tqdm import tqdm
import argparse

def load_and_process_data(data_dir: str):
    """Load and process data from CSV files."""
    print(f"Loading data from {data_dir}")
    all_texts = []
    
    # Get all CSV files in the directory
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    for csv_file in tqdm(csv_files, desc="Processing files"):
        try:
            df = pd.read_csv(os.path.join(data_dir, csv_file))
            # Assuming the text is in a column named 'text' or 'content'
            text_col = 'text' if 'text' in df.columns else 'content' if 'content' in df.columns else df.columns[1]
            texts = df[text_col].dropna().tolist()
            all_texts.extend(texts)
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
    
    # Split texts into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    print("Splitting texts into chunks...")
    chunks = []
    for text in tqdm(all_texts, desc="Splitting texts"):
        chunks.extend(text_splitter.split_text(text))
    
    print(f"Total chunks created: {len(chunks)}")
    return chunks

def create_and_save_vector_store(chunks, output_dir: str = "vector_store"):
    """Create and save FAISS vector store."""
    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/distiluse-base-multilingual-cased-v2"
    )
    
    print("Creating FAISS vector store...")
    vector_store = FAISS.from_texts(chunks, embeddings)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving vector store to {output_dir}...")
    vector_store.save_local(output_dir)
    print(f"Vector store saved successfully to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Export vector store from processed data')
    parser.add_argument('--data-dir', type=str, default='processed_data',
                       help='Directory containing processed CSV files')
    parser.add_argument('--output-dir', type=str, default='vector_store',
                       help='Directory to save the vector store')
    
    args = parser.parse_args()
    
    print("Starting vector store export...")
    chunks = load_and_process_data(args.data_dir)
    create_and_save_vector_store(chunks, args.output_dir)

if __name__ == "__main__":
    main()
