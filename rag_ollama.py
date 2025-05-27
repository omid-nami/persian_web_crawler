import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from typing import List, Dict, Any, Optional
import pandas as pd
from tqdm import tqdm
import json
import re
import faiss
import numpy as np
import string
import unicodedata
from pathlib import Path
from datetime import datetime

# Persian stopwords
PERSIAN_STOPWORDS = {
    'و', 'در', 'به', 'از', 'که', 'با', 'را', 'این', 'برای', 'آن',
    'یک', 'خود', 'تا', 'کرد', 'شده', 'است', 'نیز', 'شود', 'های', 'اما',
    'کردن', 'کردم', 'کرده', 'کردی', 'کردید', 'کردند', 'خواهد', 'خواهیم',
    'خواهید', 'خواهند', 'شده', 'شده‌ام', 'شده‌ای', 'شده‌اید', 'شده‌اند',
    'شده‌بودم', 'شده‌بودی', 'شده‌بود', 'شده‌بودیم', 'شده‌بودید', 'شده‌بودند',
    'دارد', 'دارم', 'داری', 'داریم', 'دارید', 'دارند', 'داشتم', 'داشتی',
    'داشت', 'داشتیم', 'داشتید', 'داشتند', 'خواهم', 'خواهی', 'خواهد', 'خواهیم',
    'خواهید', 'خواهند', 'باشم', 'باشی', 'باشد', 'باشیم', 'باشید', 'باشند',
    'بودم', 'بودی', 'بود', 'بودیم', 'بودید', 'بودند', 'شو', 'شود', 'شویم',
    'شوید', 'شوند', 'شدم', 'شدی', 'شد', 'شدیم', 'شدید', 'شدند', 'م', 'ت', 'ش',
    'ای', 'ها', 'های', 'هایی', 'هایم', 'هایت', 'هایش', 'هایمان', 'هایتان', 'هایشان'
}

class PersianRAG:
    def __init__(self, model_name="qwen2.5"):
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/distiluse-base-multilingual-cased-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize Persian text."""
        if not isinstance(text, str):
            return ""
            
        # Convert to string if not already
        text = str(text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Normalize Persian characters
        text = unicodedata.normalize('NFC', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove emails
        text = re.sub(r'\S*@\S*\s?', '', text)
        
        # Remove numbers and English characters (optional, depending on your needs)
        # text = re.sub(r'[a-zA-Z0-9]', '', text)
        
        # Remove punctuation
        translator = str.maketrans('', '', string.punctuation + '،؛؟»«')
        text = text.translate(translator)
        
        # Remove stopwords
        words = text.split()
        words = [word for word in words if word not in PERSIAN_STOPWORDS]
        text = ' '.join(words)
        
        # Remove extra spaces again
        text = ' '.join(text.split())
        
        return text

    def load_data(self, data_dir: str) -> List[str]:
        """Load and clean processed data from CSV files in the specified directory."""
        all_data = []
        try:
            for file in os.listdir(data_dir):
                if file.endswith('.csv'):
                    file_path = os.path.join(data_dir, file)
                    print(f"Loading data from {file}")
                    try:
                        df = pd.read_csv(file_path, on_bad_lines='skip')
                        if 'text' in df.columns:
                            # Clean and normalize the text
                            df['cleaned_text'] = df['text'].apply(self.clean_text)
                            # Remove empty or very short texts after cleaning
                            df = df[df['cleaned_text'].str.len() > 20]
                            all_data.extend(df['cleaned_text'].dropna().astype(str).tolist())
                        else:
                            print(f"Warning: 'text' column not found in {file}")
                    except Exception as e:
                        print(f"Error processing {file}: {str(e)}")
                        continue
                        
            print(f"Loaded {len(all_data)} text chunks from {data_dir}")
            return all_data
        except Exception as e:
            print(f"Error loading data: {e}")
            return []

    def create_vector_store(self, data_dir: str):
        """Create a FAISS vector store from the processed data with enhanced text processing."""
        try:
            # Load and clean data
            texts = self.load_data(data_dir)
            if not texts:
                raise ValueError("No text data found to create vector store")

            # Additional filtering for quality
            texts = [text for text in texts if len(text.split()) > 5]  # Remove very short texts
            
            print(f"Processing {len(texts)} text chunks after cleaning")

            # Split texts into chunks with better handling of Persian text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,  # Slightly smaller chunks for better context
                chunk_overlap=150,
                length_function=len,
                separators=['\n\n', '\n', ' ', '']  # Better handling of Persian text
            )
            
            # Create documents with metadata
            chunks = text_splitter.create_documents(texts)
            print(f"Split into {len(chunks)} chunks of text")

            # Initialize embeddings with better configuration for Persian
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/distiluse-base-multilingual-cased-v2"
            )
            embeddings = self.embeddings
            
            print("Creating FAISS vector store (this may take a while)...")
            
            # Create FAISS vector store with optimized parameters
            vector_store = FAISS.from_documents(
                documents=chunks,
                embedding=embeddings,
                distance_strategy="COSINE"  # Better for semantic similarity
            )
            
            print("Vector store created successfully")
            return vector_store
            
        except Exception as e:
            print(f"Error creating vector store: {e}")
            raise

    def create_qa_chain(self, vector_store):
        """Create RetrievalQA chain with Ollama"""
        try:
            print(f"Initializing Ollama with model: {self.model_name}")
            # Create Ollama model with explicit parameters
            llm = Ollama(
                model=self.model_name,
                temperature=0.1,  # Lower temperature for more focused answers
                num_ctx=2048,     # Context window size
                num_predict=512   # Max tokens to generate
            )
            
            # Create prompt template with Persian instructions
            template = """
            شما یک دستیار هوشمند هستید که به سوالات بر اساس متن‌های داده شده پاسخ می‌دهید.
            
            متن‌های مرتبط:
            {context}
            
            سوال: {question}
            
            لطفاً پاسخی دقیق و کامل به زبان فارسی ارائه دهید. اگر پاسخی برای سوال نمی‌دانید، بگویید "پاسخی برای این سوال پیدا نکردم".
            
            پاسخ:"""
            
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
            
            # Create RetrievalQA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}  # Number of documents to retrieve
                ),
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True,
                verbose=True
            )
            return qa_chain
        except Exception as e:
            print(f"Error creating QA chain: {str(e)}")
            raise

    def run(self, data_dir: str):
        """Run the RAG pipeline"""
        try:
            print("Starting RAG pipeline...")
            print("1. Creating vector store...")
            vector_store = self.create_vector_store(data_dir)
            
            print("\n2. Creating QA chain...")
            qa_chain = self.create_qa_chain(vector_store)
            
            print("\nRAG pipeline initialized successfully!")
            return qa_chain
            
        except Exception as e:
            print(f"Error in RAG pipeline: {str(e)}")
            raise

def display_help():
    print("\nCommands:")
    print("  /help - Show this help message")
    print("  /quit - Exit the program")
    print("  /clear - Clear the screen")
    print("  /sources - Show sources for the last answer")
    print("  Type your question to get an answer based on the processed documents\n")

if __name__ == "__main__":
    try:
        # Initialize RAG system
        print("Initializing Persian RAG system...")
        rag = PersianRAG()
        
        # Create QA chain
        print("\nLoading and processing documents...")
        qa_chain = rag.run("processed_data")
        
        # Interactive loop
        print("\n" + "="*60)
        print("Persian RAG System - Ready!")
        print("Type /help for available commands")
        print("="*60 + "\n")
        
        last_sources = None
        
        while True:
            try:
                question = input("\nسوال خود را بپرسید (یا /help برای راهنما): ").strip()
                
                # Handle commands
                if not question:
                    continue
                    
                if question.lower() == '/quit':
                    print("\nخدانگهدار!")
                    break
                    
                if question.lower() == '/help':
                    display_help()
                    continue
                    
                if question.lower() == '/clear':
                    os.system('clear' if os.name == 'posix' else 'cls')
                    continue
                    
                if question.lower() == '/sources':
                    if last_sources:
                        print("\nمنابع پاسخ آخر (متن‌های مرجع):")
                        for i, doc in enumerate(last_sources, 1):
                            print(f"\n--- منبع {i} ---")
                            print(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                    else:
                        print("\nهیچ پاسخی قبلاً داده نشده است.")
                    continue
                
                # Process the question
                print("\nدر حال پردازش سوال...")
                result = qa_chain({"query": question})
                
                # Display the answer
                print("\n" + "="*60)
                print("\nپاسخ:")
                print(result["result"])
                print("\n" + "="*60)
                
                # Store sources for potential later reference
                last_sources = result.get("source_documents", [])
                
                if last_sources:
                    print(f"\n(برای مشاهده منابع از دستور /sources استفاده کنید)")
                
            except KeyboardInterrupt:
                print("\nبرای خروج از برنامه از دستور /quit استفاده کنید.")
                continue
                
            except Exception as e:
                print(f"\nخطا در پردازش سوال: {str(e)}")
                print("لطفاً دوباره سوال خود را مطرح کنید یا برای راهنما /help را تایپ کنید.")
                continue
                
    except KeyboardInterrupt:
        print("\nبرنامه به درستی خاتمه یافت.")
    except Exception as e:
        print(f"\nخطای غیرمنتظره: {str(e)}")
        print("لطفاً مطمئن شوید که سرور Ollama در حال اجرا است و مدل qwen2.5 نصب شده است.")
        print("دستورات راه‌اندازی:")
        print("1. سرور Ollama را اجرا کنید: ollama serve")
        print("2. مدل را نصب کنید: ollama pull qwen2.5")
        print("3. برنامه را دوباره اجرا کنید")
