import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import argparse

def load_vector_store(vector_store_path, model_name="sentence-transformers/distiluse-base-multilingual-cased-v2"):
    """Load the FAISS vector store."""
    print("Loading embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    print("Loading vector store...")
    vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    return vector_store

def create_qa_chain(vector_store, model_name="qwen2.5"):
    """Create a QA chain with Ollama."""
    print(f"Initializing Ollama with model: {model_name}")
    
    # Define the prompt template in Persian
    prompt_template = """به سوال زیر با توجه به متن داده شده پاسخ دهید. اگر پاسخ را نمی‌دانید، بگویید "نمی‌دانم" و سعی نکنید جعلی بسازید.

    متن:
    {context}

    سوال: {question}
    پاسخ:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
    
    # Initialize Ollama
    llm = Ollama(
        model=model_name,
        temperature=0.3,  # Lower temperature for more focused answers
        num_ctx=4096,    # Context window size
    )
    
    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),  # Retrieve top 3 most similar documents
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,
    )
    
    return qa_chain

def interactive_qa(qa_chain):
    """Run an interactive QA session."""
    print("\n" + "="*60)
    print("Persian QA System with Ollama")
    print("Type 'exit' to quit")
    print("="*60 + "\n")
    
    while True:
        question = input("\nسوال خود را بپرسید: ").strip()
        
        if question.lower() in ['exit', 'خروج']:
            print("\nخدانگهدار!")
            break
            
        if not question:
            continue
            
        try:
            print("\nدر حال پردازش...")
            result = qa_chain({"query": question})
            
            print("\n" + "="*60)
            print("\nپاسخ:")
            print(result["result"])
            
            # Show sources
            print("\nمنابع:")
            for i, doc in enumerate(result["source_documents"], 1):
                print(f"\n--- منبع {i} ---")
                print(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                
        except Exception as e:
            print(f"\nخطا در پردازش سوال: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Persian QA System with Ollama and RAG')
    parser.add_argument('--vector-store', type=str, default='my_vector_store',
                       help='Path to the vector store directory')
    parser.add_argument('--model', type=str, default='qwen2.5',
                       help='Ollama model name (default: qwen2.5)')
    
    args = parser.parse_args()
    
    try:
        # Load the vector store
        vector_store = load_vector_store(args.vector_store)
        
        # Create QA chain
        qa_chain = create_qa_chain(vector_store, args.model)
        
        # Start interactive QA
        interactive_qa(qa_chain)
        
    except Exception as e:
        print(f"\nخطا: {str(e)}")
        print("\nمطمئن شوید که:")
        print("1. سرور Ollama در حال اجرا است (با دستور 'ollama serve')")
        print(f"2. مدل {args.model} نصب شده است (با دستور 'ollama pull {args.model}')")
        print("3. مسیر ذخیره‌سازی بردارها صحیح است")

if __name__ == "__main__":
    main()
