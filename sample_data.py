"""
Sample data generator for testing the Persian RAG system.

This module provides sample Persian text data that can be used to test
the RAG system without requiring web crawling.
"""

import os
import json
import pandas as pd
from datetime import datetime

# Sample Persian text data
SAMPLE_TEXTS = [
    {
        "title": "هوش مصنوعی چیست؟",
        "content": "هوش مصنوعی یا AI مجموعه‌ای از تکنولوژی‌ها و روش‌هایی است که به کامپیوترها اجازه می‌دهد به طور خودکار و هوشمند از اطلاعات یاد بگیرند و تصمیمات بگیرند. این تکنولوژی از الگوریتم‌های پیچیده و مدل‌های یادگیری ماشین استفاده می‌کند تا بتواند به طور خودکار از داده‌ها یاد بگیرد و به تدریج بهبود یابد. هوش مصنوعی در زمینه‌های مختلفی از تشخیص صدا و تصویر تا پیش‌بینی بازار و تشخیص بیماری‌ها کاربرد دارد.",
        "url": "https://example.com/ai-intro",
        "date": "2023-01-15",
        "author": "تیم تحریریه فناوری",
        "category": "فناوری"
    },
    {
        "title": "تاریخچه هوش مصنوعی",
        "content": "تاریخ هوش مصنوعی به دهه 1950 بازمی‌گردد. در این زمان، عالم ریاضی جان ون نیوم و دیگران اولین مفاهیم اساسی هوش مصنوعی را مطرح کردند. در دهه 1960 و 1970، الگوریتم‌های اولیه یادگیری ماشین معرفی شدند. از دهه 1980 به بعد، با پیشرفت‌های در علوم کامپیوتر و افزایش قدرت محاسباتی، هوش مصنوعی به سرعت پیشرفت کرد. در دهه 2000 و 2010، با ظهور شبکه‌های عصبی عمیق و داده‌های بزرگ، هوش مصنوعی به یکی از مهم‌ترین تکنولوژی‌های جهان تبدیل شد.",
        "url": "https://example.com/ai-history",
        "date": "2023-02-20",
        "author": "دکتر علی محمدی",
        "category": "تاریخچه"
    },
    {
        "title": "کاربردهای هوش مصنوعی",
        "content": "هوش مصنوعی در زمینه‌های مختلفی کاربرد دارد. در پزشکی، برای تشخیص بیماری‌ها و توصیه درمان استفاده می‌شود. در تجارت، برای پیش‌بینی بازار و توصیه محصولات به کار می‌رود. در صنعت، برای بهینه‌سازی فرآیندها و کنترل کیفیت استفاده می‌شود. در ترجمه و پردازش زبان طبیعی، برای ترجمه متن و تشخیص زبان استفاده می‌شود. در رباتیک، برای ساخت ربات‌های هوشمند و خودکار استفاده می‌شود.",
        "url": "https://example.com/ai-applications",
        "date": "2023-03-10",
        "author": "دکتر مریم حسینی",
        "category": "کاربردها"
    },
    {
        "title": "چالش‌های هوش مصنوعی",
        "content": "یکی از مهم‌ترین چالش‌های هوش مصنوعی مسئله اخلاقی و حریم خصوصی است. با پیشرفت این تکنولوژی، می‌توان اطلاعات شخصی را به راحتی جمع‌آوری و تحلیل کرد. همچنین، اعتماد به هوش مصنوعی و صحت تصمیمات آن نیز یکی از چالش‌های مهم است. در نهایت، نیاز به داده‌های کیفیت بالا و متنوع برای آموزش مدل‌ها نیز یک مسئله مهم است.",
        "url": "https://example.com/ai-challenges",
        "date": "2023-04-05",
        "author": "دکتر رضا احمدی",
        "category": "چالش‌ها"
    },
    {
        "title": "آینده هوش مصنوعی",
        "content": "آینده هوش مصنوعی بسیار پرآیند است. در آینده می‌توان انتظار داشت که هوش مصنوعی به سمت هوش مصنوعی عمومی (AGI) پیش برود که بتواند در زمینه‌های مختلف به طور هوشمند عمل کند. همچنین، انتظار می‌رود که هوش مصنوعی در درمان بیماری‌های ناشناخته و حل مسائل پیچیده جهانی نقش مهمی داشته باشد.",
        "url": "https://example.com/ai-future",
        "date": "2023-05-15",
        "author": "پروفسور سعید رضوی",
        "category": "آینده‌نگر"
    }
]

def create_sample_csv(output_dir: str = "sample_data"):
    """
    Create a sample CSV file with Persian text data.
    
    Args:
        output_dir: Directory to save the CSV file
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame
    df = pd.DataFrame(SAMPLE_TEXTS)
    
    # Save to CSV
    output_path = os.path.join(output_dir, 'sample_data.csv')
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Sample data created at: {output_path}")
    
    # Also save as JSON for reference
    json_path = os.path.join(output_dir, 'sample_data.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(SAMPLE_TEXTS, f, ensure_ascii=False, indent=2)
    print(f"Sample data (JSON) created at: {json_path}")
    
    # Save individual text files for each document
    txt_dir = os.path.join(output_dir, 'texts')
    os.makedirs(txt_dir, exist_ok=True)
    
    for i, doc in enumerate(SAMPLE_TEXTS, 1):
        txt_path = os.path.join(txt_dir, f'doc_{i:03d}.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"عنوان: {doc['title']}\n")
            f.write(f"نویسنده: {doc['author']}\n")
            f.write(f"تاریخ: {doc['date']}\n")
            f.write(f"دسته‌بندی: {doc['category']}\n")
            f.write("\nمتن:\n")
            f.write(doc['content'])
    
    print(f"Individual text files created in: {txt_dir}")

if __name__ == "__main__":
    create_sample_csv()
    print("\nSample data created successfully!")
    print("You can now use this data to test the RAG system.")
    print("Run test_enhanced_rag.py to test the system with this sample data.")
