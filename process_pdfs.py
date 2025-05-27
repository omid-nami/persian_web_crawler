#!/usr/bin/env python3
"""
Script to process PDF files in a directory using the PersianWebCrawler.
Supports both Persian and English PDFs with automatic language detection.
"""
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import pandas as pd
from tqdm import tqdm

from persian_crawler import PersianWebCrawler

def save_results(data: List[Dict], output_dir: Path, format: str = 'all'):
    """Save the processed data in the specified format(s)."""
    if not data:
        print("No data to save!")
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f"processed_pdfs_{timestamp}"
    
    try:
        if format in ['all', 'json']:
            # Save as JSON
            json_file = output_dir / f"{base_filename}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"\nSaved JSON output to: {json_file}")
        
        if format in ['all', 'csv']:
            # Save as CSV
            csv_file = output_dir / f"{base_filename}.csv"
            df = pd.json_normalize(data)
            
            # Flatten nested structures
            for col in df.columns:
                if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                    df[col] = df[col].astype(str)
            
            df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            print(f"Saved CSV output to: {csv_file}")
            
        if format in ['all', 'parquet']:
            # Try to save as Parquet
            try:
                parquet_file = output_dir / f"{base_filename}.parquet"
                df = pd.json_normalize(data)
                df.to_parquet(parquet_file, index=False)
                print(f"Saved Parquet output to: {parquet_file}")
            except Exception as e:
                print(f"Could not save as Parquet: {str(e)}")
                
    except Exception as e:
        print(f"Error saving results: {str(e)}")

def main():
    try:
        parser = argparse.ArgumentParser(description='Process PDF files in a directory')
        parser.add_argument(
            'directory',
            type=str,
            help='Directory containing PDF files to process'
        )
        parser.add_argument(
            '--output-dir',
            type=str,
            default='processed_pdfs',
            help='Directory to save processed files (default: processed_pdfs)'
        )
        parser.add_argument(
            '--format',
            type=str,
            default='all',
            choices=['json', 'csv', 'parquet', 'all'],
            help='Output format (default: all)'
        )
        parser.add_argument(
            '--max-files',
            type=int,
            default=0,
            help='Maximum number of files to process (0 for no limit)'
        )
        parser.add_argument(
            '--use-ocr',
            action='store_true',
            help='Use OCR for scanned PDFs'
        )
        
        args = parser.parse_args()
        
        # Create output directory if it doesn't exist
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the crawler with a dummy URL
        print("Initializing PDF processor...")
        crawler = PersianWebCrawler('http://example.com')
        
        # Set the output directory
        crawler.processed_data_dir = output_dir
        
        # Find all PDF files
        pdf_files = list(Path(args.directory).rglob('*.pdf'))
        
        if not pdf_files:
            print(f"No PDF files found in: {args.directory}")
            return
        
        # Limit number of files if specified
        if args.max_files > 0:
            pdf_files = pdf_files[:args.max_files]
        
        print(f"\nFound {len(pdf_files)} PDF files to process")
        if args.use_ocr:
            print("OCR is ENABLED for scanned documents")
        else:
            print("OCR is DISABLED, only directly extractable text will be processed")
        print("\nProcessing PDFs (this may take a while, especially for OCR)...")
        
        # Process files with progress bar
        all_results = []
        ocr_used_count = 0
        
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            try:
                result = crawler.extract_pdf_content(str(pdf_file), use_ocr=args.use_ocr)
                if result:
                    all_results.append(result)
                    if result.get('extraction_method') == 'ocr':
                        ocr_used_count += 1
            except Exception as e:
                print(f"\nError processing {pdf_file.name}: {str(e)}")
        
        # Save results
        if all_results:
            print("\nSaving results...")
            
            # Generate timestamp for filenames
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_filename = f'processed_pdfs_{timestamp}'
            
            # Save as JSON
            json_path = output_dir / f'{base_filename}.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"Saved JSON output to: {json_path}")
            
            # Save as CSV and Parquet if possible
            try:
                df = pd.json_normalize(all_results)
                
                # Save as CSV
                csv_path = output_dir / f'{base_filename}.csv'
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                print(f"Saved CSV output to: {csv_path}")
                
                # Save as Parquet
                parquet_path = output_dir / f'{base_filename}.parquet'
                df.to_parquet(parquet_path, index=False)
                print(f"Saved Parquet output to: {parquet_path}")
            except Exception as e:
                print(f"Warning: Could not create DataFrame: {str(e)}")
            
            # Calculate language distribution
            lang_dist = {}
            for item in all_results:
                lang = item.get('language', 'unknown')
                lang_dist[lang] = lang_dist.get(lang, 0) + 1
            
            # Print summary
            print("\n" + "="*50)
            print("PROCESSING SUMMARY")
            print("="*50)
            print(f"Total PDFs found: {len(pdf_files)}")
            print(f"Successfully processed: {len(all_results)}")
            if args.use_ocr:
                print(f"  - Using OCR: {ocr_used_count} files")
                print(f"  - Direct extraction: {len(all_results) - ocr_used_count} files")
            
            print("\nLanguage distribution:")
            for lang, count in sorted(lang_dist.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {lang.upper()}: {count} files")
            
            print("\nOutput saved to:")
            print(f"  - {json_path.absolute()}")
            if 'df' in locals():
                print(f"  - {csv_path.absolute()}")
                print(f"  - {parquet_path.absolute()}")
        else:
            print("\nNo valid content was extracted from any PDF files.")
            
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user!")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        if 'all_results' in locals() and all_results:
            print("\nAttempting to save processed data before exiting...")
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                error_file = output_dir / f'error_recovery_{timestamp}.json'
                with open(error_file, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)
                print(f"Recovery data saved to: {error_file}")
            except Exception as save_error:
                print(f"Failed to save recovery data: {str(save_error)}")

if __name__ == "__main__":
    main()
