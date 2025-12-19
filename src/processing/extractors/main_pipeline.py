import os
import json
import time
import fitz  # PyMuPDF
from pdf2image import convert_from_path

# Import your modules
from src.processing.extractors.qwen_extractor import analyze_layout 
from src.processing.extractors.hybrid_extractor import extract_hybrid_content

# --- CONFIGURATION ---
INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/output/processing"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_document(filename):
    pdf_path = os.path.join(INPUT_DIR, filename)
    doc_id = filename.replace(".pdf", "")
    
    # Create a specific folder for this document's assets
    doc_output_dir = os.path.join(OUTPUT_DIR, doc_id)
    os.makedirs(doc_output_dir, exist_ok=True)
    
    final_doc_data = {
        "doc_id": doc_id,
        "original_file": filename,
        "metadata": {},
        "pages": []
    }
    
    print(f"🚀 Processing Document: {filename}")
    
    # 1. Get Page Count (Lightweight)
    try:
        with fitz.open(pdf_path) as doc:
            total_pages = doc.page_count
            final_doc_data["metadata"]["total_pages"] = total_pages
    except Exception as e:
        print(f"❌ Failed to open PDF {filename}: {e}")
        return

    # --- CRITICAL FIX 1: Initialize Context OUTSIDE the loop ---
    last_anchor_context = None 

    # 2. Process Page by Page
    for page_num in range(1, total_pages + 1):
        print(f"   📄 Page {page_num}/{total_pages}...")
        
        try:
            # A. Convert ONLY the current page to Image
            images = convert_from_path(
                pdf_path, 
                dpi=300, 
                first_page=page_num, 
                last_page=page_num
            )
            
            if not images:
                print(f"      ⚠️ Warning: No image generated for page {page_num}")
                continue
                
            current_img = images[0]
            
            # Save Temp Image
            temp_img_path = os.path.join(doc_output_dir, f"temp_proc_{page_num}.jpg")
            current_img.save(temp_img_path, quality=95)
            
            # --- PHASE 1: LAYOUT SCAN ---
            layout_map = analyze_layout(temp_img_path) 
            
            # --- PHASE 2: HYBRID EXTRACTION (Corrected) ---
            # We call this ONCE per page.
            # It returns the elements for this page AND the updated context for the next page.
            page_elements, last_anchor_context = extract_hybrid_content(
                pdf_path=pdf_path, 
                page_num=page_num, 
                layout_map=layout_map, 
                temp_image_path=temp_img_path,
                output_dir=doc_output_dir,
                previous_context=last_anchor_context # Pass in the context from the previous loop
            )
            
            # Add to Document Structure
            final_doc_data["pages"].append({
                "page_num": page_num,
                "element_count": len(page_elements),
                "elements": page_elements
            })
            
            # Cleanup Temp File
            if os.path.exists(temp_img_path): 
                os.remove(temp_img_path)

        except Exception as e:
            print(f"      ❌ Error processing page {page_num}: {e}")
            continue

    # 3. Save Final JSON Structure
    json_path = os.path.join(doc_output_dir, "full_structure.json")
    with open(json_path, "w", encoding='utf-8') as f:
        json.dump(final_doc_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Finished {doc_id}. Data saved to {doc_output_dir}")

# --- RUNNER ---
if __name__ == "__main__":
    target_file = "fomcprojtabl20200610.pdf"
    
    if os.path.exists(os.path.join(INPUT_DIR, target_file)):
        start_time = time.time()
        process_document(target_file)
        print(f"⏱️ Total Time: {time.time() - start_time:.2f}s")
    else:
        print(f"❌ File not found: {os.path.join(INPUT_DIR, target_file)}")