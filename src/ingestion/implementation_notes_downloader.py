import requests
import time
import os
import glob
from datetime import datetime
from typing import Set
from bs4 import BeautifulSoup
import re
from fpdf import FPDF

class NoteDownloader:
    def __init__(self, config):
        self.cfg = config
        self.base_url = "https://www.federalreserve.gov/newsevents/pressreleases"
        if not os.path.exists(self.cfg.output_dir):
            os.makedirs(self.cfg.output_dir)

    def get_target_dates_from_files(self) -> Set[str]:
        """
        Scans existing 'monetary' PDFs to find meeting dates.
        Returns: {'20200303', '20200315', ...}
        """
        files = glob.glob(os.path.join(self.cfg.output_dir, "monetary*.pdf"))
        dates = set()
        for filepath in files:
            filename = os.path.basename(filepath)
            # Expecting format "monetaryYYYYMMDD..."
            try:
                date_str = filename[8:16]
                if date_str.isdigit() and len(date_str) == 8:
                    dates.add(date_str)
            except IndexError:
                continue
        return sorted(list(dates))

    import re  # Add this import at the top of your file

    def clean_text(self, text):
        """
        Cleans text for PDF compatibility and removes excessive whitespace.
        """
        # 1. Replace Unicode characters (your existing logic)
        replacements = {
            '\u201c': '"', '\u201d': '"',
            '\u2018': "'", '\u2019': "'",
            '\u2013': '-', '\u2014': '-',
            '\u00a0': ' ',
        }
        for orig, new in replacements.items():
            text = text.replace(orig, new)

        # 2. NEW: Collapse excessive newlines
        # This replaces 3 or more newlines with just 2 (standard paragraph break)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 3. Trim leading/trailing whitespace
        return text.strip()

    def save_text_as_pdf(self, title, text_content, output_path):
        """
        Generates a simple, clean PDF with the text content.
        This ensures the text is 'selectable' and 'extractable' later.
        """
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, title, ln=True, align='C')
        pdf.ln(10)
        
        # Body
        pdf.set_font("Arial", size=11) # Standard font for easy extraction
        
        # Write text (multi_cell handles line wrapping automatically)
        # We use 'latin-1' encoding usually, but replacing chars above helps
        try:
            pdf.multi_cell(0, 6, text_content)
        except UnicodeEncodeError:
            # Fallback if unhandled characters slip through
            cleaned = text_content.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 6, cleaned)
            
        pdf.output(output_path)

    def download_implementation_notes(self):
        dates = self.get_target_dates_from_files()
        print(f"Checking {len(dates)} meeting dates for Implementation Notes...")

        for date_str in dates:
            # Target output file
            output_filename = os.path.join(self.cfg.output_dir, f"implementation{date_str}.pdf")
            
            if os.path.exists(output_filename):
                print(f"Skipping {date_str} (PDF already exists)")
                continue

            # Construct URL
            url = f"{self.base_url}/monetary{date_str}a1.htm"
            
            try:
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # --- UPDATED SELECTOR LOGIC ---
                    # We try a list of potential content containers in order of preference.
                    # 1. id="article": Common in newer press releases
                    # 2. id="content": Common in older press releases/implementation notes
                    # 3. class="col-md-8": A generic fallback for the main column
                    
                    article = soup.find('div', id='article')
                    
                    if not article:
                        article = soup.find('div', id='content')
                        
                    if not article:
                        # Find any div that HAS the class 'col-md-8' (order doesn't matter)
                        article = soup.find('div', class_='col-md-8')

                    if article:
                        # Extract text and strip extra whitespace
                        raw_text = article.get_text(separator='\n\n')
                        clean_content = self.clean_text(raw_text)
                        
                        # Save as PDF
                        title = f"Implementation Note - {date_str}"
                        self.save_text_as_pdf(title, clean_content, output_filename)
                        
                        print(f"✅ Generated PDF: implementation{date_str}.pdf")
                    else:
                        # Debugging: Print available IDs if we fail, to help you diagnose
                        ids = [tag.get('id') for tag in soup.find_all('div') if tag.get('id')]
                        print(f"⚠️  Found Page but no content for {date_str}. Available div IDs: {ids[:5]}")

                elif response.status_code == 404:
                    pass 
                
                time.sleep(1)

            except Exception as e:
                print(f"❌ Error on {date_str}: {e}")

# Example Configuration to run it
from src.utils.config import config

if __name__ == "__main__":
    downloader = NoteDownloader(config.note_downloader)
    downloader.download_implementation_notes()