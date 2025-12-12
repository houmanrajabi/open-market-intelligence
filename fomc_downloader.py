import logging
import requests
from datetime import datetime
from typing import List
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent
from config import settings, FOMCSettings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FOMCDownloader:
    def __init__(self, config: FOMCSettings):
        self.cfg = config
        # Ensure output directory exists
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_date_range(self) -> List[datetime]:
        """Generates dates based on settings start/end year."""
        dates = []
        for year in range(self.cfg.start_year, self.cfg.end_year + 1):
            for month in range(1, 13):
                for day in range(1, 32):
                    try:
                        dates.append(datetime(year, month, day))
                    except ValueError:
                        pass
        return dates
    
    def build_document_url(self, date: datetime, doc_type: str) -> str:
        date_str = date.strftime("%Y%m%d")
        doc_patterns = {
            "statement": f"FOMC{date_str}statement.pdf",
            "minutes": f"fomcminutes{date_str}.pdf",
            "presconf": f"FOMCpresconf{date_str}.pdf",
            "sep": f"fomcprojtabl{date_str}.pdf",
            "implementation": f"monetary{date_str}a1.pdf"
        }
        filename = doc_patterns.get(doc_type, f"FOMC{date_str}{doc_type}.pdf")
        return f"{self.cfg.base_url}/files/{filename}"
    
    def download_document(self, url: str, filename: str):
        save_path = self.cfg.output_dir / filename
        
        # Skip if already downloaded
        if save_path.exists():
            logging.info(f"Skipping (exists): {filename}")
            return

        try:
            response = requests.get(url, timeout=self.cfg.timeout)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                logging.info(f"Downloaded: {save_path}")
            else:
                pass 
        except requests.RequestException as e:
            logging.error(f"Error fetching {url}: {e}")

if __name__ == "__main__":
    downloader = FOMCDownloader(config=settings)
    
    dates = downloader.get_date_range()
    
    logging.info(f"Scanning from {settings.start_year} to {settings.end_year}...")

    for date in dates:
        for doc_type in settings.target_docs:
            url = downloader.build_document_url(date, doc_type)
            filename = url.split('/')[-1]
            downloader.download_document(url, filename)