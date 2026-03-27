import csv
import logging
import os
from typing import Any
import requests
from bs4 import BeautifulSoup
from typing import Optional, Any
import pandas as pd
from io import StringIO

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class CrawlService:
    def __init__(self):
        ...

    def crawl_data(self, url: str, parser_type: str = "html.parser"):
        try:
            response = requests.get(url)
            
            response.raise_for_status() 

            soup = BeautifulSoup(
                response.text, 
                features=parser_type
            )
            tables = soup.find_all("table")
            
            self._process_tables(tables)
        except Exception as e:
            log.error(f"An unexpected error occurred while crawling: {e}")
            return None  
        ...

    def _process_tables(self, tables):
        try:
            processed_df = []
            date_part = ""
            for idx, table in enumerate(tables):
                df_list = pd.read_html(StringIO(str(table)))

                for df in df_list:
                    df_clean = df
                    if 'http' in df.to_string().lower():
                        df_clean = df.iloc[:-1]  
                        
                        last_row = df.iloc[-1]
                        url = last_row.astype(str).values[0]  
                        
                        if 'lich-su/' in url:
                            date_part = url.split('lich-su/')[1].replace('.html', '')
                    
                    processed_df.append(df_clean)

                df_comb = pd.concat(processed_df, ignore_index=True) 
                format_date = f"{date_part.split('-')[2]}_{date_part.split('-')[1]}_{date_part.split('-')[0]}"
                    
                filename = f"gia_vang_{format_date}_{idx+1}.csv"
                filepath = os.path.join("./static/gia_vang_hom_nay", filename)
                df_comb.to_csv(filepath, index=False)
        except Exception as e:
            log.error(f"An unexpected error occurred while crawling: {e}")
            return None  
        ...