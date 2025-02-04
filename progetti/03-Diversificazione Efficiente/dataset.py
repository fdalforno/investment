import requests
from bs4 import BeautifulSoup
import re
from typing import Optional, Union
import pandas as pd
import numpy as np
from collections import defaultdict
import yfinance as yf

def create_soup(url:str, **params: Union[str, int]) -> Optional[BeautifulSoup]:
    
    for key, value in params.items():
        url = url.replace(f"{{{key}}}", value)
    
    page = requests.get(url)
    page.raise_for_status()

    content = page.content
    content = b" ".join(content.splitlines())
    regex = br"<body.*?>(.*?)<\/body>"
    matches = re.search(regex, content)
    soupPage = BeautifulSoup(matches[0], "html.parser")

    return soupPage


def get_component_table(soup: BeautifulSoup) -> pd.DataFrame:
    regex = r"[A-Z]{2}[0-9A-Z]+\.html"

    table = soup.find_all('table', {'class': 'm-table'})
    
    titoli = []
    isins =[]
    
    for row in table[1].find_all("tr",class_=''):
        columns = row.find_all("td")
        title = columns[1]

        link = title.find("a")

        titolo = link.get_text()
        titoli.append(titolo.strip())

        isin = link.get("href")
        matches = re.search(regex, isin)
        isins.append(matches[0].replace(".html",""))

    return pd.DataFrame({'titolo': titoli, 'isin': isins})


def get_number_of_pages(soup: BeautifulSoup) -> int:
    pages = soup.find('div', {'class': 'm-pagination'})
    links = pages.find_all("a")

    max_page = 0

    for link in links:
        number = link.get_text()
        if number.isnumeric():
            page = int(number)
            if page > max_page:
                max_page = page
    
    return max_page


def get_stock_market_data(isin:str) -> defaultdict(str):
    url = f"https://www.borsaitaliana.it/borsa/azioni/dati-completi.html?isin={isin}&lang=en"
    soup = create_soup(url)

    result = defaultdict(str)
    grids = soup.find_all("table",'m-table')
    for grid in grids:
        for row in grid.find_all("tr"):
            cols = row.find_all("td")
            if len(cols) > 1:
                index = cols[0].get_text()
                value = cols[1].get_text()
                result[index.strip()] = value.strip()
    return result

def get_return_data(tickers:pd.Series, market:str, start_date:str, end_date:str) -> pd.DataFrame:
        val = tickers.values

        returnData = None

        for chunk in np.array_split(val, 4):
            chList = chunk.tolist()
            chList = [symbol + f".{market}" for symbol in chList]

            dfChunk = yf.download(chList, start=start_date, end=end_date)
            if returnData is None:
                returnData = dfChunk['Close']
            else:
                returnData = pd.concat([returnData, dfChunk['Close']], axis=1)

        return returnData.pct_change()


class FTSE_MIB:
    def __init__(self):
        self.index = "ftse-mib"
        self.base = self.__get_base()
        self.__calc_market_data()
        
    def __get_base(self):
        url = "https://www.borsaitaliana.it/borsa/azioni/{indice}/lista.html?&page={page}"
        soup = create_soup(url, indice=self.index, page="1")

        pages = get_number_of_pages(soup)
        component_table = get_component_table(soup)

        for i in range(2, pages+1):
            soup = create_soup("https://www.borsaitaliana.it/borsa/azioni/{indice}/lista.html?&page={page}", indice=self.index, page=str(i))
            component_table_page = get_component_table(soup)
            component_table = pd.concat([component_table, component_table_page], ignore_index=True)
        
        component_table.set_index('isin', inplace=True)
        return component_table

    def __calc_market_data(self):

        for isin in self.base.index:
            data = get_stock_market_data(isin)
            self.base.loc[isin, 'sector'] = data['Super Sector']

            cap = data['Market Capitalization']
            cap = cap.replace(",", "")

            self.base.loc[isin, 'capitalization'] = float(cap)
            self.base.loc[isin, 'ticker'] = data['Alphanumeric Code']

    def get_dataframe(self):
        return self.base

    
        

if __name__ == '__main__':
    ftse = FTSE_MIB()
    print(get_return_data(ftse.get_dataframe()['ticker'],'MI', '2000-01-01', '2024-12-31'))

    

   
    