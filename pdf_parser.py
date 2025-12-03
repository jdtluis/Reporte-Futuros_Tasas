import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import tabula.io as tabio
import numpy as np
import pandas as pd
from datetime import date
import io


def search_links():
    url = "https://www.iamc.com.ar/informeslecap/"
    resp = requests.get(url, verify=False)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    #report_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().find("informeletrasybonos")!=-1:
            # make full absolute URL
            full_url = urljoin(url, href)
            #report_links.append(full_url)
            # Only first match. Last report
            return full_url


def fetch_pdf_links():
    # 1. Fetch the page
    url = search_links()
    resp = requests.get(url, verify=False)
    resp.raise_for_status()

    # 2. Parse HTML
    soup = BeautifulSoup(resp.text, "html.parser")

    # 3. Find links to PDF files
    pdf_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".pdf"):
            # make full absolute URL
            full_url = urljoin(url, href)
            pdf_links.append(full_url)
    return pdf_links


def get_data_from_pdf(file):
    tables = tabio.read_pdf(file, pages="2")
# Asumo que hay una sola tabla
    df = tables[0]
    mask = df.astype(str).apply(lambda col: col.str.contains('Simultáneas', case=False))

# Get the integer-based row and column indices from the mask
    row_indices, col_indices = np.where(mask)

    df = df.iloc[int(row_indices[0]):df.shape[0],int(col_indices[0]):df.shape[1]]
    df = df.loc[~df.isna().any(axis=1), ~df.isna().all(axis=0)]
    df = pd.concat([df.iloc[:,0].str.split(' ', expand=True), df.iloc[:,1]], axis=1)
    df = df.set_axis(range(df.shape[1]), axis=1)
    df.rename(columns={0:'Fecha', 1:'Tasa', 2:'Monto'}, inplace=True)
    df['Fecha'] = pd.to_datetime(df['Fecha']).dt.strftime('%Y-%m-%d')
    df['Tasa'] = df['Tasa'].str.replace('%', '').astype(float)
    df['Monto'] = df['Monto'].str.replace(',', '').astype(float)
    df.sort_values(by='Fecha',inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_simu_data():
    files = fetch_pdf_links()
    r = requests.get(files[0], verify=False)
    f = io.BytesIO(r.content)
    if files:
        data = get_data_from_pdf(f)
        data.to_csv('simu_byma.csv', index=False)
        return data
