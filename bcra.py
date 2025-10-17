import requests as req
import pandas as pd
from datetime import date, timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def get_tamar_serie(start, end, id_variable=44):

    #end = date.today()
    #start = end - timedelta(days=30)

    #start = '2025-09-01'
    #end = '2025-10-30'
    #id_variable = 44 # 'TAMAR Bancos Privados'

    catalog = 'https://www.bcra.gob.ar/Catalogo/Content/files/json/principales-variables-v4.json'
    cat = req.get(catalog, verify=False).json()
    base_url = cat['servers'][0]['url']

    serie_values = req.get(base_url + f'estadisticas/v4.0/Monetarias/{str(id_variable)}/', verify=False)
    if serie_values.status_code != 200:
        raise Exception(f"Error fetching data: {serie_values.status_code}")
    serie_values = serie_values.json()
    serie_values = pd.DataFrame(serie_values['results'][0]['detalle'])
    serie_values.set_index('fecha', inplace=True)

    serie_values_filtered = serie_values[(serie_values.index >= start.isoformat()) & (serie_values.index <= end.isoformat())]
    serie_values_filtered = serie_values_filtered.sort_values(by='fecha', ascending=True)
    serie_values_filtered.rename(columns={'valor': 'Tasa TAMAR'}, inplace=True)
    #serie_values_filtered.plot(kind='line', figsize=(15, 5), title='Tasa TAMAR Bancos Privados')
    return serie_values_filtered



def plot_tamar_serie(df, figsize=(15, 5)):
    """
    Plots the TAMAR series using matplotlib.
    :param df: DataFrame with a 'valor' column and a datetime index.
    :param figsize: Tuple for the figure size.
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df.index, df['Tasa TAMAR']/100, color='blue', linestyle=':', marker='o', markersize=4)
    #df.index = pd.to_datetime(df.index)
    #df.reset_index(inplace=True)
    #dates = df.fecha
    for x, y in zip(df.index, df['Tasa TAMAR'].values):
        ax.text(x, y/100, f"{y:.0f}%", ha="left", va="top", fontsize=8)
    ax.set_title('Tasa TAMAR Bancos Privados')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Tasa TAMAR')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    plt.xticks(rotation=45)
    plt.legend(title=df.columns[0], fontsize=8, handles=[])
    plt.show()
