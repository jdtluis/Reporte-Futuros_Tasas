import requests as req
import pandas as pd
from datetime import date, timedelta


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

    #serie_values_filtered.plot(kind='line', figsize=(15, 5), title='Tasa TAMAR Bancos Privados')
    return serie_values_filtered
