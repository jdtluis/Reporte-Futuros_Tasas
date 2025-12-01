# %%
import requests
import json
import pandas as pd
from datetime import date, timedelta


def get_data_mae(type='repo'):
    from_date = (date.today()-timedelta(days=30)).isoformat()
    to_today = date.today().isoformat()
    if type=='repo':
        key = "oRueda"
        params = {
        "fechaDesde": from_date,
        "fechaHasta": to_today,
        "codigoPlazo": "",
        "unit": "REPOS",
        }

        url = "https://api.marketdata.mae.com.ar/api/mercado/repo"
    elif type=='cau':
        key = "oTitulo"
        params = {
            "fechaDesde": from_date,
            "fechaHasta": to_today
        }

        url = "https://api.marketdata.mae.com.ar/api/mercado/titulo/historicocauciones"

    response = requests.get(url, params={key: json.dumps(params)})

    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}")
    else:
        data = response.json()
        rows = []
        for day in data:
            for d in day["details"]:
                rows.append(d)
        return rows
    
def get_repo_mae():
    rows = get_data_mae('repo')
    df = pd.DataFrame(rows)
    df = df[df.moneda=='$']
    df.fecha = df.fecha.str.split('T',expand=True)[0]
    result = df.loc[df.groupby(['fecha', 'rueda'])['volumen'].idxmax()].reset_index(drop=True)
    result = result[['fecha', 'rueda', 'volumen', 'tasaPP']].pivot(columns= 'rueda', values= ['volumen', 'tasaPP'], index='fecha')
    volume = result.xs('volumen', axis=1, level=0)
    rate = result.xs('tasaPP', axis=1, level=0)
    return rate, volume


def get_cau_mae():
    rows = get_data_mae('cau')
    df = pd.DataFrame(rows)
    df = df[df.moneda=='$']
    df.fecha = df.fecha.str.split('T',expand=True)[0]
    result = df.loc[df.groupby(['fecha'])['volumen'].idxmax()].reset_index(drop=True)
    result = result[['fecha', 'plazo', 'tasaPP', 'minimo', 'maximo', 'montoConcertado']]
    result.set_index('fecha', inplace=True)
    return result

cau = get_cau_mae()
rate, volume = get_repo_mae()
rate[['REPO', 'REPX', 'SIMU']]
volume[['REPO', 'REPX', 'SIMU']]

# %%


# %%
