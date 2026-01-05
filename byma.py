import requests
import pandas as pd
from datetime import date
import plotly.graph_objects as go



def get_cau_byma():
    cau = requests.post('https://open.bymadata.com.ar/vanoms-be-core/rest/api/bymadata/free/cauciones', json = {"excludeZeroPxAndQty":True,"Content-Type":"application/json, text/plain"}, verify=False)
    if cau.status_code != 200:
        raise Exception(f"API request failed with status code {cau.status_code}")
    else:
        cau_data = pd.DataFrame(cau.json())
        if cau.json()==[]:
            raise Exception("API returned empty data")
        else:
            # Get server date
            date_req = requests.get('https://open.bymadata.com.ar/vanoms-be-core/rest/api/bymadata/free/server-time',verify=False)
            date = date_req.json()
            today = date['datetime'].split('T')[0]
            # Process data
            cau_data = cau_data[cau_data.denominationCcy=='ARS']
            row_data = cau_data.loc[cau_data.index==cau_data['tradeVolume'].idxmax(),:][['daysToMaturity' ,'vwap', 'tradingLowPrice', 'tradingHighPrice', 'volumeAmount']]
            row_data[['vwap', 'tradingLowPrice', 'tradingHighPrice']] = row_data[['vwap', 'tradingLowPrice', 'tradingHighPrice']] * 100
            row_data.insert(0,'date', today)
            row_data.date = pd.to_datetime(row_data.date).dt.strftime('%Y-%m-%d')
            row_data.set_index('date', inplace=True)
            # Load data
            hist_cau_byma = pd.read_csv('cau_byma.csv')
            hist_cau_byma.date = pd.to_datetime(hist_cau_byma.date).dt.strftime('%Y-%m-%d')
            hist_cau_byma.set_index('date', inplace=True)
            hist_cau_byma = pd.concat([hist_cau_byma, row_data],axis=0)
            hist_cau_byma = hist_cau_byma[~hist_cau_byma.index.duplicated(keep='last')]
            # Save with added data
            hist_cau_byma.to_csv('cau_byma.csv')
            return hist_cau_byma


def plot_cau_plotly(df):
    fig = go.Figure()

    for col in df.columns:
        x = df.index
        y = df[col]
        mode = 'lines+markers+text'

        # Add the line
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode=mode,
            name=col,
            text=[f'{val:.2f}' for val in y],  # format the values
            textposition='top right',
            textfont=dict(size=10)
        ))

    # Update layout
    fig.update_layout(
        title="Cauciones BYMA y MAE",
        xaxis_title="Fecha",
        yaxis_title="Tasa",
        legend_title="Tasas",
        xaxis=dict(
            tickformat="%Y-%m-%d",  # Format dates
            tickangle=45
        ),
        height=600,
        width=1000
    )
    fig.update_xaxes(type='category')
    fig.show()
