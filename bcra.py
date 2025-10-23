import requests as req
import pandas as pd
from datetime import date, timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.graph_objects as go


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
        print(f"Request failed with status code {serie_values.status_code}")
        print(f"Request failed with status code {serie_values.text}")
        #raise Exception(f"Error fetching data: {serie_values.status_code}")
        return None
    serie_values = serie_values.json()
    serie_values = pd.DataFrame(serie_values['results'][0]['detalle'])
    serie_values.set_index('fecha', inplace=True)

    serie_values_filtered = serie_values[(serie_values.index >= start.isoformat()) & (serie_values.index <= end.isoformat())]
    serie_values_filtered = serie_values_filtered.sort_values(by='fecha', ascending=True)
    serie_values_filtered.rename(columns={'valor': 'Tasa TAMAR'}, inplace=True)
    #serie_values_filtered.plot(kind='line', figsize=(15, 5), title='Tasa TAMAR Bancos Privados')
    return serie_values_filtered


def bcra_fx_limits(date_value):
    first_date = date.fromisoformat('2025-04-14')
    initial_upper = 1400
    initial_lower = 1000
    days = (date.fromisoformat(date_value) - first_date).days
    upper = round(initial_upper * (1 + 0.01) ** (days / 30),2)
    lower = round(initial_lower * (1 - 0.01) ** (days / 30),2)
    return lower, upper


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


def plot_tamar_serie_ia(df, col='Tasa TAMAR', figsize=(15, 5)):
    """
    Plots the TAMAR series as a percentage line chart.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a numeric column (default 'Tasa TAMAR') and datetime index.
    col : str
        Column name to plot.
    figsize : tuple
        Size of the matplotlib figure.
    """
    # Ensure datetime index
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Apply style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=figsize)

    # Normalize to 0–1 range for PercentFormatter
    y = df[col] / 100
    ax.plot(df.index, y, color='blue', linestyle=':', marker='o', markersize=4, label=col)

    # Annotate points
    for x, val in zip(df.index, df[col].values):
        ax.text(x, val / 100, f"{val:.0f}%", ha="left", va="top", fontsize=8)

    # Format axes and labels
    ax.set_title(f"Tasa {col} Bancos Privados", fontsize=12)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Tasa (%)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    plt.xticks(rotation=45)
    ax.legend(title="Serie", fontsize=8, loc='upper left')
    plt.tight_layout()
    plt.show()

    
def plot_tamar_serie_plotly(df, col='Tasa TAMAR'):
    """
    Interactive Plotly version of the TAMAR series plot.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a numeric column (default 'Tasa TAMAR') and datetime index.
    col : str
        Column name to plot.
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    y = df[col] / 100  # normalize for percentage display

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=y,
        mode='lines+markers+text',
        text=[f"{v:.0f}%" for v in df[col]],
        textposition="top center",
        line=dict(color='blue', dash='dot'),
        marker=dict(size=6),
        name=col
    ))

    fig.update_layout(
        title=f"Tasa {col} Bancos Privados",
        xaxis_title="Fecha",
        yaxis_title="Tasa (%)",
        yaxis_tickformat=".0%",
        template="plotly_white",
        hovermode="x unified",
        width=1400,
        height=600,
        legend_title="Serie"
    )

    fig.show()