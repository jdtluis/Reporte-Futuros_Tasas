import requests
import pandas as pd
from datetime import date, timedelta
import urllib3
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
from bcra import bcra_fx_limits
from datetime import date, timedelta
from scipy.interpolate import make_interp_spline
plt.style.use('seaborn-v0_8-darkgrid')


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def get_futures(start,end):
    
    # API endpoint
    url = "https://apicem.matbarofex.com.ar/api/v2/closing-prices"

    # Query parameters
    params = {
        "product": "DLR",
        "segment": "Monedas",
        "type": "FUT",
        "excludeEmptyVol": "true",
        "from": start.isoformat(),
        "to": end.isoformat(),
        #"page": "1",
        "pageSize": "500",
        "sortDir": "ASC",
        "market": "ROFX"
    }

    # Send GET request
    response = requests.get(url, params=params)

    # Check if request was successful
    if response.status_code == 200:
        data = response.json()
        
        # Often APIs return data under a key like "content" or "results"
        # If so, replace 'data' below with data["content"]
        try:
            return pd.DataFrame(data['data'])
        except ValueError:
            df = pd.json_normalize(data['data'])
        
        #print(df.head())
    else:
        print(f"Request failed with status code {response.status_code}")
        print(response.text)
    return pd.DataFrame()


def get_implied_rates(df):
    rates = df[df['volume'] > 10000].pivot(index='dateTime',columns='symbol',values=['impliedRate'])
    rates.index = pd.to_datetime(rates.index)
    cols = rates.count()
    rates = rates[[c for c in cols.index if cols[c] > 10]] # Keep only columns with more than 10 non-NA values
    rates.columns = rates.columns.droplevel(0)  # Drop the top level of the MultiIndex
    return rates


def get_last_futures_close_fx_limits(df):
    from utils import last_working_day_of_month
    last_futures_close = df[df['dateTime'] == df['dateTime'].max()][['symbol','settlement']]
    last_futures_close['last_day_contract'] = last_futures_close['symbol'].str.replace('DLR', '').apply(last_working_day_of_month)
    last_futures_close[['lower', 'upper']] = last_futures_close['last_day_contract'].apply(lambda x: pd.Series(bcra_fx_limits(x.strftime("%Y-%m-%d"))))
    return last_futures_close


def plot_limits(df):
    df.rename(columns={'settlement':'Ajuste', 'upper':'Limite superior', 'lower':'Limite inferior'}, inplace=True)
    fig, ax = plt.subplots(figsize=(12,5))
    for col in df[['Ajuste','Limite superior', 'Limite inferior']]:
        y = df[col].dropna().values
        x = df['last_day_contract']

        #ax.plot(x_smooth, y_smooth)
        ax.plot(x, y, label=col)  # original points  - timedelta(hours=5)
        for x, y in zip(x, y):
            ax.text(x, y,  f'{y}', ha="left", va='top', fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precios")
    ax.legend()
    plt.xticks(rotation=45)
    plt.legend(title= 'Futuros y bandas', loc="upper left", fontsize=8, )
    plt.title("Futuros y bandas de flotación BCRA")
    plt.show()


def plot_limits_plotly(df):
    df.rename(columns={'settlement':'Ajuste', 'upper':'Limite superior', 'lower':'Limite inferior'}, inplace=True)
    import plotly.graph_objects as go

    fig = go.Figure()

    # Define the columns to plot
    columns = ['Ajuste', 'Limite superior', 'Limite inferior']

    for col in columns:
        # Drop NaNs to ensure x and y align
        temp_df = df[['last_day_contract', col]].dropna()
        x = temp_df['last_day_contract']
        y = temp_df[col]

        # Add the line
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines+markers+text',
            name=col,
            text=[f'{val:.2f}' for val in y],  # format the values
            textposition='top right',
            textfont=dict(size=10)
        ))

    # Update layout
    fig.update_layout(
        title="Futuros y bandas de flotación BCRA",
        xaxis_title="Fecha",
        yaxis_title="Precios",
        legend_title="Futuros y bandas",
        xaxis=dict(
            tickformat="%Y-%m",  # Format dates
            tickangle=45
        ),
        height=500,
        width=1000
    )

    fig.show()



def plot_smooth(df, k=3, n_points=400, figsize=(16,6)):
    """
    Plot smooth curves for each column in a DataFrame using cubic spline interpolation.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with a DatetimeIndex and numeric columns (e.g. interest rates).
    k : int, optional
        Degree of the spline (default=3 for cubic).
    n_points : int, optional
        Number of points for interpolation (default=300).
    figsize : tuple, optional
        Size of the figure (default=(12,6)).
    """
    df = df/100
    df.index = pd.to_datetime(df.index)
    df.reset_index(inplace=True)
    dates = df['dateTime']
    df.drop(columns=['dateTime'], inplace=True)
    df.columns
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(df.index))  # numeric representation of dates

    for col in df.columns:
        y = df[col].dropna().values
        x = dates[df[col].dropna().index]
        y_ = df[col].fillna(0).values  # for scatter plot
        x_col = np.arange(len(y))  # adjust for missing values

        if len(y) > k:  # spline needs enough points
            spline = make_interp_spline(x_col, y, k=k)
            x_smooth = np.linspace(x_col.min(), x_col.max(), n_points)
            y_smooth = spline(x_smooth)

            # Map x back to datetime index
            dates_smooth = np.linspace(df.index[0], df.index[-1], n_points)

            #ax.plot(x_smooth, y_smooth)
            ax.scatter(x - timedelta(hours=5), y, s=20, label=col)  # original points
            for x, y in zip(x, y):
                ax.text(x, y, f"{y*100:.0f}%", ha="left", va="center", fontsize=8)

        else:
            #pass
            ax.plot(dates, y_, label=col)  # fallback: straight line
    #ax.plot(TAMAR_SERIE.index, TAMAR_SERIE['valor']/100, label='TAMAR Bancos Privados', color='black', linestyle='--')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))  # every 1 day
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Tasa implicita")
    ax.legend()
    plt.xticks(rotation=45)
    plt.legend(title= 'Implied Rates', loc="lower left", fontsize=8)
    plt.title("Tasas implicitas por contrato y fecha")
    plt.show()


def plot_futures(data, kind='line', ylabel="Implied Rate", title="Implied Rates Over Time", legend_name = "Rates"):
    # Ensure index is datetime
    def human_format(num):
        if num >= 1_000_000_000:
            return f"{num/1_000_000_000:.1f}B"
        elif num >= 100_000:
            return f"{num/1_000_000:.1f}M"
        elif num >= 1_000:
            return f"{num/1_000:.1f}k"
        else:
            return str(int(num))
    
    data.index = pd.to_datetime(data.index)

    # Plot all columns
    if kind == 'line':
        ax = data.plot(figsize=(15, 6), kind=kind, marker='o', markersize=4)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        # Add grid
        plt.grid(True, linestyle="-", alpha=0.6)
    elif kind == 'bar':
        data_bar = data.copy()
        data_bar.index = data_bar.index.strftime("%Y-%m-%d")
        ax = data_bar.plot(figsize=(24, 8), kind=kind, stacked=True, width=0.9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else f"{int(x)}"
        ))
        
        for container in ax.containers:
            ax.bar_label(container, labels=[human_format(v.get_height()) if v.get_height() > 100000 else "" 
                                    for v in container], 
                 label_type="center", fontsize=10, color="black")

    # Format x-axis: show date every 5 days
    #ax.xaxis.set_ticks(data.index)


    # Rotate labels for readability
    plt.xticks(rotation=45)

    # Add labels and title
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.title(title)

    # Add legend
    plt.legend(title=legend_name, loc="upper left", fontsize=10, ncols=len(data.columns))

    plt.tight_layout()
    plt.show()

