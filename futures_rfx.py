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
import plotly.graph_objects as go
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
        width=1500
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


def plot_smooth_ia(df, k=3, n_points=400, figsize=(16,6)):
    # Ensure datetime index
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if 'dateTime' in df.columns:
        df['dateTime'] = pd.to_datetime(df['dateTime'])
        df.set_index('dateTime', inplace=True)

    # Convert to percentage (if values look like percents)
    df = df / 100

    fig, ax = plt.subplots(figsize=figsize)

    for col in df.columns:
        y = df[col].dropna()
        if len(y) <= k:
            ax.plot(y.index, y.values, label=col, marker='o')
            continue

        # Create smooth curve
        x_numeric = np.arange(len(y))
        spline = make_interp_spline(x_numeric, y.values, k=k)
        x_smooth = np.linspace(0, len(y)-1, n_points)
        y_smooth = spline(x_smooth)

        # Map numeric x back to datetime
        x_dates = np.linspace(y.index[0].value, y.index[-1].value, n_points)
        x_dates = pd.to_datetime(x_dates)

        # Plot
        ax.plot(x_dates, y_smooth, label=col)
        ax.scatter(y.index - timedelta(hours=5), y.values, s=20)
        for xi, yi in zip(y.index, y.values):
            ax.text(xi, yi, f"{yi*100:.0f}%", ha="left", va="center", fontsize=8)

    # Format axes
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Tasa implícita")
    ax.legend(title='Implied Rates', loc="lower left", fontsize=8)
    plt.title("Tasas implícitas por contrato y fecha")
    plt.tight_layout()
    plt.show()
  

def plot_smooth_plotly(df, k=3, n_points=400):
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if 'dateTime' in df.columns:
        df['dateTime'] = pd.to_datetime(df['dateTime'])
        df.set_index('dateTime', inplace=True)

    df = df / 100

    fig = go.Figure()

    for col in df.columns:
        y = df[col].dropna()
        if len(y) <= k:
            fig.add_trace(go.Scatter(
                x=y.index, y=y,
                mode='lines+markers+text',
                name=col,
                text=[f"{val*100:.0f}%" for val in y],
                textposition="top center"
            ))
            continue

        # Smooth curve
        x_numeric = np.arange(len(y))
        spline = make_interp_spline(x_numeric, y.values, k=k)
        x_smooth = np.linspace(0, len(y)-1, n_points)
        y_smooth = spline(x_smooth)
        x_dates = np.linspace(y.index[0].value, y.index[-1].value, n_points)
        x_dates = pd.to_datetime(x_dates)

        fig.add_trace(go.Scatter(
            x=x_dates,
            y=y_smooth,
            mode='lines',
            name=f"{col} (smooth)"
        ))

        fig.add_trace(go.Scatter(
            x=y.index,
            y=y,
            mode='markers+text',
            name=f"{col} (points)",
            text=[f"{val*100:.0f}%" for val in y],
            textposition="top center"
        ))

    fig.update_layout(
        title="Tasas implícitas por contrato y fecha",
        xaxis_title="Fecha",
        yaxis_title="Tasa implícita (%)",
        yaxis_tickformat=".0%",
        hovermode="x unified",
        template="plotly_white",
        legend_title="Implied Rates",
        height=600,
        width=1000
    )

    fig.show()


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



def plot_futures_ia(data, kind='line', ylabel="Implied Rate", title="Implied Rates Over Time", legend_name="Rates"):
    """
    Plot futures data as line or stacked bar chart.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with datetime index and numeric columns.
    kind : str, optional
        Type of plot: 'line' or 'bar'. Default 'line'.
    ylabel : str
        Label for y-axis.
    title : str
        Plot title.
    legend_name : str
        Title for legend.
    """
    def human_format(num):
        if num >= 1_000_000_000:
            return f"{num / 1_000_000_000:.1f}B"
        elif num >= 1_000_000:
            return f"{num / 1_000_000:.1f}M"
        elif num >= 1_000:
            return f"{num / 1_000:.1f}k"
        else:
            return f"{num:.0f}"

    # Copy and ensure datetime index
    df = data.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    plt.style.use('seaborn-v0_8-darkgrid')

    if kind == 'line':
        ax = df.plot(figsize=(15, 6), marker='o', markersize=4)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.grid(True, linestyle="--", alpha=0.6)

    elif kind == 'bar':
        df_bar = df.copy()
        df_bar.index = df_bar.index.strftime("%Y-%m-%d")
        ax = df_bar.plot(kind='bar', stacked=True, figsize=(18, 7), width=0.9)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: human_format(x))
        )
        for container in ax.containers:
            labels = [
                human_format(v.get_height()) if v.get_height() > 100000 else ""
                for v in container
            ]
            ax.bar_label(container, labels=labels, label_type="center", fontsize=9)

    # Format layout
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title=legend_name, fontsize=9, loc="upper left", ncols=min(len(df.columns), 3))
    plt.tight_layout()
    plt.show()


def plot_futures_plotly(data, kind='line', ylabel="Implied Rate", title="Implied Rates Over Time", legend_name="Rates"):
    """
    Interactive Plotly version of the futures plot.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with datetime index and numeric columns.
    kind : str, optional
        'line' or 'bar' (stacked). Default 'line'.
    ylabel : str
        Label for y-axis.
    title : str
        Plot title.
    legend_name : str
        Title for legend.
    """
    def human_format(num):
        if num >= 1_000_000_000:
            return f"{num / 1_000_000_000:.1f}B"
        elif num >= 1_000_000:
            return f"{num / 1_000_000:.1f}M"
        elif num >= 1_000:
            return f"{num / 1_000:.1f}k"
        else:
            return f"{num:.0f}"

    df = data.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    fig = go.Figure()

    if kind == 'line':
        for col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col],
                mode='lines+markers',
                name=col,
                text=[f"{y:,.0f}" for y in df[col]],
                hovertemplate='%{x|%Y-%m-%d}<br>%{y:,.2f}<extra>' + col + '</extra>'
            ))

    elif kind == 'bar':
        for col in df.columns:
            fig.add_trace(go.Bar(
                x=df.index, y=df[col],
                name=col,
                text=[human_format(v) for v in df[col]],
                hovertemplate='%{x|%Y-%m-%d}<br>%{y:,.0f}<extra>' + col + '</extra>'
            ))
        fig.update_layout(barmode='stack')

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=ylabel,
        hovermode="x unified",
        template="plotly_white",
        legend_title=legend_name,
        width=1000,
        height=600
    )

    fig.show()
