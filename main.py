import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import itertools
import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA









# Set page config
st.set_page_config(page_title="Steam Game Trends", layout="centered")

# functions and calls for inserting background GIFs)
def set_bg_gif(gif_url, opacity=0.2):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: transparent;
        }}

        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: url("{gif_url}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
            opacity: {opacity};
            z-index: -1; 
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


set_bg_gif("https://forums.terraria.org/index.php?attachments/snowbiome720p-gif.412700/", opacity=0.2)

# 加载 Google Fonts：Inter 字体（可放在页面最上方执行一次即可）
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@600&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Title layout and color setting
col1, col2 = st.columns([1, 10])

with col1:
    st.image("image/steamicon.png", width=90)

with col2:
    st.markdown("""
    <div style='font-family: "Inter", sans-serif; 
                font-size: 32px; 
                font-weight: 600; 
                background: linear-gradient(to right, #2a7fdb, #2ed0a4);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-top: 10px;'>
        Steam Game Peak Player Trends
    </div>
    """, unsafe_allow_html=True)


st.markdown("""
<p style="font-size:18px;">
    This is an interactive webpage to display the predictions of the trend of Steam Games.
    The data obtained for this forecasting model was obtained from
    <a href="https://steamdb.info/" target="_blank" style="color:#1f77b4; text-decoration: none; font-weight: 500;">
        Steam's Official Website
    </a>.
</p>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    games = {
        'Apex Legends': pd.read_csv('datasets/ApexLegends_players.csv'),
        'Baldur\'s Gate 3': pd.read_csv("datasets/Baldur'sGate3_players.csv"),
        'CSGO': pd.read_csv('datasets/CSGO_players.csv'),
        'Cyberpunk 2077': pd.read_csv('datasets/Cyberpunk2077_players.csv'),
        'Destiny 2': pd.read_csv('datasets/Destiny2_players.csv'),
        'Dota 2': pd.read_csv('datasets/Dota2_players.csv'),
        'Elden Ring': pd.read_csv('datasets/EldenRing_players.csv'),
        'Monster Hunter World': pd.read_csv('datasets/MonsterHunterWorld_players.csv'),
        'PUBG': pd.read_csv('datasets/PUBG_players.csv'),
        'Terraria': pd.read_csv('datasets/Terraria_players.csv'),
    }

    for name, df in games.items():
        df['Date'] = pd.to_datetime(df['Date'])
        df['Peak'] = df['Peak'].astype(str).str.replace(',', '')  # Remove commas
        df['Peak'] = pd.to_numeric(df['Peak'], errors='coerce')
        df = df.dropna(subset=['Peak'])
        games[name] = df.sort_values('Date')

    return games

games = load_data()
#--------------------------------------------------------
# sidebar
with st.sidebar:
    st.markdown("""
    <style>
    .sidebar-title {
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
    }
    .sidebar-link {
        display: block;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 8px;
        text-decoration: none !important;
        font-size: 16px;
        color: black;
        transition: all 0.2s ease-in-out;
    }
    .sidebar-link:hover {
        background-color: #c8d1e0;
        color: white !important;
        font-weight: bold;
        text-decoration: none !important;
    }
    hr {
        border: 1px solid #ddd;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

    # sidebar - title
    st.markdown('<div class="sidebar-title">📋   Main Menu</div>', unsafe_allow_html=True)

    # sidebar - link
    st.markdown("""
    <a class="sidebar-link" href="#previous-trend">📈   Previous Trend</a>
    <a class="sidebar-link" href="#time-series-analysis">📊   Time-Series Analysis</a>
    <a class="sidebar-link" href="#forecasting-future-player-trends-prophet">🔭   Forecasting</a>
    <a class="sidebar-link" href="#steam-game-player-clustering">✨   Clustering</a>
    <hr>
    """, unsafe_allow_html=True)


#--------------------------------------------------------
st.write("Select A Game To View Its Known Trend")
# Sidebar dropdown
selected_game = st.selectbox("Choose a game:", list(games.keys()))

st.header("📈   Previous Trend")
st.write("The Chart Below Shows Its Previous Trend")
# Plot

import matplotlib.dates as mdates

df = games[selected_game].copy()
fig, ax = plt.subplots(figsize=(10, 5))

fig.patch.set_alpha(0.0)      # 设置整张图的背景为透明
ax.patch.set_alpha(0.0)       # 设置绘图区背景为透明

# Plot 区域填充 + 线
ax.fill_between(df['Date'], df['Peak'], color='#A1D6E2', alpha=0.6, label='Peak Area')
ax.plot(df['Date'], df['Peak'], color='#0E5E85', linewidth=2.5, label='Peak Trend')

# 突出最大值点（可选）
max_date = df.loc[df['Peak'].idxmax(), 'Date']
max_value = df['Peak'].max()
ax.scatter([max_date], [max_value], color='red', s=50, zorder=5)
ax.annotate(f'Max: {int(max_value):,}', xy=(max_date, max_value), xytext=(15, 10),
            textcoords='offset points', fontsize=10, color='red', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red'))

# 样式美化
ax.set_title(f"{selected_game} - Peak Player Trend", fontsize=18, fontweight='bold', pad=15)
ax.set_xlabel("Year/Month", fontsize=12)
ax.set_ylabel("Peak Players", fontsize=12)
ax.grid(True, linestyle='--', alpha=0.5)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

# 日期格式优化
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))  # 每月一个刻度
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # 年-月格式
fig.autofmt_xdate()

# 去掉右边和上边边框线
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 图例（可选）
ax.legend()

# 显示图表
st.pyplot(fig)

#-------------------------------------------------------------------------
# Time series analysis
import plotly.graph_objects as go

# 设置索引和填补缺失值
df = games[selected_game].copy()
df.set_index('Date', inplace=True)
df = df.asfreq('D')
df['Peak'] = df['Peak'].interpolate(method='linear')

# 标题
st.header("📊 Time Series Analysis")
st.write("The chart below shows daily peak players (linear interpolation applied) and Augmented Dickey-Fuller (ADF) stationarity check.")

# 创建 Plotly 图表
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df.index, y=df['Peak'],
    mode='lines',
    name='Daily Peak Players',
    line=dict(color="#0E5E85", width=2.5)
))

fig.update_layout(
    title=dict(
        text=f'Monthly Peak Players - {selected_game}',
        font=dict(size=22),  # ✅ 设置字号略小于 header
        x=0.5,  # 标题居中
        xanchor='center'
    ),
    xaxis_title='Year/Month',
    yaxis_title='Peak Players',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Segoe UI', size=14),
    xaxis=dict(
        tickformat="%Y-%m",
        showgrid=True,
        gridcolor='rgba(200,200,200,0.3)'
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='rgba(200,200,200,0.3)',
        tickformat=','
    )
)

# 显示图表
st.plotly_chart(fig, use_container_width=True)


#-----------------------------------------------------------------------------
# Rolling Mean
import plotly.graph_objects as go

# 计算滚动均值（30日）
df['Rolling Mean (30 days)'] = df['Peak'].rolling(window=30).mean()

# 创建交互式折线图
fig = go.Figure()

# 原始数据线
fig.add_trace(go.Scatter(
    x=df.index, y=df['Peak'],
    mode='lines',
    name='Daily Peak',
    line=dict(color='#0E5E85', width=2)
))

# 滚动均线
fig.add_trace(go.Scatter(
    x=df.index, y=df['Rolling Mean (30 days)'],
    mode='lines',
    name='30-Day Rolling Mean',
    line=dict(color='#F765A3', width=3, dash='dash')
))

# 图表布局
fig.update_layout(
    title=dict(
        text=" 30-Day Rolling Mean vs Daily Peak Players",
        font=dict(size=21),
        x=0.5,
        xanchor='center'
    ),
    xaxis_title='Year/Month',
    yaxis_title='Peak Players',
    plot_bgcolor='rgba(0,0,0,0)',     # 图背景透明
    paper_bgcolor='rgba(0,0,0,0)',    # 页面背景透明
    font=dict(family='Segoe UI', size=14),
    xaxis=dict(
        tickformat="%Y-%m",
        showgrid=True,
        gridcolor='rgba(200,200,200,0.3)'
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='rgba(200,200,200,0.3)',
        tickformat=','
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    )
)

st.plotly_chart(fig, use_container_width=True)


# Seasonal Decomposition
import matplotlib.pyplot as plt

st.subheader("Seasonal Decomposition")

# 执行分解
decomposition = seasonal_decompose(df['Peak'], model='additive', period=30)

# 创建子图，设置透明背景
fig, axes = plt.subplots(4, 1, figsize=(10, 10), constrained_layout=True)
fig.patch.set_alpha(0.0)  # 整体背景透明

# 子图样式统一参数
plot_kwargs = dict(grid=True, linewidth=2, alpha=0.9)

# 绘制各部分
decomposition.observed.plot(ax=axes[0], title='Observed', color='#1995AD', **plot_kwargs)
decomposition.trend.plot(ax=axes[1], title='Trend', color='#1C4E80', **plot_kwargs)
decomposition.seasonal.plot(ax=axes[2], title='Seasonal', color='#F39C12', **plot_kwargs)
decomposition.resid.plot(ax=axes[3], title='Residual', color='#7F8C8D', **plot_kwargs)

# 美化每个子图
for ax in axes:
    ax.set_facecolor('none')  # 区域背景透明
    ax.title.set_fontsize(14)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# 展示图表
st.pyplot(fig)


# ADF Test
st.subheader("Stationarity Check (ADF Test)")
adf_result = adfuller(df['Peak'].dropna())
st.markdown(f"**ADF Statistic**: {adf_result[0]:.4f}")
st.markdown(f"**p-value**: {adf_result[1]:.4f}")
st.markdown("**Critical Values:**")
for key, value in adf_result[4].items():
    st.markdown(f"- {key}: {value:.4f}")
if adf_result[1] < 0.05:
    st.success("The series is stationary (reject H0)")
else:
    st.warning("The series is non-stationary (fail to reject H0)")
    
    


# ACF & PACF Plots

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

st.subheader("ACF and PACF")

# 创建子图
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.patch.set_alpha(0.0)  # 整张图背景透明

# 绘制 ACF 和 PACF
plot_acf(df['Peak'].dropna(), ax=axes[0], alpha=0.05)
plot_pacf(df['Peak'].dropna(), ax=axes[1], alpha=0.05)

# 样式优化
axes[0].set_title('Autocorrelation (ACF)', fontsize=14, fontweight='bold')
axes[1].set_title('Partial Autocorrelation (PACF)', fontsize=14, fontweight='bold')

for ax in axes:
    ax.set_facecolor('none')  # 绘图区透明
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# 显示图
st.pyplot(fig)

#------------------------------------------------------

# --- Section: Forecasting with Prophet ---
st.header("🔭 Forecasting Future Player Trends (Prophet)")
st.write("This section provides a 12-month forecast of peak players using Facebook's Prophet model.")

from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")

# Prepare data for Prophet
import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 预处理数据
import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 预处理数据
df_prophet = games[selected_game][['Date', 'Peak']].rename(columns={'Date': 'ds', 'Peak': 'y'})
df_prophet = df_prophet.dropna()

if df_prophet.shape[0] >= 12:
    with st.spinner("Training Prophet model..."):
        model = Prophet(growth='linear', yearly_seasonality=True, seasonality_mode='multiplicative')
        model.fit(df_prophet)

        future = model.make_future_dataframe(periods=12, freq='M')
        forecast = model.predict(future).rename(columns={
            'ds': 'Time',
            'yhat': 'forecast_value',
            'yhat_lower': 'forecast_value_lower',
            'yhat_upper': 'forecast_value_upper'
        })

        # ✅ 1. 主图：预测 vs 实际
        yhat = forecast[['Time', 'forecast_value', 'forecast_value_lower', 'forecast_value_upper']]
        actual = df_prophet.copy()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=actual['ds'], y=actual['y'],
            mode='lines+markers', name='Actual',
            line=dict(color='#1995AD'), marker=dict(size=4)
        ))

        fig.add_trace(go.Scatter(
            x=yhat['Time'], y=yhat['forecast_value'],
            mode='lines', name='Forecast',
            line=dict(color='#F39C12', width=3)
        ))

        fig.add_trace(go.Scatter(
            x=yhat['Time'], y=yhat['forecast_value_upper'],
            line=dict(width=0), hoverinfo='skip', mode='lines', name='Upper Bound', showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=yhat['Time'], y=yhat['forecast_value_lower'],
            fill='tonexty', fillcolor='rgba(243, 156, 18, 0.2)',
            line=dict(width=0), hoverinfo='skip', mode='lines', name='Lower Bound', showlegend=False
        ))

        fig.update_layout(
            title='Interactive Forecast - Peak Players',
            xaxis_title='Date',
            yaxis_title='Peak Players',
            font=dict(size=14),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )

        st.subheader("Forecast Chart (Next 12 Months)")
        st.plotly_chart(fig, use_container_width=True)

        # ✅ 2. 分解图：Trend / Seasonality
        components = ['trend']
        if 'weekly' in forecast.columns: components.append('weekly')
        if 'yearly' in forecast.columns: components.append('yearly')

        fig_comp = make_subplots(rows=len(components), cols=1, shared_xaxes=True, subplot_titles=[c.title() for c in components])

        for i, comp in enumerate(components, start=1):
            fig_comp.add_trace(go.Scatter(
                x=forecast['Time'], y=forecast[comp],
                mode='lines', name=comp.title(), line=dict(width=2)
            ), row=i, col=1)

        fig_comp.update_layout(
            height=300 * len(components),
            title_text="Forecast Components (Trend / Seasonality)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )

        st.subheader("Forecast Components")
        st.plotly_chart(fig_comp, use_container_width=True)

        # ✅ 3. 可展开预测数据表
        with st.expander("Show Forecast Data"):
            st.dataframe(forecast[['Time', 'forecast_value', 'forecast_value_lower', 'forecast_value_upper']].tail(12).reset_index(drop=True))

        st.success("Forecast complete.")
else:
    st.warning("Not enough data to train the forecasting model (need at least 12 months).")

#----------------------------------------------------




# Feature extraction
features = []
game_names_initial = []

for name, df in games.items():
    df['Peak'] = pd.to_numeric(df['Peak'], errors='coerce')
    df['Gain'] = pd.to_numeric(df['Gain'], errors='coerce')
    df['%Gain'] = df['%Gain'].astype(str).str.replace('%', '').str.replace(',', '')
    df['%Gain'] = pd.to_numeric(df['%Gain'], errors='coerce')

    # 计算各列均值
    col_means = df[['Peak', 'Gain', '%Gain']].mean()

    # 如果所有列均值都为NaN，则跳过
    if col_means.isnull().any():
        print(f"❌ Skipped (all NaN): {name}")
        continue

    # 用均值填补缺失值
    df_filled = df[['Peak', 'Gain', '%Gain']].fillna(col_means)

    features.append([
        df_filled['Peak'].mean(),
        df_filled['Peak'].max(),
        df_filled['Peak'].std(),
        df_filled['Gain'].mean(),
        df_filled['%Gain'].mean()
    ])
    game_names_initial.append(name)

feature_df = pd.DataFrame(features, columns=[
    'Avg Peak', 'Max Peak', 'Peak StdDev', 'Avg Gain', 'Avg %Gain'
], index=game_names_initial)

feature_df.dropna(inplace=True)
game_names = feature_df.index.tolist()





st.header("✨ Steam Game Player Clustering")
st.write("The chart below shows the grouping of Steam game trends and visualizes them using PCA.")


# Scaling and clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(feature_df)

kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
clusters = kmeans.fit_predict(X_scaled)
feature_df['Cluster'] = clusters

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plotting
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import FancyBboxPatch

# 使用更鲜明的颜色映射，如 Set2（也可试试 tab10, tab20）
cmap = cm.get_cmap('Set2', np.unique(clusters).size)

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_alpha(0.0)
ax.set_facecolor('none')

scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap=cmap, s=100, edgecolor='black', linewidth=0.5)

# 添加标签：带描边/白底，提高可读性
for i, name in enumerate(game_names):
    ax.text(
        X_pca[i, 0] + 0.03,
        X_pca[i, 1],
        name,
        fontsize=10,
        weight='bold',
        ha='left',
        va='center',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')
    )

# 美化坐标轴和标题
ax.set_title("🎮 Steam Game Clustering (K-Means + PCA)", fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel("PCA Component 1", fontsize=12)
ax.set_ylabel("PCA Component 2", fontsize=12)
ax.grid(True, linestyle='--', alpha=0.4)

# 去掉上边和右边边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 添加颜色条
cbar = fig.colorbar(scatter, ax=ax, label='Cluster', orientation='vertical')
cbar.ax.tick_params(labelsize=10)

st.pyplot(fig)




# ----------------------------------------------



def run_prophet_grid_search(game_name, df):
    paramGrid = {
        'changepoint_prior_scale': [0.01, 0.05, 0.1],
        'seasonality_prior_scale': [1.0, 5.0, 10.0],
        'seasonality_mode': ['additive', 'multiplicative']
    }
    allParamCombinations = [dict(zip(paramGrid.keys(), v)) for v in itertools.product(*paramGrid.values())]

    dfProphet = df[['Date', 'Peak']].rename(columns={'Date': 'ds', 'Peak': 'y'}).dropna()
    if dfProphet.shape[0] < 24:
        return "Insufficient data points for optimization", None, None

    rmses = []
    mapes = []
    valid_combos = []

    for params in allParamCombinations:
        try:
            modelGrid = Prophet(
                growth='linear',
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                **params
            )
            modelGrid.fit(dfProphet)

            num_points = len(dfProphet['ds'])
            initial_days = str(max(365, int(num_points * 0.5 * 30))) + ' days'
            period_days = str(max(90, int(num_points * 0.15 * 30))) + ' days'
            horizon_days = str(max(180, int(num_points * 0.2 * 30))) + ' days'

            dfCv = cross_validation(modelGrid, initial=initial_days, period=period_days, horizon=horizon_days, parallel="processes")
            dfP = performance_metrics(dfCv, metrics=['rmse', 'mape'])

            if not dfP.empty:
                mape = dfP['mape'].iloc[-1]
                rmse = dfP['rmse'].iloc[-1]
                if not np.isnan(mape) and not np.isinf(mape):
                    rmses.append(rmse)
                    mapes.append(mape)
                    valid_combos.append(params)
        except Exception:
            continue

    if not mapes:
        return "All parameter combinations fail", None, None

    result_df = pd.DataFrame(valid_combos)
    result_df['MAPE'] = mapes
    result_df['RMSE'] = rmses
    best_idx = result_df['MAPE'].idxmin()
    best_params = valid_combos[best_idx]

    finalModel = Prophet(
        growth='linear',
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        **best_params
    )
    finalModel.fit(dfProphet)
    future = finalModel.make_future_dataframe(periods=12, freq='MS')
    forecast = finalModel.predict(future)

    return result_df.sort_values('MAPE'), finalModel, forecast


with st.expander("🔍 Perform Prophet parameter optimisation"):
    if st.button("Start grid search"):
        result_df, final_model, forecast = run_prophet_grid_search(selected_game, games[selected_game])

        if isinstance(result_df, str):
            st.warning(result_df)
        else:
            st.success("The grid search is complete and the best model has been trained.")
            st.dataframe(result_df)

            # plot visualization
            fig1 = final_model.plot(forecast)
            fig1.set_facecolor("none")  # 设置透明背景
            fig1.patch.set_alpha(0.0)
            fig1.gca().set_title("Forecasted Peak Player Trend", fontsize=16)
            fig1.gca().set_xlabel("Date", fontsize=12)
            fig1.gca().set_ylabel("Peak Players", fontsize=12)
            fig1.gca().grid(True, linestyle='--', alpha=0.6)

            st.pyplot(fig1)

            
            fig2 = final_model.plot_components(forecast)

            # ✅ 统一美化所有子图
            for ax in fig2.get_axes():
                ax.set_facecolor("none")                      # 背景透明
                ax.grid(True, linestyle='--', alpha=0.5)      # 柔和网格线
                ax.title.set_fontsize(14)                     # 子图标题字号
                ax.xaxis.label.set_fontsize(12)               # x轴标签字号
                ax.yaxis.label.set_fontsize(12)               # y轴标签字号

            fig2.patch.set_alpha(0.0)                         # 整体背景透明
            fig2.tight_layout()

            st.pyplot(fig2)
            
