import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Set page config
st.set_page_config(page_title="Steam Game Trends", layout="centered")

st.title("🎮 Steam Game Peak Player Trends")
st.write("This is an interactive webpage to display the predicions of the trend of Steam Games. The data obtain for this forecastting model was obtained from [Steam's Official Website](https://steamdb.info/).")

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
st.markdown("---")
st.markdown("## Table of Contents")
st.markdown("""
* [Section 1: Previous Trend](#previous-trend)
* [Section 2: Time-Series Analysis](#time-series-analysis)
* [Section 3: ](#)
* [Section 4: ](#)
""")
st.markdown("---")

st.write("Select A Game To View Its Known Trend")
# Sidebar dropdown
selected_game = st.selectbox("Choose a game:", list(games.keys()))

st.header("Previous Trend")
st.write("The Chart Below Shows Its Previous Trend")
# Plot
df = games[selected_game].copy()
fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(df['Date'], df['Peak'], color='skyblue', alpha=0.5)
ax.plot(df['Date'], df['Peak'], color='blue')
ax.set_title(f"{selected_game} - Peak Player Trend")
ax.set_xlabel("Date")
ax.set_ylabel("Peak Players")
ax.grid(True)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

st.pyplot(fig)

st.header("Time Series Analysis")
st.write("The Charts Below Shows Its Time Series Analysis and Augmented Dickey-Fuller (ADF) test")
df.set_index('Date', inplace=True)
df = df.asfreq('D')  # Resample to daily
df['Peak'] = df['Peak'].interpolate(method='linear')

st.subheader(f"Daily Peak Players - {selected_game}")
st.line_chart(df['Peak'])

# Rolling Mean
df['Rolling Mean (30 days)'] = df['Peak'].rolling(window=30).mean()
st.subheader("Rolling Mean (30 Days)")
st.line_chart(df[['Peak', 'Rolling Mean (30 days)']])

# Seasonal Decomposition
st.subheader("Seasonal Decomposition")
decomposition = seasonal_decompose(df['Peak'], model='additive', period=30)

fig, axes = plt.subplots(4, 1, figsize=(10, 10))
decomposition.observed.plot(ax=axes[0], title='Observed')
decomposition.trend.plot(ax=axes[1], title='Trend')
decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
decomposition.resid.plot(ax=axes[3], title='Residual')
plt.tight_layout()
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
st.subheader("ACF and PACF")
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(df['Peak'].dropna(), ax=axes[0])
plot_pacf(df['Peak'].dropna(), ax=axes[1])
axes[0].set_title('Autocorrelation (ACF)')
axes[1].set_title('Partial Autocorrelation (PACF)')
st.pyplot(fig)

st.header("Steam Game Player Clustering")
st.write("The chart below shows the grouping of Steam game trends and visualizes them using PCA.")

# --- Section: Forecasting with Prophet ---
st.header("📈 Forecasting Future Player Trends (Prophet)")
st.write("This section provides a 12-month forecast of peak players using Facebook's Prophet model.")

from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")

# Prepare data for Prophet
df_prophet = games[selected_game][['Date', 'Peak']].rename(columns={'Date': 'ds', 'Peak': 'y'})
df_prophet = df_prophet.dropna()

if df_prophet.shape[0] >= 12:  # Ensure at least 12 months of data
    with st.spinner("Training Prophet model..."):
        model = Prophet(growth='linear', yearly_seasonality=True, seasonality_mode='multiplicative')
        model.fit(df_prophet)

        future = model.make_future_dataframe(periods=12, freq='M')
        forecast = model.predict(future)

        st.subheader("Forecast Chart (Next 12 Months)")
        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        st.subheader("Forecast Components")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

        st.success("Forecast complete.")
else:
    st.warning("Not enough data to train the forecasting model (need at least 12 months).")





# Feature extraction
features = []
game_names_initial = []

for name, df in games.items():
    df['Peak'] = pd.to_numeric(df['Peak'], errors='coerce')
    df['Gain'] = pd.to_numeric(df['Gain'], errors='coerce')
    df['%Gain'] = df['%Gain'].astype(str).str.replace('%', '').str.replace(',', '')
    df['%Gain'] = pd.to_numeric(df['%Gain'], errors='coerce')

    df_clean = df.dropna(subset=['Peak', 'Gain', '%Gain'])

    if df_clean.empty:
        continue

    features.append([
        df_clean['Peak'].mean(),
        df_clean['Peak'].max(),
        df_clean['Peak'].std(),
        df_clean['Gain'].mean(),
        df_clean['%Gain'].mean()
    ])
    game_names_initial.append(name)

feature_df = pd.DataFrame(features, columns=[
    'Avg Peak', 'Max Peak', 'Peak StdDev', 'Avg Gain', 'Avg %Gain'
], index=game_names_initial)

feature_df.dropna(inplace=True)
game_names = feature_df.index.tolist()

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
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')

for i, name in enumerate(game_names):
    ax.text(X_pca[i, 0] + 0.01, X_pca[i, 1], name, fontsize=9)

ax.set_title("Game Clusters (K-Means)")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.grid(True)
cbar = fig.colorbar(scatter, ax=ax, label='Cluster')

st.pyplot(fig)

with st.expander("Show Cluster Data Table"):
    st.dataframe(feature_df)
    
    
    
st.header("🔧 Prophet 参数网格搜索优化")

# 勾选触发优化功能
enable_grid_search = st.checkbox("启用 Prophet 参数网格搜索优化")

if enable_grid_search:
    st.info("将使用 GridSearch 对 Prophet 模型的参数进行优化。")

    if df_prophet.shape[0] < 24:
        st.warning("当前数据点少于 24，不适合进行交叉验证。")
    else:
        with st.spinner("正在进行网格搜索优化..."):
