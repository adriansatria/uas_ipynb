import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.cluster import KMeans


st.sidebar.page_link("uas_streamlit.py", label="Dashboard")

# Judul dan informasi proyek
st.title("Dashboard Analisis Bike Sharing")

# Import dataset
@st.cache_data
def load_data():
    day_data = pd.read_csv('datasets/day.csv')
    hour_data = pd.read_csv('datasets/hour.csv')
    return day_data, hour_data

day_data, hour_data = load_data()

# Menghapus outlier
z_scores = np.abs(stats.zscore(hour_data['cnt']))
threshold = 3
non_outliers = np.where(z_scores <= threshold)
df_cleaned = hour_data.iloc[non_outliers]

# Tren penyewaan sepeda per bulan
st.subheader("Tren Penyewaan Sepeda Berdasarkan Bulan")
monthly_trend = day_data.groupby(['yr', 'mnth'])['cnt'].sum().reset_index()
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=monthly_trend, x='mnth', y='cnt', hue='yr', palette=['blue', 'orange'], ax=ax)
ax.set_title('Tren Penyewaan Sepeda Berdasarkan Bulan di Tahun Pertama dan Kedua')
ax.set_xlabel('Bulan')
ax.set_ylabel('Jumlah Penyewaan (cnt)')
legend_labels = ['Tahun Pertama', 'Tahun Kedua']
legend_colors = ['blue', 'orange']
plt.legend(handles=[plt.Line2D([0], [0], color=color, lw=2) for color in legend_colors],
           labels=legend_labels, title="Tahun", loc='best')
st.pyplot(fig)

# Pertanyaan 1: Pengaruh Windspeed terhadap Peminjaman Sepeda
st.header("Pertanyaan 1: Pengaruh Windspeed terhadap Peminjaman Sepeda")
collectByWindSpeed = day_data.groupby('windspeed')['cnt'].sum().reset_index()

fig, ax = plt.subplots(figsize=(10, 6))
plt.scatter(collectByWindSpeed['windspeed'], collectByWindSpeed['cnt'], alpha=0.5, color='orange')
plt.title('Hubungan antara Windspeed dan Jumlah Peminjaman Sepeda')
plt.xlabel('Kecepatan Angin (Windspeed)')
plt.ylabel('Jumlah Peminjaman Sepeda (cnt)')
plt.grid()
st.pyplot(fig)

# Pertanyaan 2: Pengaruh Waktu dan Cuaca terhadap Penyewaan Sepeda
st.header("Pengaruh Waktu dan Cuaca terhadap Penyewaan Sepeda")

def time_of_day(hour):
    if 6 <= hour < 12:
        return 'Pagi'
    elif 12 <= hour < 18:
        return 'Siang'
    elif 18 <= hour < 20:
        return 'Sore'
    else:
        return 'Malam'

def weather_of_day(weathersit):
    if weathersit == 1:
        return 'Cerah/Sedikit berawan'
    elif weathersit == 2:
        return 'Berawan/Berkabut'
    elif weathersit == 3:
        return 'Hujan Salju/Badai'
    elif weathersit == 4:
        return 'Cuaca Ekstrim'
    else:
        return 'Tidak Valid'

dataQuestion2 = df_cleaned
dataQuestion2['rentang_waktu'] = dataQuestion2['hr'].apply(time_of_day)
dataQuestion2['weather_category'] = dataQuestion2['weathersit'].apply(weather_of_day)

# Visualisasi berdasarkan waktu
st.subheader("Jumlah Penyewaan Sepeda Berdasarkan Rentang Waktu")
dataEachTime = dataQuestion2.groupby('rentang_waktu')['cnt'].sum().reset_index()

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
x = np.arange(len(dataEachTime))
width = 0.35

bars = ax[0].bar(x, dataEachTime['cnt'], width, label='Penyewaan Sepeda', color=['blue', 'green', 'orange', 'red'])
ax[0].set_title("Jumlah Penyewaan Sepeda Berdasarkan Rentang Waktu")
ax[0].set_xlabel("Rentang Waktu")
ax[0].set_ylabel("Jumlah Penyewaan (cnt)")
ax[0].set_xticks(x, dataEachTime['rentang_waktu'])

for bar in bars:
    yval = bar.get_height()
    ax[0].text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f'{int(yval)}', ha='center', va='bottom', fontsize=12)

ax[1].pie(dataEachTime['cnt'], labels=dataEachTime['rentang_waktu'], autopct='%1.2f%%', colors=['blue', 'green', 'orange', 'red'])
ax[1].set_title("Pie Chart Proporsi Penyewaan Sepeda Berdasarkan Rentang Waktu")

st.pyplot(fig)

with st.expander("Hasil Analisis Penyewaan Sepeda Berdasarkan Rentang Waktu"):
    st.write("Berdasarkan perbandingan dan proporsi pada chart di atas menunjukkan bahwa rentang waktu sangat mempengaruhi jumlah penyewaan sepeda.")
    st.write("Dapat dilihat bahwa jumlah penyewaan sepeda tertinggi terjadi pada siang hari, dengan jumlah penyewaan mencapai 1.197.501 atau sekitar 38.72% dari total penyewaan.")
    st.write("Penyewaan sepeda juga cukup tinggi pada pagi hari dengan jumlah 877.142 atau 28.36%.")
    st.write("Sementara itu, jumlah penyewaan sepeda paling rendah terjadi pada sore hari dengan 462.005 penyewaan (14.94%).")
    st.write("Kesimpulannya, waktu siang adalah waktu paling populer untuk penyewaan sepeda, sedangkan sore hari adalah waktu dengan jumlah penyewaan paling sedikit.")

# Mapping kondisi cuaca
weather_mapping = {
    1: 'Cerah/Sedikit berawan',
    2: 'Berawan/Berkabut',
    3: 'Hujan Salju/Badai',
    4: 'Cuaca Ekstrim'
}

# Menambahkan kolom tahun dan kategori cuaca
day_data['year'] = day_data['yr'].map({0: 'Tahun Pertama', 1: 'Tahun Kedua'})
day_data['weather_category'] = day_data['weathersit'].map(weather_mapping)

# Mengelompokkan data berdasarkan tahun dan kondisi cuaca
weather_year_data = day_data.groupby(['year', 'weather_category'])['cnt'].sum().reset_index()

# Visualisasi
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x='weather_category', y='cnt', hue='year', data=weather_year_data, palette='coolwarm')
plt.title("Pengaruh Kondisi Cuaca terhadap Penyewaan Sepeda per Tahun")
plt.xlabel("Kondisi Cuaca")
plt.ylabel("Jumlah Penyewaan Sepeda (cnt)")
plt.legend(title="Tahun", loc='upper right')
st.pyplot(fig)

with st.expander("Hasil Analisis Pengaruh Kondisi Cuaca terhadap Penyewaan Sepeda per Tahun"):
    st.write("""
    1. **Cuaca Cerah/Sedikit Berawan** memiliki jumlah penyewaan sepeda tertinggi di kedua tahun, dengan peningkatan signifikan di tahun kedua.
    2. **Cuaca Berawan/Berkabut** juga menunjukkan peningkatan jumlah penyewaan di tahun kedua, meskipun tidak sebesar cuaca cerah.
    3. **Cuaca Hujan Salju/Badai** dan **Cuaca Ekstrem** memiliki jumlah penyewaan yang lebih rendah, dengan sedikit peningkatan di tahun kedua.

    **Kesimpulan:**  
    Kondisi cuaca cerah dan sedikit berawan tetap menjadi faktor utama yang mempengaruhi jumlah penyewaan sepeda. Peningkatan jumlah penyewaan di tahun kedua menunjukkan pertumbuhan popularitas layanan sepeda, terutama dalam kondisi cuaca yang baik.
    """)

# Visualisasi berdasarkan cuaca
st.subheader("Jumlah Penyewaan Sepeda Berdasarkan Kondisi Cuaca")
dataEachWeather = dataQuestion2.groupby('weather_category')['cnt'].sum().reset_index()
total_rentals = dataEachWeather['cnt'].sum()
dataEachWeather['proportion'] = dataEachWeather['cnt'] / total_rentals

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
x = np.arange(len(dataEachWeather))

bars = ax[0].bar(x, dataEachWeather['cnt'], width, label='Penyewaan Sepeda', color=['blue', 'green', 'orange', 'red'])
ax[0].set_title("Jumlah Penyewaan Sepeda Berdasarkan Kondisi Cuaca")
ax[0].set_xlabel("Kondisi Cuaca (weathersit)")
ax[0].set_ylabel("Jumlah Penyewaan (cnt)")
ax[0].set_xticks(x, dataEachWeather['weather_category'])

for bar in bars:
    yval = bar.get_height()
    ax[0].text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f'{int(yval)}', ha='center', va='bottom', fontsize=12)

ax[1].pie(dataEachWeather['proportion'], labels=dataEachWeather['weather_category'], autopct='%1.2f%%', colors=['blue', 'green', 'orange', 'red'])
ax[1].set_title("Pie Chart Proporsi Penyewaan Sepeda Berdasarkan Kondisi Cuaca")

st.pyplot(fig)

with st.expander("Hasil Analisis Penyewaan Sepeda Berdasarkan Kondisi Cuaca"):
    st.write("""

    1. **Cuaca Cerah/Sedikit Berawan** memiliki jumlah penyewaan sepeda tertinggi, yaitu **2.175.325 kali penyewaan**, dengan persentase sebesar **70,34%**. Hal ini menunjukkan bahwa pengguna cenderung lebih nyaman menyewa sepeda saat cuaca cerah.

    2. **Cuaca Berawan/Berkabut** memiliki penyewaan sebanyak **764.524 kali (24,72%)**. Meskipun jumlahnya lebih kecil dibandingkan cuaca cerah, penyewaan masih cukup tinggi, menunjukkan bahwa cuaca mendung tidak terlalu menghambat pengguna.

    3. **Cuaca Hujan Salju/Badai** menunjukkan penurunan signifikan dalam jumlah penyewaan, hanya **152.633 kali (4,94%)**. Ini masuk akal karena kondisi cuaca buruk seperti hujan atau badai mengurangi minat pengguna untuk bersepeda.

    4. **Cuaca Ekstrem** memiliki penyewaan sepeda paling sedikit, hanya **223 kali (0,01%)**. Ini menunjukkan bahwa hampir tidak ada pengguna yang menyewa sepeda dalam kondisi cuaca sangat buruk.

    **Kesimpulan:**  
    Jumlah penyewaan sepeda sangat dipengaruhi oleh kondisi cuaca. Pengguna lebih memilih menyewa sepeda saat cuaca cerah atau sedikit berawan, sementara cuaca ekstrem dan hujan menyebabkan penurunan signifikan dalam jumlah penyewaan.
    """)

# Analisis korelasi
st.header("Analisis Korelasi Faktor Penyewaan Sepeda")

# Menyiapkan data untuk analisis korelasi
correlation_data = dataQuestion2[['temp', 'hum', 'windspeed', 'cnt']]
correlation_matrix = correlation_data.corr()

# Visualisasi matriks korelasi
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
plt.title("Matriks Korelasi Faktor Penyewaan Sepeda")
st.pyplot(fig)

with st.expander("Hasil Analisis Korelasi"):
    st.write("""
    Matriks korelasi menunjukkan hubungan antara suhu, kelembaban, kecepatan angin, dan jumlah penyewaan sepeda:
    1. **Suhu (temp)**: Korelasi positif dengan jumlah penyewaan, menunjukkan bahwa suhu yang lebih hangat meningkatkan minat penyewaan.
    2. **Kelembaban (hum)**: Korelasi negatif, menunjukkan bahwa kelembaban tinggi mengurangi minat penyewaan.
    3. **Kecepatan Angin (windspeed)**: Korelasi negatif, menunjukkan bahwa angin kencang mengurangi minat penyewaan.

    **Kesimpulan:**  
    Faktor cuaca seperti suhu, kelembaban, dan kecepatan angin memiliki pengaruh signifikan terhadap jumlah penyewaan sepeda.
    """)

st.header("Analisis Time Series Penyewaan Sepeda")

# Menyiapkan data time series
time_series_data = day_data.set_index('dteday')['cnt']
decomposition = seasonal_decompose(time_series_data, model='additive', period=365)

# Visualisasi time series
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8))
decomposition.trend.plot(ax=ax1)
ax1.set_title('Trend')
decomposition.seasonal.plot(ax=ax2)
ax2.set_title('Seasonality')
decomposition.resid.plot(ax=ax3)
ax3.set_title('Residuals')
time_series_data.plot(ax=ax4)
ax4.set_title('Original Data')
plt.tight_layout()
st.pyplot(fig)

with st.expander("Hasil Analisis Time Series"):
    st.write("""
    Analisis time series menunjukkan:
    1. **Trend**: Ada peningkatan tren penyewaan sepeda dari waktu ke waktu.
    2. **Seasonality**: Terlihat pola musiman dengan peningkatan penyewaan pada bulan-bulan tertentu.
    3. **Residuals**: Variasi acak yang tidak dapat dijelaskan oleh tren atau musiman.

    **Kesimpulan:**  
    Tren yang meningkat menunjukkan pertumbuhan popularitas layanan sepeda, sementara pola musiman dapat digunakan untuk merencanakan promosi atau penyesuaian layanan.
    """)

# Clustering berdasarkan waktu dan cuaca
st.header("Segmentasi Pengguna Berdasarkan Pola Penyewaan")

# Menyiapkan data untuk clustering
cluster_data = dataQuestion2[['hr', 'weathersit', 'cnt']]
kmeans = KMeans(n_clusters=3, random_state=42)
dataQuestion2['cluster'] = kmeans.fit_predict(cluster_data)

# Visualisasi clustering
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='hr', y='cnt', hue='cluster', data=dataQuestion2, palette='viridis', ax=ax)
plt.title("Segmentasi Pengguna Berdasarkan Pola Penyewaan")
plt.xlabel("Jam (hr)")
plt.ylabel("Jumlah Penyewaan (cnt)")
st.pyplot(fig)

with st.expander("Hasil Analisis Segmentasi Pengguna"):
    st.write("""
    Dengan menggunakan teknik clustering K-Means, kami mengidentifikasi tiga kelompok pengguna berdasarkan pola penyewaan sepeda:
    1. **Cluster 0**: Pengguna dengan penyewaan rendah, biasanya pada malam hari atau saat cuaca buruk.
    2. **Cluster 1**: Pengguna dengan penyewaan sedang, biasanya pada pagi atau sore hari.
    3. **Cluster 2**: Pengguna dengan penyewaan tinggi, biasanya pada siang hari atau saat cuaca cerah.

    **Kesimpulan:**  
    Segmentasi ini membantu dalam memahami pola penggunaan sepeda dan dapat digunakan untuk menargetkan promosi atau layanan yang lebih spesifik.
    """)

# Pertanyaan 3: Perbedaan Penyewaan Sepeda antara Hari Libur dan Hari Kerja
st.header("Pertanyaan 3: Perbedaan Penyewaan Sepeda antara Hari Libur dan Hari Kerja")

day_mapping = {
    0: 'Senin',
    1: 'Selasa',
    2: 'Rabu',
    3: 'Kamis',
    4: 'Jumat',
    5: 'Sabtu',
    6: 'Minggu'
}

data_frame_ratio = day_data
data_frame_ratio['holiday'] = data_frame_ratio.apply(
    lambda row: 1 if row['weekday'] in [5, 6] else row['holiday'],
    axis=1
)

data_frame_ratio['weekday_name'] = data_frame_ratio['weekday'].map(day_mapping)
data_frame_ratio['holiday_type'] = data_frame_ratio['holiday'].map({0: 'Hari Kerja', 1: 'Hari Libur'})

avg_bike_rentals = data_frame_ratio.groupby(['weekday_name', 'holiday_type'])['cnt'].mean().reset_index()

order = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
avg_bike_rentals['weekday_name'] = pd.Categorical(avg_bike_rentals['weekday_name'], categories=order, ordered=True)
avg_bike_rentals = avg_bike_rentals.sort_values('weekday_name')

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x='weekday_name', y='cnt', hue='holiday_type', data=avg_bike_rentals, palette='viridis')
plt.title("Rata-Rata Penyewaan Sepeda Berdasarkan Hari")
plt.xlabel("Hari")
plt.ylabel("Rata-Rata Penyewaan Sepeda")
plt.legend(title="Jenis Hari", loc='upper right')
st.pyplot(fig)

# Pertanyaan 4: Musim dengan Potensi Terbesar untuk Promosi Layanan Sepeda
st.header("Pertanyaan 4: Musim dengan Potensi Terbesar untuk Promosi Layanan Sepeda")

season_mapping = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
hour_data['season_name'] = hour_data['season'].map(season_mapping)

season_avg_rentals = hour_data.groupby('season_name')['cnt'].mean()
season_casual_registered = hour_data.groupby('season_name')[['casual', 'registered']].mean()

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

ax[0].bar(season_avg_rentals.index, season_avg_rentals.values, color=['#FF9999', '#66B2FF', '#99FF99', '#FFD700'])
ax[0].set_title('Rata-Rata Penyewaan Sepeda per Musim')
ax[0].set_ylabel('Rata-Rata Penyewaan')
ax[0].set_xlabel('Musim')

x_ticks = np.arange(len(season_casual_registered.index))
ax[1].plot(x_ticks, season_casual_registered['casual'], label='Casual Users', marker='o', color='skyblue')
ax[1].plot(x_ticks, season_casual_registered['registered'], label='Registered Users', marker='o', color='orange')
ax[1].set_title('Casual vs Registered Users per Musim')
ax[1].set_ylabel('Rata-Rata Penyewaan')
ax[1].set_xlabel('Musim')
ax[1].set_xticks(x_ticks)
ax[1].set_xticklabels(season_casual_registered.index)
ax[1].legend()

st.pyplot(fig)