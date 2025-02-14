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

monthly_trend_label = monthly_trend.copy()

# Mengubah nilai pada kolom 'yr'
monthly_trend_label['yr'] = monthly_trend_label['yr'].map({0: 'Tahun Pertama', 1: 'Tahun Kedua'})

# Mengubah nama kolom
monthly_trend_label = monthly_trend_label.rename(columns={'yr': 'Tahun', 'mnth': 'Bulan', 'cnt': 'Jumlah'})

# Menampilkan dataframe tanpa index
st.dataframe(monthly_trend_label, hide_index=True)

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

with st.expander("Hasil Analisis Tren Penyewaan Sepeda Berdasarkan Bulan"):
    st.write("""
    1. **Tren Umum**: Terlihat peningkatan jumlah penyewaan sepeda dari tahun pertama ke tahun kedua, menunjukkan pertumbuhan popularitas layanan sepeda.
    2. **Bulan Puncak**: Pada kedua tahun, penyewaan sepeda mencapai puncaknya di bulan-bulan pertengahan tahun (sekitar bulan 6 hingga 8), yang kemungkinan besar terkait dengan cuaca yang lebih hangat dan kondisi yang lebih nyaman untuk bersepeda.
    3. **Bulan Rendah**: Penyewaan sepeda cenderung lebih rendah di bulan-bulan awal dan akhir tahun (sekitar bulan 1 hingga 3 dan 10 hingga 12), yang mungkin dipengaruhi oleh cuaca yang lebih dingin dan kurang mendukung.

    **Kesimpulan:**  
    Tren penyewaan sepeda menunjukkan pola musiman yang jelas dengan peningkatan selama bulan-bulan musim panas dan penurunan selama bulan-bulan musim dingin. Pertumbuhan dari tahun pertama ke tahun kedua menunjukkan peningkatan minat dan penggunaan layanan sepeda, yang dapat dimanfaatkan untuk perencanaan dan promosi lebih lanjut.
    """)

# Pertanyaan 2: Pengaruh Waktu dan Cuaca terhadap Penyewaan Sepeda
st.header("Pengaruh Waktu dan Cuaca terhadap Penyewaan Sepeda")

# Menambahkan filter tahun multi-select
selected_years = st.multiselect("Pilih Tahun", options=[2011, 2012], default=[2011, 2012])

# Mapping tahun ke nilai 'yr' dalam dataset
year_mapping = {2011: 0, 2012: 1}
selected_yr_values = [year_mapping[year] for year in selected_years]

# Filter data berdasarkan tahun yang dipilih
filtered_data = df_cleaned[df_cleaned['yr'].isin(selected_yr_values)]

dataQuestion = df_cleaned
dataQuestion2 = filtered_data

if not dataQuestion2.empty:
    # Mapping nama hari
    filtered_data['weekday'] = filtered_data['weekday'].map({
        0: 'Senin', 1: 'Selasa', 2: 'Rabu', 3: 'Kamis',
        4: 'Jumat', 5: 'Sabtu', 6: 'Minggu'
    })

    # Mengelompokkan data berdasarkan hari dalam seminggu
    total_eachday = filtered_data.groupby('weekday')[['casual', 'registered']].sum().reset_index()

    # Urutkan berdasarkan urutan hari dalam seminggu
    order = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
    total_eachday = total_eachday.set_index('weekday').loc[order].reset_index();

    # Streamlit UI
    st.title("Analisis Penggunaan Sepeda")
    st.subheader("Data Pengguna Casual dan Registered per Hari")

    # Menampilkan tabel data
    total_eachday_display = total_eachday.rename(columns={'weekday': 'Hari', 'casual': 'Casual', 'registered': 'Registered'})
    st.dataframe(total_eachday_display, hide_index=True)

    # Visualisasi dengan grafik batang berkelompok
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(order))  # Lokasi weekday pada sumbu x
    width = 0.35  # Lebar batang

    ax.bar(x - width/2, total_eachday['casual'], width, label='Casual', color='blue')
    ax.bar(x + width/2, total_eachday['registered'], width, label='Registered', color='orange')

    # Menambahkan label, judul, dan legenda
    ax.set_title("Jumlah Pengguna Casual dan Registered per Hari")
    ax.set_xlabel("Hari")
    ax.set_ylabel("Jumlah Pengguna")
    ax.set_xticks(x)
    ax.set_xticklabels(total_eachday['weekday'])
    ax.legend()

    # Menampilkan grafik di Streamlit
    st.pyplot(fig)

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


    dataQuestion2['rentang_waktu'] = dataQuestion2['hr'].apply(time_of_day)
    dataQuestion2['weather_category'] = dataQuestion2['weathersit'].apply(weather_of_day)

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

    # Visualisasi berdasarkan waktu
    st.subheader("Jumlah Penyewaan Sepeda Berdasarkan Rentang Waktu")
    dataEachTime = dataQuestion2.groupby('rentang_waktu')['cnt'].sum().reset_index()

    time_rental_data = (
        dataQuestion2.groupby(["yr", "rentang_waktu"])["cnt"]
        .sum()
        .reset_index()
        .replace({"yr": {0: 2011, 1: 2012}})
    )
    time_rental_data = time_rental_data.rename(columns={'yr': 'Tahun', 'rentang_waktu': 'Waktu', 'cnt': 'Jumlah'})

    st.dataframe(time_rental_data.style.format({"Tahun": "{:d}"}), hide_index=True)

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

    ax[1].pie(dataEachTime['cnt'], labels=dataEachTime['rentang_waktu'], autopct='%1.2f%%',
              colors=['blue', 'green', 'orange', 'red'])
    ax[1].set_title("Pie Chart Proporsi Penyewaan Sepeda Berdasarkan Rentang Waktu")

    st.pyplot(fig)

    # Analisis dinamis berdasarkan tahun yang dipilih
    with st.expander("Hasil Analisis Penyewaan Sepeda Berdasarkan Rentang Waktu"):
        st.write(
            f"Berdasarkan perbandingan dan proporsi pada chart di atas menunjukkan bahwa rentang waktu sangat mempengaruhi jumlah penyewaan sepeda pada tahun {', '.join(map(str, selected_years))}.")

        # Menghitung total penyewaan
        total_rentals_time = dataEachTime['cnt'].sum()

        # Menampilkan hasil analisis dinamis
        for index, row in dataEachTime.iterrows():
            percentage = (row['cnt'] / total_rentals_time) * 100
            st.write(
                f"Jumlah penyewaan sepeda tertinggi terjadi pada {row['rentang_waktu']} hari, dengan jumlah penyewaan mencapai {row['cnt']:,} atau sekitar {percentage:.2f}% dari total penyewaan.")

        st.write(
            "Kesimpulannya, waktu siang adalah waktu paling populer untuk penyewaan sepeda, sedangkan sore hari adalah waktu dengan jumlah penyewaan paling sedikit.")

    # Visualisasi berdasarkan cuaca
    st.subheader("Jumlah Penyewaan Sepeda Berdasarkan Kondisi Cuaca")

    dataEachWeather = dataQuestion2.groupby('weather_category')['cnt'].sum().reset_index()
    total_rentals = dataEachWeather['cnt'].sum()
    dataEachWeather['proportion'] = dataEachWeather['cnt'] / total_rentals

    dataEachWeatherLabel = dataEachWeather
    dataEachWeatherLabel = dataEachWeatherLabel.rename(columns={'weather_category': 'Kategori Cuaca', 'cnt': 'Jumlah', 'proportion': 'Proporsi'})
    dataEachWeatherLabel['Proporsi'] = dataEachWeatherLabel['Proporsi'] * 100
    st.dataframe(dataEachWeatherLabel, hide_index=True)

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

    ax[1].pie(dataEachWeather['proportion'], labels=dataEachWeather['weather_category'], autopct='%1.2f%%',
              colors=['blue', 'green', 'orange', 'red'])
    ax[1].set_title("Pie Chart Proporsi Penyewaan Sepeda Berdasarkan Kondisi Cuaca")

    st.pyplot(fig)

    # Analisis dinamis berdasarkan tahun yang dipilih
    with st.expander("Hasil Analisis Penyewaan Sepeda Berdasarkan Kondisi Cuaca"):
        st.write(
            f"Berdasarkan data pada tahun {', '.join(map(str, selected_years))}, berikut adalah hasil analisis pengaruh kondisi cuaca terhadap penyewaan sepeda:")

        for index, row in dataEachWeather.iterrows():
            percentage = (row['cnt'] / total_rentals) * 100
            st.write(
                f"- **{row['weather_category']}** memiliki jumlah penyewaan sepeda sebanyak **{row['cnt']:,} kali penyewaan**, dengan persentase sebesar **{percentage:.2f}%**.")

        st.write("""
        **Kesimpulan:**  
        Jumlah penyewaan sepeda sangat dipengaruhi oleh kondisi cuaca. Pengguna lebih memilih menyewa sepeda saat cuaca cerah atau sedikit berawan, sementara cuaca ekstrem dan hujan menyebabkan penurunan signifikan dalam jumlah penyewaan.
        """)
else:
    st.warning("Beberapa data tidak tersedia untuk filter tahun yang Anda pilih")

# Analisis korelasi
st.header("Analisis Korelasi Faktor Penyewaan Sepeda")

# Menyiapkan data untuk analisis korelasi
correlation_data = dataQuestion[['temp', 'hum', 'windspeed', 'cnt']]
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

# Pertanyaan 1: Pengaruh Windspeed terhadap Peminjaman Sepeda
st.header("Pengaruh Windspeed terhadap Peminjaman Sepeda")
collectByWindSpeed = day_data.groupby('windspeed')['cnt'].sum().reset_index()

fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.scatter(collectByWindSpeed['windspeed'], collectByWindSpeed['cnt'], alpha=0.5, color='orange')
ax1.set_title('Hubungan antara Windspeed dan Jumlah Peminjaman Sepeda')
ax1.set_xlabel('Kecepatan Angin (Windspeed)')
ax1.set_ylabel('Jumlah Peminjaman Sepeda (cnt)')
ax1.grid()
st.pyplot(fig1)  # Menampilkan plot pertama

# Mengelompokkan Windspeed menjadi Kategori
collectByWindSpeed['windspeed_range'] = pd.cut(
    collectByWindSpeed['windspeed'],
    bins=[0, 0.1, 0.2, 0.3, 0.4, 1.0],
    labels=['Sangat Rendah', 'Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi']
)

# Rata-rata jumlah peminjaman berdasarkan kategori Windspeed
grouped = collectByWindSpeed.groupby('windspeed_range', observed=False)['cnt'].mean()

# Bar Plot: Rata-rata Peminjaman Berdasarkan Kategori Windspeed
fig2, ax2 = plt.subplots(figsize=(8, 5))
grouped.plot(kind='bar', ax=ax2, color='skyblue')
ax2.set_title('Rata-rata Peminjaman Sepeda berdasarkan Windspeed')
ax2.set_xlabel('Kategori Windspeed')
ax2.set_ylabel('Rata-rata Jumlah Peminjaman Sepeda')
ax2.set_xticklabels(grouped.index, rotation=45)
ax2.grid(axis='y')
fig2.tight_layout()  # Menghindari tampilan yang terpotong
st.pyplot(fig2)

with st.expander("Hasil Analisis Pengaruh Windspeed Terhadap Peminjaman Sepeda"):
    st.write("""
    Statistik deskriptif menunjukkan bahwa kolom windspeed memiliki nilai rata-rata tertentu dengan variasi data yang signifikan. Hal ini menunjukkan bahwa kecepatan angin bervariasi dari sangat rendah hingga tinggi dalam dataset.
    1. Kecepatan angin (windspeed) memiliki hubungan negatif lemah dengan jumlah peminjaman sepeda. Angin kencang memang mengurangi jumlah peminjaman, tetapi tidak secara signifikan.
    2. Pada kecepatan angin rendah, jumlah peminjaman lebih tinggi, sedangkan angin yang lebih kuat mengurangi aktivitas peminjaman.
    3. Meskipun windspeed memiliki dampak kecil, faktor lain seperti suhu (temp), kelembapan (hum), dan kondisi cuaca (weathersit) juga berperan dalam memengaruhi jumlah peminjaman sepeda.
    """)

# Pertanyaan 3: Perbedaan Penyewaan Sepeda antara Hari Libur dan Hari Kerja
st.header("Perbedaan Penyewaan Sepeda antara Hari Libur dan Hari Kerja")

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

with st.expander("Hasil Analisis Rata Rata Penyewaan Sepeda Berdasarkan Hari Libur dan Kerja"):
    st.write("""
    Statistik penyewaan sepeda secara keseluruhan menunjukkan perbedaan yang jelas antara hari libur (termasuk akhir pekan) dan hari kerja. Akhir pekan/liburan memiliki rata-rata penyewaan yang lebih tinggi, yang mengindikasikan peningkatan penggunaan untuk bersantai, sementara hari kerja menunjukkan rata-rata penyewaan yang lebih rendah, yang mungkin mencerminkan lebih banyak penggunaan yang bersifat fungsional atau yang berhubungan dengan perjalanan. Wawasan ini sangat berharga bagi bisnis penyewaan sepeda untuk memahami perilaku pelanggan, menyesuaikan strategi penetapan harga, dan merencanakan kampanye promosi untuk menargetkan segmen pengguna yang berbeda secara efektif sepanjang minggu.
    """)

# Pertanyaan 4: Musim dengan Potensi Terbesar untuk Promosi Layanan Sepeda
st.header("Musim dengan Potensi Terbesar untuk Promosi Layanan Sepeda")

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

with st.expander("Hasil Analisis Rata-Rata Penyewaan Sepeda Permusim"):
    st.write("""
    **Grafik Batang**: Rata-rata Jumlah Peminjaman per Musim
    Penjelasan: Grafik ini menunjukkan rata-rata peminjaman sepeda untuk setiap musim: Musim Gugur (Fall) memiliki jumlah peminjaman tertinggi. Musim Semi (Spring) memiliki jumlah peminjaman terendah. Jawaban: Musim gugur adalah musim dengan potensi terbesar untuk promosi layanan sepeda karena penggunaannya paling tinggi. Kampanye promosi dapat diperkuat di musim ini untuk memaksimalkan keuntungan.
    
    **Grafik Garis**: Perbandingan Pengguna Kasual vs Terdaftar per Musim
    Penjelasan: Grafik ini membandingkan rata-rata peminjaman oleh pengguna kasual dan terdaftar: Pengguna terdaftar mendominasi di semua musim. Pengguna kasual meningkat signifikan di musim panas dan musim gugur, menunjukkan peluang untuk promosi musiman yang lebih santai. Jawaban: Promosi dapat difokuskan pada pengguna kasual selama musim panas dan gugur karena segmen ini lebih responsif terhadap kondisi cuaca yang mendukung.""")
