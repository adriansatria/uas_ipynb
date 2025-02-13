import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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

with st.expander("Hasil Analisis"):
    st.write("Berdasarkan perbandingan dan proporsi pada chart di atas menunjukkan bahwa rentang waktu sangat mempengaruhi jumlah penyewaan sepeda.")
    st.write("Dapat dilihat bahwa jumlah penyewaan sepeda tertinggi terjadi pada siang hari, dengan jumlah penyewaan mencapai 1.197.501 atau sekitar 38.72% dari total penyewaan.")
    st.write("Penyewaan sepeda juga cukup tinggi pada pagi hari dengan jumlah 877.142 atau 28.36%.")
    st.write("Sementara itu, jumlah penyewaan sepeda paling rendah terjadi pada sore hari dengan 462.005 penyewaan (14.94%).")
    st.write("Kesimpulannya, waktu siang adalah waktu paling populer untuk penyewaan sepeda, sedangkan sore hari adalah waktu dengan jumlah penyewaan paling sedikit.")

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