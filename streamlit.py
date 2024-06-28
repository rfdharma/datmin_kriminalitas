import streamlit as st
import pandas as pd
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import kstest
import statsmodels.formula.api as smf
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_option('deprecation.showPyplotGlobalUse', False)

df = pd.read_excel('data_gemastik_kriminalitas.xlsx')
data = pd.read_csv('Indo_34_20146762.csv')
data = data.drop(data.loc[data['Kepolisian/Resort'] == 'Jawa Timur'].index)
see = df.copy()

le = LabelEncoder()
df['daerah_encoded'] = le.fit_transform(df['daerah'])

# Simpan mapping invers dari LabelEncoder
sorted_classes = sorted(le.classes_)
daerah_mapping = dict(zip(range(len(sorted_classes)), sorted_classes))

def inverse_transform_daerah(encoded_value):
    return daerah_mapping[encoded_value]

columns_to_transform = ['indeks_pembangunan_manusia', 'tingkat_pengangguran_terbuka', 'jumlah_tindak_pidana', 'jumlah_penduduk_miskin']

# Membuat objek PowerTransformer
pt = PowerTransformer(method='yeo-johnson')

df[columns_to_transform] = pt.fit_transform(df[columns_to_transform])

def run_kstest(column):
    stat, p = kstest(df[column], 'norm')
    return stat, p

def home():
    # Vertically align the content
    st.markdown(
        "<div style='display: flex; align-items: center; justify-content: center; flex-direction: column;'>"
        "<h1 style='text-align: center;'>⚜️ Tim Timan</h1>"
        "</div>",
        unsafe_allow_html=True
    )

    st.image('https://asset.kompas.com/crops/npn8tIiHvIzitpFhLMNi1Is6avw=/2x0:737x490/750x500/data/photo/2022/11/01/6360bd4112fe1.jpg')
    st.markdown('***')

    st.markdown(
        "<div style='display: flex; align-items: center; justify-content: center; flex-direction: column;'>"
        "<h3>Anggota :</h3>"
        "<h5>Denis Muhammad Jethro</p>"
        "<h5>R Firdaus Dharmawan Akbar</p>"
        "<h5>Asfa Lazuardi Wicaksono</p>"
        "</div>",
        unsafe_allow_html=True
    )

def eda():
    st.title("📊 Exploratory Data Analysis")
    st.markdown("***")
    st.image('wordcloud.png')
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("🔍 See explanation"):
            st.write("""
            Wordcloud yang muncul dari media Suara Surabaya menggambarkan fokus pemberitaan yang dominan pada peristiwa kriminalitas di Surabaya. Kata-kata seperti Polrestabes Surabaya, kejadian, laporan, pelapor, korban, tersangka, dan operasi Sikat menonjol sebagai representasi utama dalam berita mereka. Hal ini menunjukkan bahwa media Suara Surabaya secara konsisten memberitakan berbagai kejadian kriminal serta respons kepolisian dan masyarakat terhadapnya di kota Surabaya.
                """)
    
    with col2:
        with st.expander("❓ Why do we use this?"):
            st.write("""
Kami menggunakan wordcloud seperti ini untuk secara visual mewakili frekuensi kata-kata dalam sebuah teks atau kumpulan teks. Hal ini membantu kami dengan cepat memahami tema atau topik yang paling dominan dalam konten tersebut, sehingga lebih mudah mengidentifikasi pola, tren, atau area fokus utama yang diminati.
""")
    import plotly.express as px

    topics = ['Surabaya Student Bootcamp 2024', 'Surabaya Vehicle Theft Arrests', 
            'Satgas Pemberantasan Judi Online', 'Haji Pilgrims Activities Updates', 
            'Suara Surabaya Blood Donation', 'Jazz Traffic Festival 2024', 
            'Hendy Setiono & Surabaya News']
    counts = [46, 84, 63, 34, 30, 24, 19]

    # Sorting data
    sorted_data = sorted(zip(topics, counts), key=lambda x: x[1], reverse=True)
    sorted_topics = [x[0] for x in sorted_data]
    sorted_counts = [x[1] for x in sorted_data]

    # Create a dataframe for Plotly
    topic_count = pd.DataFrame({'Topic': sorted_topics, 'Count': sorted_counts})

    # Plot using Plotly
    fig = px.bar(topic_count, x='Count', y='Topic', orientation='h', color='Topic',
                title='Topic Counts (Highest to Lowest)',
                labels={'Count': 'Count', 'Topic': 'Topic'},
                color_discrete_sequence=['skyblue'] * len(topic_count))

    # Highlighting the highest bar in red
    fig.data[0].marker.color = ['red' if topic == sorted_topics[0] else 'skyblue' for topic in sorted_topics]

    st.plotly_chart(fig)

    st.image('topic_model.png')
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("🔍 See explanation"):
            st.write("""
            Kami melakukan riset pada SuaraSurabaya (media Berita kota Surabaya) dengan mengambil data secara acak didapatkan sebanyak 300 data. Dari 300 data tersebut dilakukan topic modeling sehingga menghasilkan topik pada visualisasi bar plot di atas. Ditemukan bahwa topik yang paling sering muncul adalah topik mengenai Penangkapan Pencurian Kendaraan Bermotor di Surabaya berdasarkan 300 berita tersebut. 
                """)
    
    with col2:
        with st.expander("❓ Why do we use this?"):
            st.write("""
Kami menggunakan topic modeling untuk menganalisis data dari SuaraSurabaya karena dapat membantu mengidentifikasi dan memahami topik atau isu yang paling sering muncul dalam berita mereka. Hal ini berguna untuk menyoroti tren berita yang relevan dan penting, seperti penangkapan pencurian kendaraan bermotor di Surabaya, yang memberikan wawasan yang berharga bagi kepentingan riset dan pemahaman mendalam terhadap peristiwa-peristiwa kriminalitas yang terjadi di kota tersebut.
""")

    st.write('---')
    st.header("Data Predictive")
    st.write('10 rows of the dataset')
    st.write(see.tail(10))

    with st.expander("💡 Variables explanation"):
        st.write("""
        1. daerah: Nama daerah atau kabupaten/kota di Jawa Timur.

2. tahun: Tahun pengukuran data.

3. indeks_pembangunan_manusia: Indeks yang mengukur tingkat pembangunan manusia di suatu daerah, mencakup pendidikan, kesehatan, dan pendapatan.

4. tingkat_pengangguran_terbuka: Persentase dari angkatan kerja yang tidak memiliki pekerjaan tetapi aktif mencari pekerjaan.
5. jumlah_tindak_pidana: Jumlah kejahatan yang dilaporkan dalam suatu daerah.
                 
6. jumlah_penyelasaian: Jumlah kasus kejahatan yang berhasil diselesaikan atau ditindaklanjuti.

7. jumlah_penduduk_miskin: Estimasi jumlah penduduk miskin dalam ribuan jiwa.

8. tingkat_pendidikan: Persentase penduduk yang telah menyelesaikan pendidikan formal tertentu, seperti tingkat SD, SMP, SMA, atau perguruan tinggi.
                 
9. selang_waktu: Indeks atau angka tambahan yang mungkin menggambarkan periode tertentu atau rentang waktu yang diukur dalam satuan tertentu
            """)
        
    ##### DESC STATS #####
    st.text("")
    st.markdown(
        "<br>"
        "<h5>Descriptive Statistics</h5>",  
        unsafe_allow_html=True
    )
    st.write(see.describe())
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("🔍 See explanation"):
            st.write("""
            Rata-rata indeks pembangunan manusia sebesar 70.37 menunjukkan tingkat pembangunan yang relatif tinggi, dengan tingkat pengangguran terbuka rata-rata 3.71% menunjukkan stabilitas pasar tenaga kerja yang cukup baik. Jumlah tindak pidana memiliki nilai rata-rata 931, menunjukkan adanya tantangan keamanan, meskipun jumlah penyelesaian yang rata-rata lebih rendah (701.75) menunjukkan upaya penegakan hukum yang kurang efektif. Jumlah penduduk miskin memiliki nilai rata-rata 119.76, yang menunjukkan masalah signifikan dalam kesenjangan ekonomi. Tingkat pendidikan rata-rata 0.62 menunjukkan akses yang cukup baik ke pendidikan dasar, sementara selang waktu rata-rata 655.23 mencerminkan periode waktu tertentu yang diukur dalam dataset ini.
            """)
    
    with col2:
        with st.expander("❓ Why do we use this?"):
            st.write("""
            * Statistika deskriptif diperlukan untuk dataset ini agar kita dapat memahami dan menggambarkan secara ringkas karakteristik-karakteristik utama dari data kesehatan pegawai yang disurvei. 
            * Informasi ini berguna untuk memberikan pemahaman awal tentang profil kesehatan populasi pegawai tersebut, yang dapat menjadi dasar untuk analisis lebih lanjut,seperti pengambilan keputusan ataupun pemodelan.
                """)
            
##### DISTRIBUTIONS #####

    st.markdown(
        "<br>"
        "<h5>Distributions</h5>",  
        unsafe_allow_html=True
    )


    numerical_columns = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] 
                                and col not in ['daerah', 'tahun']]

    num_plots = len(numerical_columns)
    num_cols = 3
    num_rows = (num_plots + num_cols - 1) // num_cols

    # Define a list of colors
    colors = sns.color_palette("husl", num_plots)

    for i in range(num_rows):
        cols = st.columns(num_cols)
        for j in range(num_cols):
            idx = i * num_cols + j
            if idx < num_plots:
                with cols[j]:
                    col = numerical_columns[idx]
                    st.write(col)
                    # Adjust the plotting parameters
                    if col == "Usia":  # Adjust parameters for 'Usia' plot
                        sns.histplot(data=see, x=col, kde=True, color=colors[idx])
                    else:  # For other plots
                        sns.histplot(data=see, x=col, kde=True, color=colors[idx])
                    plt.xlabel('')
                    st.pyplot()

    col1, col2 = st.columns(2)
    with col1:
        with st.expander("🔍 See explanation"):
            st.write("""
            Distribusi data secara visual tidak ada yang normal, bahkan untuk variabel dependen yaitu "selang waktu" sekalipun.
                """)
    
    with col2:
        with st.expander("❓ Why do we use this?"):
            st.write("""
            * Distribusi data ini penting karena memberikan informasi tentang pola-pola umum, termasuk apakah data cenderung terkumpul di sekitar nilai-nilai tertentu atau apakah ada outlier yang signifikan. 
            * Informasi ini membantu kita dalam mengidentifikasi tren, anomali, dan karakteristik khusus dari populasi yang disurvei. 
            * Dengan pemahaman yang lebih baik tentang distribusi variabel, kita dapat membuat keputusan yang lebih tepat dan merancang strategi intervensi yang lebih efektif dalam selang waktu terjadinya tindak pidana.
                """)
    
    top_5_manusia = see[see['tahun'] == 2018].nlargest(5, 'indeks_pembangunan_manusia')['daerah'].tolist()
    top_5_pengangguran = see[see['tahun'] == 2018].nlargest(5, 'tingkat_pengangguran_terbuka')['daerah'].tolist()
    top_5_tindak_pidana = see[see['tahun'] == 2018].nlargest(5, 'jumlah_tindak_pidana')['daerah'].tolist()
    top_5_penduduk_miskin = see[see['tahun'] == 2018].nlargest(5, 'jumlah_penduduk_miskin')['daerah'].tolist()
    top_5_pendidikan = see[see['tahun'] == 2018].nlargest(5, 'tingkat_pendidikan')['daerah'].tolist()

    fig = make_subplots(rows=5, cols=1, subplot_titles=('Indeks Pembangunan Manusia', 'Tingkat Pengangguran Terbuka',
                                                        'Jumlah Tindak Pidana', 'Jumlah Penduduk Miskin ',
                                                        'Tingkat Pendidikan'))

    for daerah in top_5_manusia:
        data_daerah = see[see['daerah'] == daerah]
        fig.add_trace(go.Scatter(x=data_daerah['tahun'], y=data_daerah['indeks_pembangunan_manusia'],
                                    mode='lines+markers', name=daerah), row=1, col=1)
    for daerah in top_5_pengangguran:
        data_daerah = see[see['daerah'] == daerah]
        fig.add_trace(go.Scatter(x=data_daerah['tahun'], y=data_daerah['tingkat_pengangguran_terbuka'],
                                    mode='lines+markers', name=daerah), row=2, col=1)
    for daerah in top_5_tindak_pidana:
        data_daerah = see[see['daerah'] == daerah]
        fig.add_trace(go.Scatter(x=data_daerah['tahun'], y=data_daerah['jumlah_tindak_pidana'],
                                    mode='lines+markers', name=daerah), row=3, col=1)
    for daerah in top_5_penduduk_miskin:
        data_daerah = see[see['daerah'] == daerah]
        fig.add_trace(go.Scatter(x=data_daerah['tahun'], y=data_daerah['jumlah_penduduk_miskin'],
                                    mode='lines+markers', name=daerah), row=4, col=1)
    for daerah in top_5_pendidikan:
        data_daerah = see[see['daerah'] == daerah]
        fig.add_trace(go.Scatter(x=data_daerah['tahun'], y=data_daerah['tingkat_pendidikan'],
                                    mode='lines+markers', name=daerah), row=5, col=1)

    fig.update_layout(height=1500, width=800, title_text='Trend Line Data Berdasarkan Top 5 Daerah', xaxis_title='Tahun')
    st.plotly_chart(fig)
    # Data aggregation and plotting for multiple years

    years = ['2016', '2017', '2018']
    df_all_years = []

    for year in years:
        df_year = data.groupby('Kepolisian/Resort').agg(total_jumlah=pd.NamedAgg(column=f'{year}_jumlah', aggfunc='sum'),
                                                        total_penyelesaian=pd.NamedAgg(column=f'{year}_penyelesaian', aggfunc='sum'))
        df_year['Year'] = year
        df_all_years.append(df_year)

    df_combined = pd.concat(df_all_years).reset_index()

    # Plotting total cases for multiple years
    fig_jumlah = px.bar(df_combined.sort_values(by='total_jumlah', ascending=False),
                        x='Kepolisian/Resort',
                        y='total_jumlah',
                        color='Year',
                        barmode='group',
                        title='Jumlah Kasus di Kepolisian/Resort (2016-2018)')
    st.plotly_chart(fig_jumlah)

    # Plotting resolved cases for multiple years
    fig_penyelesaian = px.bar(df_combined.sort_values(by='total_penyelesaian', ascending=False),
                            x='Kepolisian/Resort',
                            y='total_penyelesaian',
                            color='Year',
                            barmode='group',
                            title='Penyelesaian Kasus di Kepolisian/Resort (2016-2018)')
    st.plotly_chart(fig_penyelesaian)
            



def hipo_testing():
    st.title("Hypothesis Testing")
    st.markdown("***")

    st.header("Uji Multikolinearitas")
    vif = df.copy()
    vif.drop(columns=['tingkat_pendidikan','jumlah_penyelasaian'],inplace=True)
    X_train_resampled_d = sm.add_constant(vif.iloc[:, 2:11])

    vif_data = pd.DataFrame()
    vif_data["Fitur"] = X_train_resampled_d.columns
    vif_data["VIF"] = [variance_inflation_factor(X_train_resampled_d.values, i) for i in range(X_train_resampled_d.shape[1])]
    st.write(vif_data)

    st.write('---')
    st.header("Uji Distribusi Normal")
    for column in df.iloc[:, 2:11].columns:
        stat, p = run_kstest(column)
        st.write(f'**Kolmogorov-Smirnov test for {column}:**')
        st.code(f'- Statistic: {stat:.4f}')
        st.code(f'- p-value: {p:.4f}')

        if p < 0.05:
            st.write(f'- **Reject null hypothesis:** {column} tidak terdistribusi secara normal')
        else:
            st.write(f'Tidak cukup bukti untuk menolak null hypothesis: {column} terdistribusi secara normal')

        st.write('---')

model_fe = smf.ols('selang_waktu ~ 1 + indeks_pembangunan_manusia + tingkat_pengangguran_terbuka + jumlah_tindak_pidana + jumlah_penduduk_miskin + C(daerah) + C(tahun)', data=df).fit()

def model_page():
    st.title('Information Best Model (OLS Regression)')
    st.write('---')
    st.header('Hasil Uji Asumsi Model')
    with st.expander("🔍 Uji Determinasi (R2)"):
        st.write("""
        Model regresi yang telah diestimasi menunjukkan bahwa variabel independen yang digunakan (indeks pembangunan manusia, tingkat pengangguran terbuka, jumlah tindak pidana, jumlah penduduk miskin, serta variabel dummy untuk daerah dan tahun) secara bersama-sama dapat menjelaskan sekitar 50.9% dari variasi dalam variabel terikat (selang waktu). Hal ini menunjukkan bahwa model ini memiliki tingkat kecocokan yang cukup baik dalam menjelaskan hubungan antara variabel-variabel tersebut dengan selang waktu. Sisanya, 49.1% merupakan variabel diluar penelitian.
            """)
    with st.expander("🔍 Uji F-Statistik"):
        st.write("""
            Model regresi didapatkan nilai uji F sebesar P-value(0.158) > alpha(0.05) sehingga gagal tolak H0. Artinya, model dapat dipercaya dengan selang kepercayaan 95% dinyatakan bahwa variabel dependen dan variabel independen tidak signifikan secara serentak atau simultan.
            """)
    with st.expander("🔍 Uji T-Statistik"):
        st.write("""
            Pada model regresi ini, variabel dummy untuk daerah dan tahun serta variabel indeks pembangunan manusia, tingkat pengangguran terbuka, jumlah tindak pidana, dan jumlah penduduk miskin diuji menggunakan uji t untuk menentukan signifikansi masing-masing koefisien. Hasil uji menunjukkan bahwa hanya variabel tahun dengan nilai P-value < alpha (0.05) pada tingkat kepercayaan 95% sehingga tidak menunjukkan signifikansi statistik pada tingkat alpha 0.05 secara parsial terhadap variabel dependen (selang waktu).
            """)
    with st.expander("🔍 Uji Normalitas Residual"):
        st.write("""
            Nilai skewness sebesar 0.276 menunjukkan bahwa distribusi dari suatu variabel memiliki sedikit kemiringan positif, yang menunjukkan bahwa distribusi tersebut cenderung agak lebih condong ke kiri (negatif).
            Nilai kurtosis sebesar 2.502 menunjukkan bahwa distribusi dari suatu variabel memiliki puncak yang sedikit lebih tinggi dan lebih tebal daripada distribusi normal standar, yang menunjukkan sedikit kecenderungan terhadap ekor yang lebih tebal.
            Selain itu, nilai Prob(Omnibus) sebesar 0.225 menunjukkan bahwa model regresi ini memiliki asumsi normalitas pada residual yang cukup terpenuhi, karena nilainya lebih besar dari 0.05. Begitu juga dengan nilai Prob(JB) sebesar 0.268, yang menunjukkan bahwa residual model regresi ini juga tidak signifikan departemen dari distribusi normal.
            """)
    with st.expander("🔍 Uji Autokorelasi"):
        st.write("""
            Nilai Durbin-Watson sebesar 2.775 menunjukkan bahwa tidak ada indikasi kuat mengenai adanya autokorelasi pada residual model regresi ini, karena nilai ini mendekati nilai target yaitu 2 (yang menunjukkan tidak adanya autokorelasi).
            """)

    st.write('---')
    st.code(model_fe.summary())

def input_predict():
    st.title('Input Predict **Best Model (OLS)**')
    st.write('---')
    
    # Pilihan untuk 'daerah' menggunakan inversi transformasi
    daerah = st.selectbox('Daerah', df['daerah'].unique())
    
    tahun = st.selectbox('Tahun', df['tahun'].unique())
    indeks_pembangunan_manusia = st.number_input('Indeks Pembangunan Manusia', value=65)
    tingkat_pengangguran_terbuka = st.number_input('Tingkat Pengangguran Terbuka', value=3.38)
    jumlah_tindak_pidana = st.number_input('Jumlah Tindak Pidana', min_value=0, value=1403)
    jumlah_penduduk_miskin = st.number_input('Jumlah Penduduk Miskin', value=85)
    
    # Prediksi fungsi
    def predict(daerah, tahun, indeks_pembangunan_manusia, tingkat_pengangguran_terbuka, jumlah_tindak_pidana, jumlah_penduduk_miskin):
        # Buat DataFrame untuk prediksi
        df_new = pd.DataFrame({
            'daerah': [daerah],
            'tahun': [tahun],
            'indeks_pembangunan_manusia': [indeks_pembangunan_manusia],
            'tingkat_pengangguran_terbuka': [tingkat_pengangguran_terbuka],
            'jumlah_tindak_pidana': [jumlah_tindak_pidana],
            'jumlah_penduduk_miskin': [jumlah_penduduk_miskin],
        })
        
        # Transformasi dengan PowerTransformer
        df_new[columns_to_transform] = pt.transform(df_new[columns_to_transform])
        
        # Melakukan prediksi dengan model yang sudah di-fit
        prediction = model_fe.predict(df_new)
        return prediction.iloc[0]  # Return the predicted value
    
    if st.button('Predict'):
        prediction_result = predict(daerah, tahun, indeks_pembangunan_manusia, tingkat_pengangguran_terbuka, jumlah_tindak_pidana, jumlah_penduduk_miskin)
        st.write('---')
        st.markdown('**Prediksi Selang Waktu Tindak Pidana:**')
        st.success(f'{prediction_result:.3f} Menit')


st.sidebar.title("Navigation")
selected_page = st.sidebar.radio(
    "Go to",
    # ("Home", "Exploratory Data Analysis", "Hypothesis Testing", "Input Predict","Information Best Model")
    ("Exploratory Data Analysis", "Hypothesis Testing", "Input Predict","Information Best Model")
)

# if selected_page == "Home":
#     home()
if selected_page == "Exploratory Data Analysis":
    eda()
elif selected_page == "Hypothesis Testing":
    hipo_testing()
elif selected_page == "Information Best Model":
    model_page()
elif selected_page == "Input Predict":
    input_predict()

