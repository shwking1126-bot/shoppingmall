import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

st.set_page_config(page_title='AI 쇼핑몰 고객 유형 분석', page_icon='🛍️', layout='wide')
st.title('🛍️ AI 쇼핑몰 고객 유형 분석 및 마케팅 추천')
st.caption('Kaggle E-Commerce Data | RFM 분석 + K-Means 비지도학습')
st.info('Kaggle에서 ecommerce-data의 data.csv 파일을 다운로드한 후 아래에 업로드하세요. 원본 데이터는 앱에 포함되지 않습니다.')
file = st.file_uploader('data.csv 업로드', type=['csv', 'xlsx'])
with st.sidebar:
    st.header('분석 설정')
    clusters = st.slider('군집 수', 2, 7, 4)
    st.markdown('[Kaggle 데이터 다운로드](https://www.kaggle.com/datasets/carrie1/ecommerce-data)')
if file is None:
    st.markdown('### 사용 순서\n1. Kaggle 데이터 다운로드\n2. data.csv 업로드\n3. AI 고객 분류 결과 확인\n4. 고객별 CSV 다운로드')
    st.stop()
try:
    if file.name.lower().endswith('.xlsx'):
        raw = pd.read_excel(file)
    else:
        raw = pd.read_csv(file, encoding='utf-8-sig', low_memory=False)
    required = {'InvoiceNo','InvoiceDate','CustomerID','Quantity','UnitPrice'}
    missing = required - set(raw.columns)
    if missing:
        st.error(f'필수 열이 없습니다: {", ".join(sorted(missing))}')
        st.stop()
    df = raw.copy()
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
    df = df.dropna(subset=['CustomerID','InvoiceDate','Quantity','UnitPrice'])
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df = df[~df['InvoiceNo'].astype(str).str.upper().str.startswith('C')]
    df['CustomerID'] = df['CustomerID'].astype(str).str.replace(r'\.0$', '', regex=True)
    df['Amount'] = df['Quantity'] * df['UnitPrice']
    if df.empty:
        st.error('전처리 후 분석할 거래가 없습니다.')
        st.stop()
    reference = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerID').agg(LastPurchase=('InvoiceDate','max'), Frequency=('InvoiceNo','nunique'), Monetary=('Amount','sum')).reset_index()
    rfm['Recency'] = (reference-rfm['LastPurchase']).dt.days
    if len(rfm) < clusters:
        st.error('고객 수보다 군집 수가 많습니다. 군집 수를 줄이세요.')
        st.stop()
    features = rfm[['Recency','Frequency','Monetary']].copy()
    features[['Frequency','Monetary']] = np.log1p(features[['Frequency','Monetary']])
    scaled = StandardScaler().fit_transform(features)
    model = KMeans(n_clusters=clusters, random_state=42, n_init=10)
    rfm['Cluster'] = model.fit_predict(scaled)
    profiles = rfm.groupby('Cluster').agg(Recency=('Recency','mean'),Frequency=('Frequency','mean'),Monetary=('Monetary','mean'))
    # 군집의 상대적 특성에 따라 설명 가능한 라벨을 부여 (고정 정답이 아님)
    rrank = profiles['Recency'].rank(ascending=False, pct=True)
    frank = profiles['Frequency'].rank(pct=True)
    mrank = profiles['Monetary'].rank(pct=True)
    score = rrank + frank + mrank
    best = score.idxmax()
    worst = score.idxmin()
    labels = {i: '일반 고객군' for i in profiles.index}
    labels[best] = '핵심 우수 고객군'
    if worst != best:
        labels[worst] = '관심 필요 고객군'
    rfm['CustomerType'] = rfm['Cluster'].map(labels)
    advice = {'핵심 우수 고객군':'VIP 혜택 및 로열티 프로그램', '관심 필요 고객군':'재방문 쿠폰 및 맞춤형 리마인드', '일반 고객군':'교차 판매 및 상품 추천'}
    rfm['MarketingRecommendation'] = rfm['CustomerType'].map(advice)
    a,b,c,d = st.columns(4)
    a.metric('유효 거래 건수',f'{len(df):,}')
    b.metric('분석 고객 수',f'{len(rfm):,}')
    c.metric('총 거래금액 (£)',f'{rfm.Monetary.sum():,.0f}')
    d.metric('AI 군집 수',str(clusters))
    st.subheader('고객 군집별 규모')
    st.bar_chart(rfm.groupby(['Cluster','CustomerType']).size().rename('고객 수').reset_index(), x='Cluster', y='고객 수')
    st.subheader('군집별 RFM 평균')
    st.dataframe(rfm.groupby(['Cluster','CustomerType'])[['Recency','Frequency','Monetary']].mean().round(2), use_container_width=True)
    st.subheader('고객별 분석 결과')
    st.dataframe(rfm[['CustomerID','Recency','Frequency','Monetary','Cluster','CustomerType','MarketingRecommendation']], use_container_width=True)
    st.caption('Recency: 데이터 마지막 거래일 기준 최근 구매 경과일 / Frequency: 고유 주문 수 / Monetary: 구매금액(GBP). 고객군 명칭은 군집의 상대적 특성에 따른 해석이며 실제 이탈 여부를 예측하지 않습니다.')
    if 2 <= clusters < len(rfm) and len(set(model.labels_)) > 1:
        sample = scaled if len(scaled) <= 5000 else scaled[np.random.default_rng(42).choice(len(scaled),5000,replace=False)]
        st.write(f'실루엣 점수 (참고): {silhouette_score(sample, model.predict(sample)):.3f}')
    csv = rfm[['CustomerID','Recency','Frequency','Monetary','Cluster','CustomerType','MarketingRecommendation']].to_csv(index=False).encode('utf-8-sig')
    st.download_button('📥 분석 결과 CSV 다운로드', csv, 'customer_segments.csv', 'text/csv')
except Exception as exc:
    st.error(f'데이터 처리 중 오류: {exc}')
