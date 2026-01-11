import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Загрузка модели
try:
    model = joblib.load('rf_model.pkl')
except Exception as e:
    st.error(f"Не удалось загрузить модель: {e}")
    st.stop()

# Загрузка датасета для кодирования и статистики
try:
    dataset = pd.read_csv('Students_Social_Media_Addiction.csv')
except Exception as e:
    st.error(f"Не удалось загрузить датасет: {e}\nПоложи файл в папку с app.py")
    st.stop()

# Средние значения по выборке
avg_values = {
    'Avg_Daily_Usage_Hours': dataset['Avg_Daily_Usage_Hours'].mean(),
    'Sleep_Hours_Per_Night': dataset['Sleep_Hours_Per_Night'].mean(),
    'Mental_Health_Score': dataset['Mental_Health_Score'].mean(),
    'Addicted_Score': dataset['Addicted_Score'].mean()
}

# Кодировщики для всех категорий (обучаем на датасете)
label_encoders = {}
cat_cols = ['Gender', 'Academic_Level', 'Country', 'Most_Used_Platform', 'Relationship_Status']

for col in cat_cols:
    if col in dataset.columns:
        le = LabelEncoder()
        le.fit(dataset[col].astype(str).unique())
        label_encoders[col] = le

# Отдельный энкодер специально для usage_category (как при обучении)
usage_category_encoder = LabelEncoder()
# При обучении использовались те же labels, поэтому fit на них
usage_category_encoder.fit(['Low', 'Medium', 'High', 'Extreme'])

# Функция создания новых признаков + кодирование usage_category
def create_features(df):
    df = df.copy()

    df['digital_wellbeing_index'] = (
        0.4 * (df['Avg_Daily_Usage_Hours'] / 10) +
        0.3 * (df['Addicted_Score'] / 10) +
        0.2 * ((10 - df['Mental_Health_Score']) / 10) +
        0.1 * ((12 - df['Sleep_Hours_Per_Night']) / 12)
    )

    bins = [0, 3, 5, 8, np.inf]
    labels = ['Low', 'Medium', 'High', 'Extreme']
    df['usage_category_str'] = pd.cut(df['Avg_Daily_Usage_Hours'], bins=bins, labels=labels, include_lowest=True)

    # Кодируем usage_category в числа
    df['usage_category'] = usage_category_encoder.transform(df['usage_category_str'])

    df['high_risk_user'] = ((df['Avg_Daily_Usage_Hours'] > 5) & (df['Addicted_Score'] > 7)).astype(int)

    # Удаляем временный строковый столбец
    df = df.drop(columns=['usage_category_str'], errors='ignore')

    return df

# ТОЧНЫЙ порядок признаков из твоей модели
required_columns = [
    'Age', 'Gender', 'Academic_Level', 'Country', 'Most_Used_Platform',
    'Avg_Daily_Usage_Hours', 'Addicted_Score', 'Sleep_Hours_Per_Night',
    'Mental_Health_Score', 'Relationship_Status', 'digital_wellbeing_index',
    'usage_category', 'high_risk_user'
]

# Интерфейс
st.title("ИИС: Оценка цифрового риска студента")
st.markdown("Заполни анкету → получи вероятность риска, статистику, рекомендации и сравнение с выборкой")

with st.form("anketa"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Возраст", 16, 30, value=21)
        hours = st.slider("Время в соцсетях в день (часы)", 0.5, 12.0, 4.5, 0.5)

    with col2:
        addicted = st.slider("Зависимость (1–10)", 1, 10, 5)
        sleep = st.slider("Сон за ночь (часы)", 3.0, 10.0, 7.0, 0.5)
        mental = st.slider("Психическое здоровье (1–10)", 1, 10, 7)

    st.subheader("Дополнительные данные")
    gender = st.selectbox("Пол", ["Male", "Female", "Other"], index=1)
    academic_level = st.selectbox("Уровень образования", ["High School", "Undergraduate", "Graduate"], index=1)
    country = st.text_input("Страна", "Russia")
    platform = st.selectbox("Основная платформа", ["Instagram", "TikTok", "WhatsApp", "Facebook", "Other"], index=0)
    relationship = st.selectbox("Отношения", ["Single", "In relationship", "Married"], index=0)

    submit = st.form_submit_button("Оценить риск", type="primary")

if submit:
    data = {
        'Age': age,
        'Gender': gender,
        'Academic_Level': academic_level,
        'Country': country,
        'Most_Used_Platform': platform,
        'Avg_Daily_Usage_Hours': hours,
        'Addicted_Score': addicted,
        'Sleep_Hours_Per_Night': sleep,
        'Mental_Health_Score': mental,
        'Relationship_Status': relationship
    }

    df = pd.DataFrame([data])
    df = create_features(df)

    # Кодируем категории
    for col in label_encoders:
        if col in df.columns:
            try:
                df[col] = label_encoders[col].transform(df[col])
            except ValueError:
                df[col] = 0

    # Добавляем отсутствующие столбцы и приводим к точному порядку
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[required_columns]

    # Предсказание
    try:
        proba = model.predict_proba(df)[0, 1]
        risk = round(proba * 100, 1)
    except Exception as e:
        st.error(f"Ошибка предсказания: {e}")
        st.stop()

    # Вывод результата
    st.markdown("---")
    if risk > 70:
        st.markdown(f"<h2 style='color:red;'>Риск: {risk}% (высокий)</h2>", unsafe_allow_html=True)
    elif risk > 30:
        st.markdown(f"<h2 style='color:orange;'>Риск: {risk}% (средний)</h2>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h2 style='color:green;'>Риск: {risk}% (низкий)</h2>", unsafe_allow_html=True)

    # Статистика в таблице
    st.subheader("Твои данные vs среднее по выборке")
    stats = pd.DataFrame({
        'Показатель': ['Время в соцсетях (ч)', 'Сон за ночь (ч)', 'Психическое здоровье', 'Зависимость'],
        'Твои значения': [hours, sleep, mental, addicted],
        'Среднее по выборке': [
            round(avg_values['Avg_Daily_Usage_Hours'], 2),
            round(avg_values['Sleep_Hours_Per_Night'], 2),
            round(avg_values['Mental_Health_Score'], 2),
            round(avg_values['Addicted_Score'], 2)
        ]
    })
    st.table(stats)

    # Рекомендации
    st.subheader("Рекомендации")
    if risk > 70:
        st.warning("""
        Высокий риск! Срочно:
        • Ограничь соцсети до 1–1.5 ч/день
        • Без экранов за час до сна
        • Цифровой детокс хотя бы 1 день в неделю
        • Если зависимость >7 — обратись к психологу
        """)
    elif risk > 30:
        st.info("""
        Средний риск:
        • Старайся спать 7–8 ч
        • Используй соцсети осознанно
        • Проверяй статистику времени еженедельно
        """)
    else:
        st.success("""
        Низкий риск — отлично!
        • Продолжай в том же духе
        • Раз в месяц проверяй свои привычки
        """)

    # 4 графика сравнения
    st.subheader("Сравнение с выборкой (705 студентов)")
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    sns.barplot(x=['Твоё', 'Среднее'], y=[hours, avg_values['Avg_Daily_Usage_Hours']], palette='viridis', ax=axs[0, 0])
    axs[0, 0].set_title("Время в соцсетях (ч)")

    sns.barplot(x=['Твой', 'Средний'], y=[sleep, avg_values['Sleep_Hours_Per_Night']], palette='viridis', ax=axs[0, 1])
    axs[0, 1].set_title("Сон за ночь (ч)")

    sns.barplot(x=['Твоё', 'Среднее'], y=[mental, avg_values['Mental_Health_Score']], palette='viridis', ax=axs[1, 0])
    axs[1, 0].set_title("Психическое здоровье (1–10)")

    sns.barplot(x=['Твоя', 'Средняя'], y=[addicted, avg_values['Addicted_Score']], palette='viridis', ax=axs[1, 1])
    axs[1, 1].set_title("Зависимость (1–10)")

    plt.tight_layout()
    st.pyplot(fig)

st.caption("Приложение на основе Random Forest и датасета Kaggle. Январь 2026")