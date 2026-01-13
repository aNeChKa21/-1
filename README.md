# ИИС «DigitalRisk» — Оценка риска влияния социальных сетей на успеваемость студентов

**Интеллектуальная информационная система** на Python + Streamlit для быстрой самодиагностики цифрового риска.

Ссылка на работающее приложение:  
(https://l2rgcejh985yjki8qgu6kg.streamlit.app/)  

## О проекте

Проект анализирует, как использование социальных сетей влияет на академическую успеваемость студентов.  
На основе датасета Kaggle (705 студентов) обучена модель Random Forest, которая предсказывает вероятность негативного влияния (класс "Yes/No").

Ключевые особенности:
- Создание интегрального индекса цифрового благополучия (DWI)
- Добавлены новые признаки: `usage_category`, `high_risk_user`
- Простой веб-интерфейс на Streamlit
- Сравнение с выборкой + персонализированные рекомендации

## Структура проекта
digital_risk_app/
├── app.py                  # Основной файл Streamlit (веб-приложение)
├── rf_model.pkl            # Обученная модель Random Forest
├── requirements.txt        # Зависимости для деплоя
├── Students_Social_Media_Addiction.csv  # Исходный датасет 
└── README.md


## Как запустить локально

1. Склонируй репозиторий:
https://github.com/aNeChKa21/-1.git

3. Создай и активируй виртуальное окружение :
-m venv venv
# на Windows:
venv\Scripts\activate
# на Mac/Linux:
source venv/bin/activate

3. Установи зависимости:
pip install -r requirements.txt

4. Запусти приложение:
-m streamlit run app.py

Откроется браузер: http://localhost:8501

Деплой на Streamlit Cloud 
1. Загрузи файлы в репозиторий на GitHub: app.py, rf_model.pkl, requirements.txt
2. Зайди на https://share.streamlit.io → New app
3. Подключи репозиторий → укажи app.py как основной файл
4. Нажми Deploy — получишь публичную ссылку
