import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn import metrics

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="wide")

@st.cache_resource
def load_model():
    base_dir = Path(__file__).parent
    return joblib.load(base_dir / 'model.joblib')

@st.cache_data
def load_config():
    base_dir = Path(__file__).parent
    with open(base_dir / 'config.json', 'r') as f:
        return json.load(f)

@st.cache_data
def load_test_data():
    base_dir = Path(__file__).parent
    return pd.read_csv(base_dir / 'test.csv')

@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]

model = load_model()
config = load_config()
df_test = load_test_data()

threshold = st.sidebar.slider(
    "Порог (threshold)",
    min_value=0.0,
    max_value=1.0,
    value=float(config['threshold']),
    step=0.01
)

st.title("🍷 Оценка качества вина")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Оценка одного вина", "Модельные метрики", "Пакетное предсказание", "Что выучила модель"])

with tab1:
    st.write("Введите характеристики вина, чтобы узнать, является ли оно «хорошим» (оценка экспертов ≥ 7).")

    feature_cols = [c for c in df_test.columns if c != 'target']

    with st.form("wine_form"):
        col1, col2, col3 = st.columns(3)
        input_data = {}
        cols = [col1, col2, col3]
        for i, col_name in enumerate(feature_cols):
            if col_name == 'is_red':
                continue
            series = df_test[col_name]
            min_val = float(series.min())
            max_val = float(series.max())
            med_val = float(series.median())
            step = (max_val - min_val) / 100 if (max_val - min_val) > 0 else 0.01
            with cols[i % 3]:
                input_data[col_name] = st.number_input(
                    label=col_name.replace('_', ' ').capitalize(),
                    min_value=min_val,
                    max_value=max_val,
                    value=med_val,
                    step=step,
                    format="%.4f" if step < 0.01 else "%.2f"
                )
        st.subheader("Тип вина")
        is_red = st.radio("Цвет вина", options=["Красное (1)", "Белое (0)"], horizontal=True)
        input_data['is_red'] = 1 if "Красное" in is_red else 0
        submit_button = st.form_submit_button(label="Оценить вино 🍇")

    if submit_button:
        input_df = pd.DataFrame([input_data])[feature_cols]
        proba = model.predict_proba(input_df)[0, 1]
        st.divider()
        st.subheader("Результат оценки")
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric(label="Вероятность, что вино хорошее", value=f"{proba:.2%}")
        with col_res2:
            if proba >= threshold:
                st.success(f"✅ **Вино рекомендуется!** (Вероятность {proba:.2%} ≥ порога {threshold})")
            else:
                st.error(f"❌ **Вино не рекомендуется.** (Вероятность {proba:.2%} < порога {threshold})")

with tab2:
    st.header("Метрики модели на тестовой выборке")

    X_test = df_test.drop(columns='target')
    y_test = df_test['target']

    # 4.5: вероятности считаются один раз и кэшируются
    y_proba = test_proba(model, X_test)
    y_pred = (y_proba >= threshold).astype(int)

    st.subheader("Матрица ошибок")
    cm = metrics.confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    metrics.ConfusionMatrixDisplay(cm, display_labels=['Bad (0)', 'Good (1)']).plot(ax=ax_cm, cmap='Blues',
                                                                                    colorbar=False)
    st.pyplot(fig_cm)

    st.subheader("Метрики при текущем пороге")
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Precision", f"{metrics.precision_score(y_test, y_pred, zero_division=0):.4f}")
    col_m2.metric("Recall", f"{metrics.recall_score(y_test, y_pred, zero_division=0):.4f}")
    col_m3.metric("F1", f"{metrics.f1_score(y_test, y_pred, zero_division=0):.4f}")

    st.subheader("ROC-кривая")
    fpr, tpr, _ = metrics.roc_curve(y_test, y_proba)
    fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
    ax_roc.plot(fpr, tpr, label='ROC curve')
    ax_roc.plot([0, 1], [0, 1], 'k--', label='Random')
    ax_roc.scatter(
        metrics.recall_score(y_test, y_pred, zero_division=0),
        metrics.recall_score(y_test, y_pred, zero_division=0) * 0,
        color='red', s=100, zorder=5, label=f'Threshold {threshold:.2f}'
    )
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend()
    st.pyplot(fig_roc)

    st.subheader("PR-кривая")
    precision, recall, _ = metrics.precision_recall_curve(y_test, y_proba)
    fig_pr, ax_pr = plt.subplots(figsize=(7, 5))
    ax_pr.plot(recall, precision, label='PR curve')
    ax_pr.axhline(y=y_test.mean(), color='k', linestyle='--', label='Random (prevalence)')
    ax_pr.scatter(
        metrics.recall_score(y_test, y_pred, zero_division=0),
        metrics.precision_score(y_test, y_pred, zero_division=0),
        color='red', s=100, zorder=5, label=f'Threshold {threshold:.2f}'
    )
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.legend()
    st.pyplot(fig_pr)

    st.subheader("Как читать порог")
    st.write("""
    - **Если увеличить порог:** модель станет строже. В подборку попадут только те вина, в качестве которых модель уверена. 
      Precision вырастет (меньше плохих вин), но Recall упадёт (подборка станет короче, часть хороших вин потеряем).
    - **Если уменьшить порог:** модель станет мягче. В подборку попадёт больше вин, 
      Recall вырастет (найдём больше хороших), но Precision упадёт (появится риск рекомендовать посредственное вино).
    """)

with tab3:
    st.header("Пакетное предсказание")
    st.write("Загрузите CSV-файл с признаками вин (без столбца target).")

    uploaded_file = st.file_uploader("Выберите CSV-файл", type=["csv"])

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)

            feature_cols = [c for c in df_test.columns if c != 'target']
            missing_cols = [c for c in feature_cols if c not in df_upload.columns]

            if missing_cols:
                st.error(f"Ошибка: в файле не хватает следующих столбцов: {', '.join(missing_cols)}")
            else:
                X_upload = df_upload[feature_cols]
                probas = model.predict_proba(X_upload)[:, 1]
                decisions = (probas >= threshold).astype(int)

                df_result = df_upload.copy()
                df_result['probability_good'] = probas
                df_result['decision'] = decisions

                st.subheader("Результаты предсказания")
                st.dataframe(df_result)

                csv_data = df_result.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Скачать результаты (CSV)",
                    data=csv_data,
                    file_name="wine_predictions.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Не удалось прочитать файл: {e}")

with tab4:
    st.header("Что выучила модель")
    st.write("Коэффициенты логистической регрессии (после масштабирования признаков).")

    log_reg = model.named_steps['logisticregression']

    feature_cols = [c for c in df_test.columns if c != 'target']
    coefs = pd.Series(log_reg.coef_[0], index=feature_cols)
    coefs_sorted = coefs.sort_values()

    fig_coef, ax_coef = plt.subplots(figsize=(8, 6))
    colors = ['#d62728' if v < 0 else '#2ca02c' for v in coefs_sorted.values]
    ax_coef.barh(coefs_sorted.index, coefs_sorted.values, color=colors)
    ax_coef.axvline(0, color='black', linewidth=0.8)
    ax_coef.set_xlabel("Коэффициент (вклад в log-odds)")
    ax_coef.set_title("Коэффициенты логистической регрессии")
    st.pyplot(fig_coef)

    top_pos = coefs_sorted.tail(3)[::-1]
    top_neg = coefs_sorted.head(3)

    st.subheader("Сильнее всего голосуют ЗА хорошее вино:")
    for name, val in top_pos.items():
        st.write(f"- **{name}**: {val:+.4f}")

    st.subheader("Сильнее всего голосуют ПРОТИВ (за плохое вино):")
    for name, val in top_neg.items():
        st.write(f"- **{name}**: {val:+.4f}")

# ОТВЕТЫ НА ВОПРОСЫ ЗАДАНИЯ 4.5
# 1. Почему для модели используется st.cache_resource, а для таблиц – st.cache_data?
#
# st.cache_resource используется для "тяжелых" объектов, которые нельзя просто
# скопировать (модели, подключения к БД). Он хранит ОДИН экземпляр объекта в памяти
# сервера и отдает его всем пользователям.
#
# st.cache_data используется для сериализуемых данных (DataFrame, dict, список,
# числа). Streamlit делает "снимок" данных и отдает её каждому пользователю.
# Это безопасно, потому что таблицу можно скопировать, и каждый пользователь
# получит свою независимую копию.
# test.csv и config.json — это данные, их безопасно кэшировать через cache_data.
#
# 2. Что изменится, если вызвать joblib.load без декоратора?
#
# Streamlit перезапускает скрипт сверху вниз при каждом действии пользователя. Без декоратора joblib.load будет
# вызываться каждый раз при перезапуске, то есть модель будет заново читаться
# с диска и десериализоваться на каждом шаге. Это медленно и приводит к лишней нагрузке на диск.
# С @st.cache_resource модель загружается один раз за всё время работы приложения.
