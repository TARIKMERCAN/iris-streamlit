import streamlit as st
import pandas as pd
from predict import predict
from sklearn.datasets import load_iris

st.set_page_config(page_title="Iris Classifier Pro", layout="wide")

st.markdown("""
<style>
.big-title {font-size: 2rem; font-weight: 800;}
.card {padding: 1rem; border-radius: 1rem; background: #111827; border: 1px solid #374151;}
.prob {font-variant-numeric: tabular-nums;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title"> Iris Classifier </div>', unsafe_allow_html=True)
st.caption("Predict the Iris species and visualize probability scores in real-time.")

iris = load_iris()
target_names = list(iris.target_names)

with st.sidebar:
    st.header("🎛️ Controls")
    preset = st.selectbox(
        "Preset example",
        ["Manual", "Setosa (typical)", "Versicolor (typical)", "Virginica (typical)"],
        index=0,
    )
    st.write("---")
    st.markdown("**About**")
    st.caption("Logistic Regression trained on the Iris dataset. Model loaded from `app/model.joblib`.")

defaults = {
    "Manual": (5.1, 3.5, 1.4, 0.2),
    "Setosa (typical)": (5.0, 3.4, 1.5, 0.2),
    "Versicolor (typical)": (6.0, 2.8, 4.5, 1.3),
    "Virginica (typical)": (6.5, 3.0, 5.5, 2.0),
}[preset]

c1, c2 = st.columns(2)
with c1:
    sepal_length = st.slider("Sepal length (cm)", 0.0, 10.0, float(defaults[0]), 0.1)
    petal_length = st.slider("Petal length (cm)", 0.0, 10.0, float(defaults[2]), 0.1)
with c2:
    sepal_width = st.slider("Sepal width (cm)", 0.0, 10.0, float(defaults[1]), 0.1)
    petal_width = st.slider("Petal width (cm)", 0.0, 10.0, float(defaults[3]), 0.1)

go = st.button("Predict", use_container_width=True)

if go:
    _, name, proba = predict([sepal_length, sepal_width, petal_length, petal_width])

    left, right = st.columns([1, 2])
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Result")
        st.success(f"**{name.title()}**")
        st.caption("Most likely class")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Class probabilities")
        df = pd.DataFrame(
            {"probability": [float(p) for p in proba]},
            index=[t.title() for t in target_names]
        )
        st.bar_chart(df)
        st.markdown("</div>", unsafe_allow_html=True)
