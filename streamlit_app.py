"""Interactive Abalone research demo backed exclusively by FastAPI predictions."""

import os
from typing import Any

import altair as alt
import httpx
import pandas as pd
import streamlit as st

from research_app.client import PredictionServiceError, request_prediction
from research_app.data import DESCRIPTIONS, FEATURES, SEX_LABELS, load_reference_data, range_warnings

st.set_page_config(page_title="Abalone age prediction", page_icon="🐚", layout="wide")
TEAL = "#087F8C"
CORAL = "#D96C4A"


@st.cache_data
def reference_data() -> pd.DataFrame:
    """Cache the reference measurements used for visual context only."""
    return load_reference_data()


def clear_result() -> None:
    """Discard a previous response before a new specimen is submitted."""
    st.session_state.pop("result", None)


def fill_example() -> None:
    """Populate the form with one real reference specimen, without predicting."""
    frame = reference_data()
    sample_rows = {
        f"Sample {index}": frame.loc[(frame["Sex"] == category) & (frame["Height"] > 0)].iloc[0]
        for index, category in enumerate(["M", "F", "I"], start=1)
    }
    row = sample_rows[st.session_state["sample_name"]]
    st.session_state["sex"] = row["Sex"]
    for feature in FEATURES:
        st.session_state[feature] = float(row[feature])
    clear_result()


def clear_inputs() -> None:
    """Reset editable measurements and remove the last submitted result."""
    st.session_state["sex"] = None
    for feature in FEATURES:
        st.session_state[feature] = None
    clear_result()


def draw_target_distribution(frame: pd.DataFrame, estimate: float | None = None) -> None:
    """Plot reference ring counts and optionally mark the latest estimate."""
    counts = frame.groupby("Rings").size().reset_index(name="Specimens")
    bars = (
        alt.Chart(counts)
        .mark_bar(color=TEAL, opacity=0.85)
        .encode(
            x=alt.X("Rings:Q", title="Observed shell rings", bin=alt.Bin(step=1)),
            y=alt.Y("Specimens:Q", title="Specimens"),
            tooltip=["Rings", "Specimens"],
        )
    )
    if estimate is not None:
        rule = (
            alt.Chart(pd.DataFrame({"Estimate": [estimate]}))
            .mark_rule(color=CORAL, strokeWidth=3)
            .encode(x="Estimate:Q", tooltip=[alt.Tooltip("Estimate:Q", format=".2f")])
        )
        bars = bars + rule
    st.altair_chart(bars.properties(height=260), use_container_width=True)


def render_prediction(frame: pd.DataFrame, api_url: str) -> None:
    """Render direct specimen inputs, service errors, and the last submitted estimate."""
    st.subheader("Enter measurements")
    st.write("Enter measurements on the same scale as the reference CSV, or start with a real example.")
    st.info("Use the dataset scale. For example, Length = 0.455. Do not enter raw mm or grams; no unit conversion is applied.")
    example_columns = st.columns([2, 1, 1])
    with example_columns[0]:
        st.selectbox("Sample", ["Sample 1", "Sample 2", "Sample 3"], key="sample_name")
    with example_columns[1]:
        st.button("Fill sample values", on_click=fill_example, width="stretch")
    with example_columns[2]:
        st.button("Clear inputs", on_click=clear_inputs, width="stretch")

    input_column, result_column = st.columns([1.15, 1], gap="large")
    with input_column:
        with st.form("specimen_form"):
            sex = st.selectbox(
                "Sex / maturity category",
                ["F", "M", "I"],
                index=None,
                format_func=lambda value: SEX_LABELS[value],
                placeholder="Choose a category",
                key="sex",
                help="I means infant / immature, not an additional biological sex.",
            )
            values: dict[str, Any] = {}
            columns = st.columns(2)
            for index, feature in enumerate(FEATURES):
                with columns[index % 2]:
                    example = float(frame.iloc[0][feature])
                    values[feature] = st.number_input(
                        feature,
                        min_value=0.0,
                        value=None,
                        step=0.001,
                        format="%.4f",
                        key=feature,
                        placeholder=f"e.g. {example:g}",
                        help=(
                            f"{DESCRIPTIONS[feature]}. "
                            f"Reference range: {frame[feature].min():g} to {frame[feature].max():g} (dataset scale)."
                        ),
                    )
            submitted = st.form_submit_button("Predict shell rings", type="primary", width="stretch")
        if submitted:
            clear_result()
            if sex is None or any(value is None for value in values.values()):
                st.error("Choose a category and enter all seven measurements. You can also use Fill sample values.")
            else:
                payload = {"Sex": sex, **values}
                for warning in range_warnings(payload, frame):
                    st.warning(warning)
                try:
                    with st.spinner("Requesting a prediction from FastAPI…"):
                        prediction = request_prediction(api_url, payload)
                    st.session_state["result"] = {"prediction": prediction, "observation": payload, "api_url": api_url}
                except PredictionServiceError as exc:
                    st.error(str(exc))

    with result_column:
        st.subheader("Prediction")
        result = st.session_state.get("result")
        if result is None:
            st.caption("A result appears here after you submit a specimen. Example values never generate a simulated prediction.")
            draw_target_distribution(frame)
        else:
            prediction = result["prediction"]
            st.caption("Last submitted specimen | live API response")
            metrics = st.columns(2)
            metrics[0].metric("Predicted shell rings", f"{prediction:.2f}")
            metrics[1].metric("Derived age | years", f"{prediction + 1.5:.2f}" if prediction >= 0 else "-")
            if prediction < 0:
                st.warning("The linear model returned a non-physical negative estimate. Do not interpret this as a valid age.")
            st.caption("Age = predicted Rings + 1.5.")
            draw_target_distribution(frame, prediction)
            st.caption(
                (
                    "Coral line: your estimate. Bars: observed Rings in the full reference dataset. This is not "
                    "a confidence interval."
                )
            )
            observation = result["observation"]
            context = pd.DataFrame(
                {
                    "Measurement": FEATURES,
                    "Submitted value": [observation[feature] for feature in FEATURES],
                    "Reference percentile": [100 * (frame[feature] <= observation[feature]).mean() for feature in FEATURES],
                }
            )
            st.dataframe(context, hide_index=True, width="stretch")
            st.caption(
                (
                    "Percentiles compare your submitted measurements with all reference rows; they do not "
                    "measure prediction confidence."
                )
            )
            with st.expander("Submitted request"):
                st.json(observation)
                st.caption(f"Prediction service: {result['api_url']}")
        with st.expander("How to interpret this result"):
            st.write(
                (
                    "This is a course research demo, not a field-validated age measurement. The documented "
                    "linear baseline had held-out MAE about 1.59 rings, with substantially larger errors for "
                    "older specimens. This is a historical evaluation, not a live accuracy guarantee or "
                    "uncertainty interval."
                )
            )
            st.write(
                (
                    "The API loads a saved model at startup. Retraining does not automatically reload that "
                    "running service. The current endpoint does not expose a model version."
                )
            )


def render_exploration(frame: pd.DataFrame) -> None:
    """Explore observed distributions without requiring a running API."""
    st.subheader("Explore the reference population")
    st.caption("Observed measurements and Rings from the bundled Abalone CSV. Filtering changes the charts, not the model.")
    controls = st.columns([2, 2])
    with controls[0]:
        categories = st.multiselect(
            "Sex / maturity groups", ["F", "M", "I"], default=["F", "M", "I"], format_func=lambda value: SEX_LABELS[value]
        )
    with controls[1]:
        feature = st.selectbox("Measurement to explore", FEATURES, index=6)
    selected = frame.loc[frame["Sex"].isin(categories)]
    if selected.empty:
        st.info("Choose at least one group to display observations.")
        return
    summary = st.columns(3)
    summary[0].metric("Reference specimens", f"{len(selected):,}")
    summary[1].metric("Median shell rings", f"{selected['Rings'].median():.1f}")
    summary[2].metric("Median derived age", f"{selected['Rings'].median() + 1.5:.1f} years")
    charts = st.columns(2)
    with charts[0]:
        st.markdown("**Observed shell rings**")
        draw_target_distribution(selected)
    with charts[1]:
        st.markdown(f"**{feature} distribution**")
        chart = (
            alt.Chart(selected)
            .mark_bar(color=TEAL)
            .encode(
                x=alt.X(f"{feature}:Q", bin=alt.Bin(maxbins=30), title=f"{feature} | dataset scale"),
                y=alt.Y("count():Q", title="Specimens"),
                tooltip=[alt.Tooltip("count():Q", title="Specimens")],
            )
        )
        result = st.session_state.get("result")
        if result:
            value = result["observation"][feature]
            chart += alt.Chart(pd.DataFrame({"Submitted": [value]})).mark_rule(color=CORAL, strokeWidth=3).encode(x="Submitted:Q")
        st.altair_chart(chart.properties(height=260), use_container_width=True)
        st.caption("If present, the coral line marks the last submitted specimen, even when the reference groups are filtered.")
    scatter = (
        alt.Chart(selected)
        .mark_circle(size=24, opacity=0.3)
        .encode(
            x=alt.X(f"{feature}:Q", title=f"{feature} | dataset scale"),
            y=alt.Y("Rings:Q", title="Observed shell rings"),
            color=alt.Color("Sex:N", scale=alt.Scale(domain=["F", "M", "I"], range=["#087F8C", "#536A8A", "#D99B48"])),
            tooltip=["Sex", feature, "Rings"],
        )
        .interactive()
    )
    st.altair_chart(scatter.properties(height=320), use_container_width=True)
    st.caption(
        (
            "These are observed relationships, not causal effects. Size and weight variables are also "
            "strongly correlated with each other."
        )
    )
    with st.expander("Data notes and source"):
        st.write(
            (
                "The CSV contains 4,177 observations. Two heights are zero and two are unusually large; the "
                "baseline retains the original rows. No observations are automatically corrected or deleted "
                "by this interface."
            )
        )
        st.markdown(
            (
                "[UCI Abalone dataset documentation](https://archive.ics.uci.edu/dataset/1/abalone) "
                "describes scaled continuous measurements and the Rings + 1.5 age convention. Use the CSV's "
                "values directly; this app does not infer a raw-unit conversion."
            )
        )


def main() -> None:
    """Render the research interface without loading a local prediction model."""
    st.title("🐚 Abalone age prediction")
    st.write("Enter measurements to predict shell rings and age. Explore the reference data in the second tab.")
    with st.sidebar:
        st.markdown("### Prediction service")
        api_url = st.text_input("API base URL", value=os.environ.get("API_BASE_URL", "http://127.0.0.1:8002"))
        st.caption("Predictions are requested from FastAPI. No model is loaded by this app.")
        if st.button("Check API connection"):
            try:
                response = httpx.get(f"{api_url.rstrip('/')}/", timeout=5)
                response.raise_for_status()
                st.success("API reachable. Model availability is checked when predicting.")
            except (httpx.HTTPError, httpx.InvalidURL):
                st.error("API is unavailable. Start the service or check its URL.")
        st.divider()
        st.caption("Reference data stays available when the prediction service is offline.")
    try:
        frame = reference_data()
    except (OSError, ValueError) as exc:
        st.error(f"Cannot load the reference dataset: {exc}")
        st.stop()
    predict_tab, explore_tab = st.tabs(["Predict a specimen", "Explore the data"])
    with predict_tab:
        render_prediction(frame, api_url)
    with explore_tab:
        render_exploration(frame)
    st.divider()
    st.caption("Course project | Abalone rings regression | Inputs use the reference dataset scale")


if __name__ == "__main__":
    main()
