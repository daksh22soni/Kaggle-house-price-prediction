import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import entropy


@st.cache_data
def load_data():
    return pd.read_csv("data/raw/train.csv")


def render_data_viz():
    st.markdown("## Data Visualization")

    df = load_data()

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    feature_num_cols = [c for c in num_cols if c != "SalePrice"]

    tab1, tab2, tab3 = st.tabs(
        ["EDA Overview", "Univariate Analysis", "Bivariate & Multivariate"]
    )

    # =========================================================
    # TAB 1 — EDA OVERVIEW
    # =========================================================
    with tab1:
        st.markdown("### Dataset at a Glance")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Numerical Features", len(num_cols))
        c4.metric("Categorical Features", len(cat_cols))

        st.divider()

        st.markdown("#### Missing Values")
        missing = (
            df.isnull()
            .sum()
            .reset_index()
            .rename(columns={"index": "Feature", 0: "Missing Count"})
        )
        missing["Missing %"] = (missing["Missing Count"] / len(df) * 100).round(2)
        missing = missing[missing["Missing Count"] > 0].sort_values(
            "Missing %", ascending=False
        )

        if missing.empty:
            st.success("No missing values found!")
        else:
            fig_miss = px.bar(
                missing,
                x="Feature",
                y="Missing %",
                text="Missing %",
                color="Missing %",
                color_continuous_scale="OrRd",
                title=f"{len(missing)} features have missing values",
            )
            fig_miss.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_miss.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig_miss, use_container_width=True)

        st.divider()

        st.markdown("#### Target Variable — SalePrice")
        col_a, col_b = st.columns(2)

        with col_a:
            fig_target = px.histogram(
                df,
                x="SalePrice",
                nbins=60,
                marginal="box",
                title="SalePrice Distribution (Raw)",
                color_discrete_sequence=["#636EFA"],
            )
            st.plotly_chart(fig_target, use_container_width=True)

        with col_b:
            fig_log = px.histogram(
                df,
                x=np.log1p(df["SalePrice"]),
                nbins=60,
                marginal="box",
                title="SalePrice Distribution (Log-transformed)",
                color_discrete_sequence=["#EF553B"],
                labels={"x": "log(SalePrice + 1)"},
            )
            st.plotly_chart(fig_log, use_container_width=True)

        skew_val = df["SalePrice"].skew()
        kurt_val = df["SalePrice"].kurt()
        st.caption(
            f"**Skewness:** {skew_val:.3f} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"**Kurtosis:** {kurt_val:.3f} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"**Mean:** ${df['SalePrice'].mean():,.0f} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"**Median:** ${df['SalePrice'].median():,.0f}"
        )

        st.divider()

        st.markdown("#### Skewness of Numerical Features")
        skew_df = (
            df[feature_num_cols]
            .skew()
            .reset_index()
            .rename(columns={"index": "Feature", 0: "Skewness"})
            .sort_values("Skewness", key=abs, ascending=False)
        )
        skew_df["Abs Skewness"] = skew_df["Skewness"].abs()
        skew_df["Flag"] = skew_df["Abs Skewness"].apply(
            lambda x: "High (>1)"
            if x > 1
            else ("Moderate (0.5–1)" if x > 0.5 else "Low (<0.5)")
        )

        fig_skew = px.bar(
            skew_df.head(25),
            x="Skewness",
            y="Feature",
            orientation="h",
            color="Flag",
            color_discrete_map={
                "High (>1)": "#EF553B",
                "Moderate (0.5–1)": "#FFA15A",
                "Low (<0.5)": "#00CC96",
            },
            title="Top 25 Most Skewed Features",
        )
        fig_skew.update_layout(yaxis=dict(autorange="reversed"), height=550)
        st.plotly_chart(fig_skew, use_container_width=True)

        st.divider()

        st.markdown("#### Data Types & Sample")
        st.dataframe(
            df.dtypes.reset_index().rename(columns={"index": "Feature", 0: "DType"}),
            use_container_width=True,
            height=220,
        )
        st.markdown("**Sample rows**")
        st.dataframe(df.sample(5, random_state=42), use_container_width=True)

    # =========================================================
    # TAB 2 — UNIVARIATE ANALYSIS
    # =========================================================
    with tab2:

        u_type = st.radio(
            "Analyse",
            ["Numerical Feature", "Categorical Feature", "Top Correlations with SalePrice"],
            horizontal=True,
        )

        st.divider()

        # ── Numerical ─────────────────────────────────────────
        if u_type == "Numerical Feature":

            feature = st.selectbox("Select Numerical Feature", feature_num_cols)
            series_raw = df[feature].dropna()

            log_toggle = st.toggle("Apply Log Transform (log1p)", value=False)
            series = np.log1p(series_raw) if log_toggle else series_raw
            label = f"log1p({feature})" if log_toggle else feature

            q1 = series_raw.quantile(0.25)
            q3 = series_raw.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_mask = (df[feature] < lower) | (df[feature] > upper)
            outliers = df[outlier_mask]

            kde_fn = stats.gaussian_kde(series)
            kde_x = np.linspace(series.min(), series.max(), 300)
            kde_y = kde_fn(kde_x)

            # ROW 1 — Histogram + KDE  |  Violin
            st.markdown("#### Distribution Shape")
            r1c1, r1c2 = st.columns(2)

            with r1c1:
                nbins = st.slider("Histogram bins", 10, 100, 50, key="hist_bins")
                fig_hist = go.Figure()
                fig_hist.add_trace(
                    go.Histogram(
                        x=series,
                        nbinsx=nbins,
                        name="Histogram",
                        marker_color="#636EFA",
                        opacity=0.75,
                    )
                )
                bin_width = (series.max() - series.min()) / nbins
                kde_y_scaled = kde_y * len(series) * bin_width
                fig_hist.add_trace(
                    go.Scatter(
                        x=kde_x,
                        y=kde_y_scaled,
                        mode="lines",
                        name="KDE",
                        line=dict(color="#EF553B", width=2.5),
                    )
                )
                fig_hist.update_layout(
                    title=f"Histogram + KDE — {label}",
                    xaxis_title=label,
                    yaxis_title="Count",
                    bargap=0.05,
                    legend=dict(orientation="h", y=1.1),
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with r1c2:
                fig_violin = px.violin(
                    df,
                    y=feature,
                    box=True,
                    points="outliers",
                    title=f"Violin Plot — {label}",
                    color_discrete_sequence=["#AB63FA"],
                    labels={"y": label},
                )
                st.plotly_chart(fig_violin, use_container_width=True)

            # ROW 2 — Density  |  CDF
            st.markdown("#### Density & Cumulative Distribution")
            r2c1, r2c2 = st.columns(2)

            with r2c1:
                fig_density = go.Figure()
                fig_density.add_trace(
                    go.Scatter(
                        x=kde_x,
                        y=kde_y,
                        mode="lines",
                        fill="tozeroy",
                        line=dict(color="#00CC96", width=2),
                        name="Density",
                    )
                )
                fig_density.update_layout(
                    title=f"Density Curve — {label}",
                    xaxis_title=label,
                    yaxis_title="Density",
                )
                st.plotly_chart(fig_density, use_container_width=True)

            with r2c2:
                sorted_vals = np.sort(series)
                cdf_y = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

                fig_cdf = go.Figure()
                fig_cdf.add_trace(
                    go.Scatter(
                        x=sorted_vals,
                        y=cdf_y,
                        mode="lines",
                        line=dict(color="#FFA15A", width=2),
                        name="ECDF",
                    )
                )
                for pct, val in [
                    (0.25, series.quantile(0.25)),
                    (0.50, series.quantile(0.50)),
                    (0.75, series.quantile(0.75)),
                ]:
                    fig_cdf.add_shape(
                        type="line",
                        x0=series.min(), x1=val, y0=pct, y1=pct,
                        line=dict(color="grey", dash="dot", width=1),
                    )
                    fig_cdf.add_shape(
                        type="line",
                        x0=val, x1=val, y0=0, y1=pct,
                        line=dict(color="grey", dash="dot", width=1),
                    )
                fig_cdf.update_layout(
                    title=f"Empirical CDF — {label}",
                    xaxis_title=label,
                    yaxis_title="Cumulative Probability",
                    yaxis=dict(tickformat=".0%"),
                )
                st.plotly_chart(fig_cdf, use_container_width=True)

            # ROW 3 — Outlier scatter  |  Rank plot
            st.markdown("#### Outlier Detection & Rank Structure")
            r3c1, r3c2 = st.columns(2)

            with r3c1:
                colors = outlier_mask.map({True: "#EF553B", False: "#636EFA"})
                fig_outlier = go.Figure()
                fig_outlier.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[feature],
                        mode="markers",
                        marker=dict(color=colors, size=4, opacity=0.7),
                        name="Values",
                        text=[
                            f"Index: {i}<br>{feature}: {v:.2f}"
                            for i, v in zip(df.index, df[feature])
                        ],
                        hoverinfo="text",
                    )
                )
                fig_outlier.add_hline(
                    y=upper, line_dash="dash", line_color="#EF553B",
                    annotation_text=f"Upper fence ({upper:.1f})",
                    annotation_position="top right",
                )
                fig_outlier.add_hline(
                    y=lower, line_dash="dash", line_color="#EF553B",
                    annotation_text=f"Lower fence ({lower:.1f})",
                    annotation_position="bottom right",
                )
                fig_outlier.update_layout(
                    title=f"Outlier Detection — {feature} (raw)",
                    xaxis_title="Index",
                    yaxis_title=feature,
                    showlegend=False,
                )
                st.plotly_chart(fig_outlier, use_container_width=True)
                o1, o2, o3 = st.columns(3)
                o1.metric("Outliers", len(outliers))
                o2.metric("Outlier %", f"{len(outliers)/len(df)*100:.2f}%")
                o3.metric("IQR", f"{iqr:.2f}")

            with r3c2:
                sorted_raw = np.sort(series_raw)
                fig_rank = go.Figure()
                fig_rank.add_trace(
                    go.Scatter(
                        x=np.arange(len(sorted_raw)),
                        y=sorted_raw,
                        mode="lines",
                        line=dict(color="#19D3F3", width=2),
                        name="Sorted values",
                    )
                )
                fig_rank.update_layout(
                    title=f"Rank Plot — {feature}",
                    xaxis_title="Rank (sorted index)",
                    yaxis_title=feature,
                )
                st.plotly_chart(fig_rank, use_container_width=True)

            # ROW 4 — Box plot  |  Summary stats
            st.markdown("#### Box Plot & Summary")
            r4c1, r4c2 = st.columns([1, 1])

            with r4c1:
                fig_box = px.box(
                    df,
                    y=feature,
                    points="outliers",
                    title=f"Box Plot — {feature}",
                    color_discrete_sequence=["#636EFA"],
                )
                st.plotly_chart(fig_box, use_container_width=True)

            with r4c2:
                st.markdown("**Summary Statistics**")
                desc = series_raw.describe().round(3)
                stats_df = pd.DataFrame(
                    {
                        "Statistic": [
                            "Count", "Mean", "Std Dev", "Min",
                            "25%", "Median", "75%", "Max",
                            "Skewness", "Kurtosis",
                        ],
                        "Value": [
                            f"{desc['count']:.0f}",
                            f"{desc['mean']:.3f}",
                            f"{desc['std']:.3f}",
                            f"{desc['min']:.3f}",
                            f"{desc['25%']:.3f}",
                            f"{desc['50%']:.3f}",
                            f"{desc['75%']:.3f}",
                            f"{desc['max']:.3f}",
                            f"{series_raw.skew():.3f}",
                            f"{series_raw.kurt():.3f}",
                        ],
                    }
                )
                st.dataframe(stats_df, use_container_width=True, hide_index=True, height=385)

        # ── Categorical ───────────────────────────────────────
        elif u_type == "Categorical Feature":

            feature = st.selectbox("Select Categorical Feature", cat_cols)
            series_cat = df[feature]

            # ── Pre-compute shared values ──────────────────────
            vc = series_cat.value_counts().reset_index()
            vc.columns = [feature, "Count"]
            vc["Percentage"] = (vc["Count"] / len(df) * 100).round(2)
            vc_sorted = vc.sort_values("Count", ascending=False).reset_index(drop=True)
            vc_sorted["Cum %"] = vc_sorted["Percentage"].cumsum().round(2)

            n_unique = series_cat.nunique()
            probs = vc["Count"] / vc["Count"].sum()
            ent = entropy(probs)
            dominant_pct = vc_sorted.iloc[0]["Percentage"]
            top2_pct = vc_sorted.head(2)["Percentage"].sum()

            # ── Header metrics row ─────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Unique Categories", n_unique)
            m2.metric("Entropy", f"{ent:.3f}")
            m3.metric("Dominant Category %", f"{dominant_pct:.1f}%")
            m4.metric("Top 2 Combined %", f"{top2_pct:.1f}%")

            encoding_hint = (
                "✅ Low cardinality → One-Hot Encoding recommended"
                if n_unique <= 10
                else "⚠️ High cardinality → Target / Ordinal Encoding recommended"
            )
            st.caption(encoding_hint)

            st.divider()

            # ==================================================
            # ROW 1 — Sorted Horizontal Bar  |  Donut
            # ==================================================
            st.markdown("#### Distribution Overview")
            r1c1, r1c2 = st.columns(2)

            with r1c1:
                fig_hbar = px.bar(
                    vc_sorted.sort_values("Count"),
                    x="Count",
                    y=feature,
                    orientation="h",
                    text="Count",
                    color="Count",
                    color_continuous_scale="Blues",
                    title=f"Sorted Value Counts — {feature}",
                )
                fig_hbar.update_traces(textposition="outside")
                fig_hbar.update_layout(
                    coloraxis_showscale=False,
                    height=max(300, n_unique * 28),
                    yaxis_title="",
                )
                st.plotly_chart(fig_hbar, use_container_width=True)

            with r1c2:
                fig_pie = px.pie(
                    vc_sorted,
                    names=feature,
                    values="Count",
                    title=f"Proportion — {feature}",
                    hole=0.4,
                )
                fig_pie.update_traces(textinfo="percent+label")
                st.plotly_chart(fig_pie, use_container_width=True)

            # ==================================================
            # ROW 2 — Top-N + Others  |  Pareto Chart
            # ==================================================
            st.markdown("#### Dominance & Concentration")
            r2c1, r2c2 = st.columns(2)

            with r2c1:
                max_top = min(n_unique, 20)
                if max_top<=2:
                    top_n=max_top
                    st.info(f"Only {max_top} categories available — showing all.")
                else:
                    top_n=st.slider(
                        "Top N categories",
                        2,
                        max_top,
                        min(10,max_top),
                        key="cat_top_n"
                    )

                top_rows = vc_sorted.head(top_n).copy()
                others_count = vc_sorted.iloc[top_n:]["Count"].sum()

                if others_count > 0:
                    others_row = pd.DataFrame(
                        {feature: ["Others"], "Count": [others_count],
                         "Percentage": [round(others_count / len(df) * 100, 2)],
                         "Cum %": [100.0]}
                    )
                    top_rows = pd.concat([top_rows, others_row], ignore_index=True)

                colors_topn = [
                    "#EF553B" if cat == "Others" else "#636EFA"
                    for cat in top_rows[feature]
                ]

                fig_topn = go.Figure()
                fig_topn.add_trace(
                    go.Bar(
                        x=top_rows[feature],
                        y=top_rows["Count"],
                        marker_color=colors_topn,
                        text=top_rows["Count"],
                        textposition="outside",
                        name="Count",
                    )
                )
                fig_topn.update_layout(
                    title=f"Top {top_n} + Others — {feature}",
                    xaxis_title=feature,
                    yaxis_title="Count",
                    xaxis_tickangle=-35,
                    showlegend=False,
                )
                st.plotly_chart(fig_topn, use_container_width=True)

            with r2c2:
                # Pareto: bar (count) + line (cumulative %)
                fig_pareto = go.Figure()
                fig_pareto.add_trace(
                    go.Bar(
                        x=vc_sorted[feature],
                        y=vc_sorted["Count"],
                        name="Count",
                        marker_color="#636EFA",
                        opacity=0.85,
                    )
                )
                fig_pareto.add_trace(
                    go.Scatter(
                        x=vc_sorted[feature],
                        y=vc_sorted["Cum %"],
                        name="Cumulative %",
                        mode="lines+markers",
                        line=dict(color="#EF553B", width=2),
                        marker=dict(size=6),
                        yaxis="y2",
                    )
                )
                fig_pareto.add_hline(
                    y=80, line_dash="dot", line_color="grey",
                    annotation_text="80%", annotation_position="right",
                    yref="y2",
                )
                fig_pareto.update_layout(
                    title=f"Pareto Chart — {feature}",
                    xaxis=dict(title=feature, tickangle=-35),
                    yaxis=dict(title="Count"),
                    yaxis2=dict(
                        title="Cumulative %",
                        overlaying="y",
                        side="right",
                        range=[0, 105],
                        ticksuffix="%",
                    ),
                    legend=dict(orientation="h", y=1.1),
                )
                st.plotly_chart(fig_pareto, use_container_width=True)

            # ==================================================
            # ROW 3 — Cumulative Distribution  |  Rare Categories
            # ==================================================
            st.markdown("#### Tail Analysis & Rare Categories")
            r3c1, r3c2 = st.columns(2)

            with r3c1:
                fig_cumline = px.line(
                    vc_sorted,
                    x=feature,
                    y="Cum %",
                    markers=True,
                    title=f"Cumulative Distribution — {feature}",
                    labels={"Cum %": "Cumulative %"},
                )
                fig_cumline.add_hline(
                    y=80, line_dash="dot", line_color="#EF553B",
                    annotation_text="80% threshold",
                    annotation_position="bottom right",
                )
                fig_cumline.add_hline(
                    y=95, line_dash="dot", line_color="#FFA15A",
                    annotation_text="95% threshold",
                    annotation_position="bottom right",
                )
                fig_cumline.update_layout(
                    xaxis_tickangle=-35,
                    yaxis=dict(ticksuffix="%", range=[0, 105]),
                )
                st.plotly_chart(fig_cumline, use_container_width=True)

            with r3c2:
                threshold = st.slider(
                    "Rare category threshold (%)", 0.5, 10.0, 2.0,
                    step=0.5, key="rare_thresh"
                )
                rare = vc_sorted[vc_sorted["Percentage"] < threshold].copy()

                st.markdown(f"**Rare Categories (< {threshold}%)**")

                if rare.empty:
                    st.success(f"No rare categories below {threshold}% threshold.")
                else:
                    r_m1, r_m2, r_m3 = st.columns(3)
                    r_m1.metric("Rare Count", len(rare))
                    r_m2.metric(
                        "Rows Affected",
                        int(rare["Count"].sum()),
                    )
                    r_m3.metric(
                        "% of Dataset",
                        f"{rare['Percentage'].sum():.2f}%",
                    )

                    fig_rare = px.bar(
                        rare.sort_values("Count"),
                        x="Count",
                        y=feature,
                        orientation="h",
                        text="Percentage",
                        color="Count",
                        color_continuous_scale="OrRd",
                        title=f"Rare Categories — {feature}",
                    )
                    fig_rare.update_traces(
                        texttemplate="%{text:.2f}%", textposition="outside"
                    )
                    fig_rare.update_layout(
                        coloraxis_showscale=False,
                        height=max(250, len(rare) * 35),
                        yaxis_title="",
                    )
                    st.plotly_chart(fig_rare, use_container_width=True)

                    st.caption(
                        "💡 Tip: Consider grouping rare categories into an 'Other' bucket "
                        "before one-hot encoding to prevent model instability."
                    )

        # ── Top Correlations with SalePrice ───────────────────
        elif u_type == "Top Correlations with SalePrice":
            n = st.slider("Show top N features", 5, 30, 15)

            corr_series = (
                df[num_cols]
                .corr()["SalePrice"]
                .drop("SalePrice")
                .sort_values(key=abs, ascending=False)
                .head(n)
                .reset_index()
            )
            corr_series.columns = ["Feature", "Correlation"]
            corr_series["Direction"] = corr_series["Correlation"].apply(
                lambda x: "Positive" if x >= 0 else "Negative"
            )

            fig_corr = px.bar(
                corr_series,
                x="Correlation",
                y="Feature",
                orientation="h",
                color="Direction",
                color_discrete_map={"Positive": "#00CC96", "Negative": "#EF553B"},
                title=f"Top {n} Features Correlated with SalePrice",
                text="Correlation",
            )
            fig_corr.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_corr.update_layout(yaxis=dict(autorange="reversed"), height=500)
            st.plotly_chart(fig_corr, use_container_width=True)

    # =========================================================
    # TAB 3 — BIVARIATE & MULTIVARIATE
    # =========================================================
    with tab3:

        bv_type = st.radio(
            "Analysis Type",
            ["Scatter Plot", "Categorical vs SalePrice", "Correlation Matrix", "Pair Plot"],
            horizontal=True,
        )

        st.divider()

        if bv_type == "Scatter Plot":
            col1, col2, col3 = st.columns(3)
            with col1:
                x = st.selectbox("X-axis", feature_num_cols, index=0)
            with col2:
                y_opts = num_cols
                y = st.selectbox(
                    "Y-axis",
                    y_opts,
                    index=y_opts.index("SalePrice") if "SalePrice" in y_opts else 1,
                )
            with col3:
                color_by = st.selectbox("Colour by (optional)", ["None"] + cat_cols)

            trendline = st.toggle("Show trendline (OLS)", value=True)

            fig_scatter = px.scatter(
                df,
                x=x,
                y=y,
                color=None if color_by == "None" else color_by,
                trendline="ols" if trendline else None,
                opacity=0.65,
                title=f"{x} vs {y}",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            if trendline:
                r, p = stats.pearsonr(df[[x, y]].dropna()[x], df[[x, y]].dropna()[y])
                st.caption(
                    f"**Pearson r:** {r:.4f} &nbsp;&nbsp;|&nbsp;&nbsp; **p-value:** {p:.4e}"
                )

        elif bv_type == "Categorical vs SalePrice":
            feature = st.selectbox("Select Categorical Feature", cat_cols)
            plot_kind = st.radio("Plot type", ["Violin", "Box", "Bar (Mean)"], horizontal=True)

            order = (
                df.groupby(feature)["SalePrice"]
                .median()
                .sort_values()
                .index.tolist()
            )

            if plot_kind == "Violin":
                fig = px.violin(
                    df, x=feature, y="SalePrice", box=True, points="outliers",
                    category_orders={feature: order},
                    title=f"SalePrice by {feature}",
                )
            elif plot_kind == "Box":
                fig = px.box(
                    df, x=feature, y="SalePrice", points="outliers",
                    category_orders={feature: order},
                    title=f"SalePrice by {feature}",
                    color=feature,
                )
            else:
                mean_df = (
                    df.groupby(feature)["SalePrice"]
                    .agg(["mean", "count"])
                    .reset_index()
                    .rename(columns={"mean": "Mean SalePrice", "count": "Count"})
                    .sort_values("Mean SalePrice")
                )
                fig = px.bar(
                    mean_df, x=feature, y="Mean SalePrice", text="Mean SalePrice",
                    color="Mean SalePrice", color_continuous_scale="Viridis",
                    title=f"Mean SalePrice by {feature}",
                    category_orders={feature: mean_df[feature].tolist()},
                )
                fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
                fig.update_layout(coloraxis_showscale=False)

            fig.update_layout(xaxis_tickangle=-35, height=500)
            st.plotly_chart(fig, use_container_width=True)

        elif bv_type == "Correlation Matrix":
            default_feats = feature_num_cols[:8]
            selected = st.multiselect(
                "Select numerical features (include SalePrice to see target correlation)",
                num_cols,
                default=default_feats + ["SalePrice"],
            )

            if len(selected) < 2:
                st.warning("Please select at least 2 features.")
            else:
                corr = df[selected].corr()
                fig_hm = px.imshow(
                    corr,
                    text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1,
                    title="Correlation Matrix",
                    aspect="auto",
                )
                fig_hm.update_layout(height=max(400, len(selected) * 45))
                st.plotly_chart(fig_hm, use_container_width=True)
        
        elif bv_type=="Pair Plot":
            bad_features = ["Id", "MSSubClass"]
            filtered_features = [f for f in feature_num_cols if f not in bad_features]
            selected = st.multiselect("Select 2–5 features",filtered_features,default=filtered_features[:3])
            sample_size = st.slider("Sample size", 200, len(df), 500)
            log_toggle = st.toggle("Log scale SalePrice")
            if not (2 <= len(selected) <= 5):
                st.warning("Select between 2 and 5 features.")
            else:
                cols_to_plot = ["SalePrice"] + selected
                plot_df = df[cols_to_plot].dropna().copy()
                plot_df = plot_df.sample(sample_size, random_state=42)
                if log_toggle:
                    plot_df["SalePrice"] = np.log1p(plot_df["SalePrice"])
                fig = px.scatter_matrix(
                    plot_df,
                    dimensions=cols_to_plot,
                    opacity=0.5
                )
                fig.update_traces(
                    diagonal_visible=True,
                    showupperhalf=False,
                    marker=dict(size=4)
                )
                fig.update_layout(height=700)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("### Correlation with SalePrice")
                corr = plot_df.corr()["SalePrice"].sort_values(ascending=False)
                st.dataframe(corr)