import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="T20 World Cup – Team Strength Analytics",
    layout="wide",
    page_icon="🏏"
)

plt.style.use("dark_background")

# ---- GLOBAL COMPACT VISUAL SETTINGS ----
plt.rcParams.update({
    "figure.figsize": (6, 4),
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9
})

# ------------------ MAIN TITLE ------------------
st.title("🏏 T20 World Cup – Team Strength & Match Impact Analytics")
st.markdown(
    "Data-driven comparison of top T20 teams using **batting**, **bowling**, "
    "**all-round depth**, and **match simulations**."
)

st.divider()

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Team Comparison.csv")
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
    return df

df = load_data()

# ------------------ SIDEBAR ------------------
st.sidebar.header("🎯 Team Selection")

teams = st.sidebar.multiselect(
    "Select Teams",
    options=df["country"].unique(),
    default=df["country"].unique()
)

filtered_df = df[df["country"].isin(teams)]

# ------------------ FEATURE ENGINEERING ------------------
filtered_df["batting_impact"] = filtered_df["batting_sr"] * filtered_df["batting_average"]
filtered_df["bowling_impact"] = filtered_df["wickets"] * (1 / filtered_df["economy"])
filtered_df["allrounder_index"] = (
    0.6 * filtered_df["batting_impact"] +
    0.4 * filtered_df["bowling_impact"]
)

# ------------------ COLOR PALETTE ------------------
COLORS = {
    "batting": "#4DA8DA",
    "bowling": "#F39C12",
    "allround": "#2ECC71",
    "radar": "#E91E63"
}

# ------------------ HELPER FUNCTIONS ------------------
def team_bar_chart(df, metric, title, color):
    team_df = (
        df.groupby("country")[metric]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(team_df["country"], team_df[metric], color=color, alpha=0.9)

    best_team = team_df.iloc[0]["country"]

    for bar, team in zip(bars, team_df["country"]):
        if team == best_team:
            bar.set_edgecolor("white")
            bar.set_linewidth(2)

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{bar.get_height():.1f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    return fig


def pressure_adjustment(strength, overs_left):
    if overs_left <= 5:
        return strength * 1.15
    elif overs_left <= 10:
        return strength * 1.05
    return strength


def radar_chart(team_df):
    categories = ["Batting", "Bowling", "All-Round"]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))

    for _, row in team_df.iterrows():
        values = [
            row["batting_impact"],
            row["bowling_impact"],
            row["allrounder_index"]
        ]
        values += values[:1]

        ax.plot(angles, values, linewidth=2, label=row["country"])
        ax.fill(angles, values, alpha=0.12)

    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_title("Team Strength Radar Comparison", pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))

    return fig

# ------------------ TABS ------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "🏏 Batting", "🎯 Bowling", "⚖ All-Rounders", "🎮 Simulation"]
)

# ================== OVERVIEW ==================
with tab1:
    st.subheader("🏆 Overall Team Strength (On Paper)")

    team_strength = (
        filtered_df.groupby("country")[["batting_impact", "bowling_impact"]]
        .mean()
        .reset_index()
    )

    ar_team = (
        filtered_df[filtered_df["role"].str.lower().str.contains("all", na=False)]
        .groupby("country")["allrounder_index"]
        .mean()
        .reset_index()
    )

    team_strength = team_strength.merge(ar_team, on="country", how="left").fillna(0)

    team_strength["power_index"] = (
        0.5 * team_strength["batting_impact"] +
        0.3 * team_strength["bowling_impact"] +
        0.2 * team_strength["allrounder_index"]
    )

    best_team = team_strength.sort_values("power_index", ascending=False).iloc[0]

    st.success(
        f"📌 **{best_team['country']}** appears strongest on paper based on "
        "balanced batting, bowling, and all-round depth."
    )

    st.dataframe(team_strength, use_container_width=True)

    st.subheader("🕸 Team Balance Radar")
    st.pyplot(radar_chart(team_strength))

# ================== BATTING ==================
with tab2:
    st.subheader("🏏 Batting Strength Comparison")
    st.pyplot(
        team_bar_chart(
            filtered_df,
            "batting_impact",
            "Team Batting Impact",
            COLORS["batting"]
        )
    )

# ================== BOWLING ==================
with tab3:
    st.subheader("🎯 Bowling Strength Comparison")
    st.pyplot(
        team_bar_chart(
            filtered_df,
            "bowling_impact",
            "Team Bowling Impact",
            COLORS["bowling"]
        )
    )

# ================== ALL-ROUNDERS ==================
with tab4:
    st.subheader("⚖ All-Rounder Impact Distribution")

    ar_df = filtered_df[
        filtered_df["role"].str.lower().str.contains("all", na=False)
    ]

    if ar_df.empty:
        st.warning("No all-rounders available for selected teams.")
    else:
        sizes = (
            (ar_df["allrounder_index"] - ar_df["allrounder_index"].min()) /
            (ar_df["allrounder_index"].max() - ar_df["allrounder_index"].min() + 1e-6)
        ) * 250 + 60

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(
            ar_df["batting_impact"],
            ar_df["bowling_impact"],
            s=sizes,
            alpha=0.75,
            color=COLORS["allround"],
            edgecolor="white"
        )

        top_ar = ar_df.sort_values("allrounder_index", ascending=False).head(3)

        for _, row in top_ar.iterrows():
            ax.annotate(
                row["player_name"],
                (row["batting_impact"], row["bowling_impact"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold"
            )

        ax.set_xlabel("Batting Impact")
        ax.set_ylabel("Bowling Impact")
        ax.set_title("All-Rounder Batting vs Bowling Impact")
        ax.grid(alpha=0.3)

        st.pyplot(fig)

    st.subheader("🏏 Team All-Round Strength")
    st.pyplot(
        team_bar_chart(
            ar_df,
            "allrounder_index",
            "Team All-Round Impact",
            COLORS["allround"]
        )
    )

# ================== SIMULATION ==================
with tab5:
    st.subheader("🎮 Match Win Probability Simulator")

    col1, col2 = st.columns(2)
    team1 = col1.selectbox("Team 1", teams)
    team2 = col2.selectbox("Team 2", [t for t in teams if t != team1])

    overs_left = st.slider("Overs Left", 1, 20, 10)

    s1 = team_strength.loc[team_strength["country"] == team1, "power_index"].values[0]
    s2 = team_strength.loc[team_strength["country"] == team2, "power_index"].values[0]

    adj1 = pressure_adjustment(s1, overs_left)
    adj2 = pressure_adjustment(s2, overs_left)

    prob1 = adj1 / (adj1 + adj2)
    prob2 = 1 - prob1

    st.metric(f"{team1} Win Probability", f"{prob1*100:.1f}%")
    st.metric(f"{team2} Win Probability", f"{prob2*100:.1f}%")

st.divider()
st.header("🏆 Final Verdict (On Paper)")

st.success(
    f"**{best_team['country']}** emerges as the strongest team on paper, "
    f"driven by a balanced combination of batting firepower and bowling efficiency."
)