#!/usr/bin/env python3
import json
import pandas as pd
import streamlit as st
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, PULP_CBC_CMD, LpStatus

# === CONFIG ===
SALARY_CAP = 60000
ROSTER_SLOTS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1, "DEF": 1}
TOTAL_ROSTER = sum(v for k, v in ROSTER_SLOTS.items() if k != "FLEX") + ROSTER_SLOTS["FLEX"]

# === UTILS ===
def canonical_position(pos_raw):
    if not pos_raw:
        return None
    s = str(pos_raw).upper()
    if "QB" in s:
        return "QB"
    if "RB" in s:
        return "RB"
    if "WR" in s:
        return "WR"
    if "TE" in s:
        return "TE"
    if any(k in s for k in ("DEF","DST","D/ST","D","TEAM")):
        return "DEF"
    return s

def extract_points(p, use_fppg=False, stats_for="current"):
    if not use_fppg:
        return float(p.get("projected_fantasy_points", {}).get("projected_fantasy_points", 0.0))

    stats_for_lower = stats_for.lower()
    if stats_for_lower == "current":
        val = p.get("fppg")
        if val is not None:
            return float(val)

    recent_stats = p.get("recent_games_played_stats", {})
    key_map = {"last3": "LAST_3", "last5": "LAST_5", "last10": "LAST_10"}
    key = key_map.get(stats_for_lower)
    if key and key in recent_stats:
        val = recent_stats[key].get("fppg")
        if val is not None:
            return float(val)

    return float(p.get("projected_fantasy_points", {}).get("projected_fantasy_points", 0.0))

def check_availability(df_pool):
    reasons = []
    for pos in ("QB","RB","WR","TE","DEF"):
        if df_pool[df_pool["position"]==pos].shape[0] < ROSTER_SLOTS[pos]:
            reasons.append(f"Not enough {pos}")
    if df_pool.shape[0] < TOTAL_ROSTER:
        reasons.append("Not enough total players")
    return reasons

# === STREAMLIT APP ===
st.title("FanDuel NFL Lineup Optimizer")

# --- File upload ---
uploaded_file = st.file_uploader("Upload Players JSON", type=["json"])
if uploaded_file:
    data = json.load(uploaded_file)
    raw_players = data.get("players") or data.get("_items") or []

    players = []
    for p in raw_players:
        if not p.get("draftable", False):
            continue
        try:
            proj = float(p.get("projected_fantasy_points", {}).get("projected_fantasy_points",0.0))
            sal = int(p.get("salary"))
        except:
            continue
        pos = canonical_position(p.get("position"))
        players.append({
            "id": p.get("id"),
            "name": f"{p.get('first_name','')} {p.get('last_name','')}".strip(),
            "position": pos,
            "salary": sal,
            "points": proj,
            "dvp_rank": p.get("dvp_rank"),
            "injury_status": p.get("injury_status"),
            "team": p.get("team"),
            "raw": p,
        })
    df_all = pd.DataFrame(players)

    # --- Filters ---
    st.sidebar.header("Filters")
    top_n = st.sidebar.number_input("Top N Lineups", min_value=1, max_value=20, value=5, step=1)
    min_dvp = st.sidebar.number_input("Min DVP Rank", value=0, step=1)
    min_salary = st.sidebar.number_input("Min Salary", value=0, step=100)
    max_salary = st.sidebar.number_input("Max Salary", value=0, step=100)
    use_fppg = st.sidebar.checkbox("Use FPPG", value=False)
    stats_for = st.sidebar.selectbox("Stats For", ["current","last3","last5","last10"])
    exclude_injured = st.sidebar.checkbox("Exclude Injured Players", value=False)

    # --- Include/Exclude players (collapsible) ---
    with st.expander("Include/Exclude Players", expanded=False):
        st.markdown("Select players to **force include** or **force exclude** in lineups.")
        include_ids = []
        exclude_ids = []
        if not df_all.empty:
            df_all_sorted = df_all.sort_values("name")
            include_cols = st.columns(2)
            for idx, row in df_all_sorted.iterrows():
                if include_cols[0].checkbox(f"Include {row['name']} ({row['position']} ${row['salary']} {row['injury_status']})", key=f"inc_{row['id']}"):
                    include_ids.append(row["id"])
                if include_cols[1].checkbox(f"Exclude {row['name']} ({row['position']} ${row['salary']} {row['injury_status']})", key=f"exc_{row['id']}"):
                    exclude_ids.append(row["id"])

    # --- Run Optimization ---
    if st.button("Optimize Lineups"):
        df = df_all.copy()
        # Apply filters
        if min_dvp>0:
            df = df[(df["dvp_rank"].isna()) | (df["dvp_rank"] >= min_dvp)]
        if min_salary>0:
            df = df[(df["salary"]>=min_salary) | (df["position"]=="DEF")]
        if max_salary>0:
            df = df[df["salary"]<=max_salary]
        if exclude_injured:
            df = df[df["injury_status"].isna() | (df["injury_status"].str.upper()=="")]

        # Apply exclude
        df = df[~df["id"].isin(exclude_ids)]

        # Check feasibility
        reasons = check_availability(df)
        if reasons:
            st.error("No feasible lineup: " + "/".join(reasons))
        else:
            # Optimization
            solver = PULP_CBC_CMD(msg=False)
            df_pool = df.reset_index(drop=True).copy()
            df_pool["points"] = [extract_points(p["raw"], use_fppg, stats_for) for _, p in df_pool.iterrows()]
            x_vars = {i: LpVariable(f"p_{i}", cat="Binary") for i in df_pool.index}

            # Base problem and constraints
            base_prob = LpProblem("NFL_DFS", LpMaximize)
            base_prob += lpSum(x_vars[i] * df_pool.at[i,"points"] for i in df_pool.index)
            base_prob += lpSum(x_vars[i] * df_pool.at[i,"salary"] for i in df_pool.index) <= SALARY_CAP
            for pos in ("QB","DEF"):
                base_prob += lpSum(x_vars[i] for i in df_pool.index if df_pool.at[i,"position"]==pos) == ROSTER_SLOTS[pos]
            for pos in ("RB","WR","TE"):
                base_prob += lpSum(x_vars[i] for i in df_pool.index if df_pool.at[i,"position"]==pos) >= ROSTER_SLOTS[pos]
            base_prob += lpSum(x_vars.values()) == TOTAL_ROSTER

            # Force include
            include_indices = df_pool.index[df_pool["id"].isin(include_ids)].tolist()
            for i in include_indices:
                x_vars[i].setInitialValue(1)
                x_vars[i].fixValue()

            # Generate Top N lineups
            all_lineups = []
            prob = base_prob
            for n in range(top_n):
                prob.solve(solver)
                if LpStatus[prob.status] != "Optimal":
                    break
                chosen = [i for i in x_vars if x_vars[i].value()==1]
                chosen_df = df_pool.loc[chosen].copy()
                # FLEX detection
                counts = chosen_df["position"].value_counts().to_dict()
                flex_pos = None
                for pos in ("RB","WR","TE"):
                    if counts.get(pos,0) > ROSTER_SLOTS[pos]:
                        flex_pos = pos
                        break
                chosen_df["slot"] = chosen_df["position"]
                if flex_pos:
                    candidates = chosen_df[chosen_df["position"]==flex_pos].sort_values("points")
                    if not candidates.empty:
                        chosen_df.at[candidates.index[0], "slot"] = "FLEX"
                slot_order = {"QB":0,"RB":1,"WR":2,"TE":3,"FLEX":4,"DEF":5}
                chosen_df["slot_sort"] = chosen_df["slot"].map(slot_order).fillna(99)
                chosen_df = chosen_df.sort_values(["slot_sort","points"], ascending=[True,False]).drop(columns=["slot_sort"])
                all_lineups.append(chosen_df)
                # Add exclusion constraint to avoid duplicates
                prob += lpSum([x_vars[i] for i in chosen]) <= len(chosen)-1

            # Display lineups
            for idx, lineup in enumerate(all_lineups,1):
                st.subheader(f"=== LINEUP {idx} ===")
                st.dataframe(lineup[["name","slot","position","salary","points","dvp_rank","injury_status"]])
                st.write(f"Total Salary: {lineup['salary'].sum()} | Projected Points: {lineup['points'].sum():.2f}")
