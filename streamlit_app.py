#!/usr/bin/env python3
import json
import pandas as pd
import streamlit as st
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, PULP_CBC_CMD, LpStatus

# === CONFIG ===
SALARY_CAP = 60000
ROSTER_SLOTS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1, "DEF": 1}
TOTAL_ROSTER = sum(v for k, v in ROSTER_SLOTS.items() if k != "FLEX") + ROSTER_SLOTS["FLEX"]

SINGLE_GAME_SALARY = 60000
SINGLE_GAME_PLAYERS = 6
MVP_MULTIPLIER = 1.5

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

def check_availability(df_pool, single_game_mode=False):
    reasons = []
    if single_game_mode:
        if df_pool.shape[0] < SINGLE_GAME_PLAYERS:
            reasons.append("Not enough total players")
    else:
        for pos in ("QB","RB","WR","TE","DEF"):
            if df_pool[df_pool["position"]==pos].shape[0] < ROSTER_SLOTS[pos]:
                reasons.append(f"Not enough {pos}")
        if df_pool.shape[0] < TOTAL_ROSTER:
            reasons.append("Not enough total players")
    return reasons

def detect_game_type(players):
    positions = [canonical_position(p.get("position")) for p in players if p.get("draftable", False)]
    unique_positions = set(positions)
    # Small player pool or only skill positions → Single Game
    if len(players) <= 75 or unique_positions <= {"RB","WR","TE","FLEX"}:
        return "Single Game"
    return "Standard Game"

def get_team_code(player_team, teams):
    if isinstance(player_team, dict) and "_members" in player_team:
        team_id = player_team["_members"][0]
        for t in teams:
            if str(t.get("id")) == str(team_id):
                return t.get("code")
    return None

# === STREAMLIT APP ===
st.title("FanDuel NFL Lineup Optimizer")

# --- File upload ---
uploaded_file = st.file_uploader("Upload Players JSON", type=["json"])
if uploaded_file:
    data = json.load(uploaded_file)
    raw_players = data.get("players") or data.get("_items") or []
    teams_json = data.get("teams") or []

    # Detect game type
    game_type = detect_game_type(raw_players)
    single_game_mode = (game_type == "Single Game")
    st.sidebar.markdown(f"**Detected Game Type:** {game_type}")

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
        team_code = get_team_code(p.get("team"), teams_json)
        players.append({
            "id": p.get("id"),
            "name": f"{p.get('first_name','')} {p.get('last_name','')}".strip(),
            "position": pos,
            "salary": sal,
            "points": proj,
            "team": team_code,
            "dvp_rank": p.get("dvp_rank"),
            "injury_status": p.get("injury_status"),
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

    # --- Include/Exclude players ---
    with st.expander("Include/Exclude Players", expanded=False):
        st.markdown("Select players to **force include** or **force exclude** in lineups.")
        include_ids = []
        exclude_ids = []
        if not df_all.empty:
            df_all_sorted = df_all.sort_values("name")
            include_cols = st.columns(2)
            for idx, row in df_all_sorted.iterrows():
                if include_cols[0].checkbox(f"Include {row['name']} ({row['position']} {row['team']} ${row['salary']})", key=f"inc_{row['id']}"):
                    include_ids.append(row["id"])
                if include_cols[1].checkbox(f"Exclude {row['name']} ({row['position']} {row['team']} ${row['salary']})", key=f"exc_{row['id']}"):
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
        reasons = check_availability(df, single_game_mode)
        if reasons:
            st.error("No feasible lineup: " + "/".join(reasons))
        else:
            solver = PULP_CBC_CMD(msg=False)
            df_pool = df.reset_index(drop=True).copy()
            df_pool["points"] = [extract_points(p["raw"], use_fppg, stats_for) for _, p in df_pool.iterrows()]
            x_vars = {i: LpVariable(f"p_{i}", cat="Binary") for i in df_pool.index}
            all_lineups = []

            if single_game_mode:
                include_indices = df_pool.index[df_pool["id"].isin(include_ids)].tolist()
                used_lineups = []

                for n in range(top_n):
                    best_lineup = None
                    best_points = -1
                    best_mvp = None

                    # Try each player as MVP
                    for mvp_idx in df_pool.index:
                        temp_vars = {i: LpVariable(f"tmp_{i}", cat="Binary") for i in df_pool.index}
                        temp_prob = LpProblem("NFL_SINGLE_GAME", LpMaximize)

                        # Objective
                        temp_prob += lpSum(
                            temp_vars[i]*(df_pool.at[i,"points"]*MVP_MULTIPLIER if i==mvp_idx else df_pool.at[i,"points"])
                            for i in df_pool.index
                        )
                        # Salary
                        temp_prob += lpSum(
                            temp_vars[i]*(df_pool.at[i,"salary"]*MVP_MULTIPLIER if i==mvp_idx else df_pool.at[i,"salary"])
                            for i in df_pool.index
                        ) <= SINGLE_GAME_SALARY
                        # Number of players
                        temp_prob += lpSum(temp_vars.values()) == SINGLE_GAME_PLAYERS
                        # Include forced players
                        for i in include_indices:
                            temp_vars[i].setInitialValue(1)
                            temp_vars[i].fixValue()
                        # Exclude previous lineups
                        for used in used_lineups:
                            temp_prob += lpSum(temp_vars[i] for i in used) <= SINGLE_GAME_PLAYERS - 1

                        temp_prob.solve(solver)
                        if LpStatus[temp_prob.status] == "Optimal":
                            chosen = [i for i in temp_vars if temp_vars[i].value() == 1]
                            pts = sum(df_pool.at[i,"points"]*MVP_MULTIPLIER if i==mvp_idx else df_pool.at[i,"points"] for i in chosen)
                            if pts > best_points:
                                best_points = pts
                                best_lineup = chosen
                                best_mvp = mvp_idx

                    if best_lineup:
                        chosen_df = df_pool.loc[best_lineup].copy()
                        chosen_df["slot"] = "Player"
                        chosen_df.at[best_mvp,"slot"] = "MVP"
                        chosen_df.at[best_mvp,"points"] *= MVP_MULTIPLIER
                        chosen_df.at[best_mvp,"salary"] *= MVP_MULTIPLIER
                        all_lineups.append(chosen_df)
                        used_lineups.append(best_lineup)

            else:
                # Standard game
                base_prob = LpProblem("NFL_DFS", LpMaximize)
                base_prob += lpSum(x_vars[i] * df_pool.at[i,"points"] for i in df_pool.index)
                base_prob += lpSum(x_vars[i] * df_pool.at[i,"salary"] for i in df_pool.index) <= SALARY_CAP
                for pos in ("QB","DEF"):
                    base_prob += lpSum(x_vars[i] for i in df_pool.index if df_pool.at[i,"position"]==pos) == ROSTER_SLOTS[pos]
                for pos in ("RB","WR","TE"):
                    base_prob += lpSum(x_vars[i] for i in df_pool.index if df_pool.at[i,"position"]==pos) >= ROSTER_SLOTS[pos]
                base_prob += lpSum(x_vars.values()) == TOTAL_ROSTER

                include_indices = df_pool.index[df_pool["id"].isin(include_ids)].tolist()
                for i in include_indices:
                    x_vars[i].setInitialValue(1)
                    x_vars[i].fixValue()

                prob = base_prob
                for n in range(top_n):
                    prob.solve(solver)
                    if LpStatus[prob.status] != "Optimal":
                        break
                    chosen = [i for i in x_vars if x_vars[i].value()==1]
                    chosen_df = df_pool.loc[chosen].copy()
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
                    prob += lpSum([x_vars[i] for i in chosen]) <= len(chosen)-1

            # Display
            for idx, lineup in enumerate(all_lineups,1):
                st.subheader(f"=== LINEUP {idx} ===")
                st.dataframe(lineup[["name","slot","position","team","salary","points","dvp_rank","injury_status"]])
                st.write(f"Total Salary: {lineup['salary'].sum()} | Projected Points: {lineup['points'].sum():.2f}")
