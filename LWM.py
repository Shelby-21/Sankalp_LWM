# last_mile_interactive_sim_final.py
# Single-file Streamlit app: improved interactive last-mile simulation
# - Uses last_mile_dataset_with_coords.xlsx (same file you provided)
# - 1-minute resolution
# - Play / Pause works and map does not disappear
# - No animation-time cursor (removed)
# - Vehicles are visible as orange markers moving over time
# - Live clock, KPIs, event log, short explanation for laymen
# - Keep random seed for reproducibility (you can change it in sidebar)

import streamlit as st
import pandas as pd
import numpy as np
import math
import time
from datetime import datetime, timedelta
import pydeck as pdk
import random

# -------------------------
# CONFIG
# -------------------------
EXCEL_PATH = "last_mile_dataset_with_coords.xlsx"
SIM_START = datetime(2025, 2, 1, 8, 0)   # baseline start shown in clock
BASE_SPEED_DEFAULT = 30.0               # km/h
SERVICE_TIME_DEFAULT = 4                # minutes per delivery
TIME_STEP_MIN = 1                       # simulation minute resolution (1 minute)
ICON_URL = "https://i.imgur.com/YZ9c9xQ.png"  # orange bike (fallback: we also show colored scatter)

st.set_page_config(layout="wide", page_title="Last-mile Interactive Simulation — Final")

# -------------------------
# HELPERS
# -------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def traffic_mult(traffic_df, minute_since_start):
    tm = (SIM_START + timedelta(minutes=int(minute_since_start))).strftime("%H:%M")
    for _, r in traffic_df.iterrows():
        try:
            s,e = str(r["time_slot"]).split("-")
            if s <= tm <= e:
                return max(0.2, float(r["speed_multiplier"]) + random.uniform(-0.02, 0.02))
        except Exception:
            continue
    return 1.0

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_all(path):
    return pd.read_excel(path, sheet_name=None)

try:
    sheets = load_all(EXCEL_PATH)
except Exception as e:
    st.error(f"Could not load {EXCEL_PATH}. Make sure it is in the repo root. Error: {e}")
    st.stop()

orders_df = sheets["Orders"].copy()
vehicles_df = sheets["Vehicles"].copy()
hubs_df = sheets["Micro_Hubs"].copy()
traffic_df = sheets["Traffic_Profile"].copy()

# Basic validation
req_cols = ["order_id","customer_id","order_lat","order_lon"]
for c in req_cols:
    if c not in orders_df.columns:
        st.error(f"Orders sheet missing required column: {c}")
        st.stop()

# -------------------------
# UI: Top description & method (layman's explanation)
# -------------------------
st.title("🚚 Last-mile Delivery — Interactive Simulation (Delhi)")

st.markdown("""
**What this demo shows (simple):**
- We simulate delivery vehicles starting from micro-hubs in Delhi and visiting assigned customers.
- Vehicles move along straight-line paths (approximate routing) using realistic speeds and traffic multipliers.
- You can change fleet size, base speed, service time and random seed, then **Run Simulation**.
- Click **Play** to watch vehicles move. A live clock shows simulated time; KPIs update as deliveries happen.
""")

st.markdown("**Method chosen:** Discrete-event / timeline simulation (minute-by-minute) with interactive visualization. This method helps test how fleet size, speed, and traffic affect delivery time and distance — useful for operations and managerial decisions.")

# -------------------------
# SIDEBAR CONTROLS
# -------------------------
st.sidebar.header("Simulation Controls")
vehicle_count = st.sidebar.slider("Number of vehicles (use first N vehicles from Vehicles sheet)", 1, max(1,len(vehicles_df)), value=min(3, len(vehicles_df)))
base_speed = st.sidebar.slider("Base speed (km/h)", 10, 60, value=int(BASE_SPEED_DEFAULT))
service_time = st.sidebar.slider("Service time per delivery (min)", 1, 10, value=int(SERVICE_TIME_DEFAULT))
seed = st.sidebar.number_input("Random seed (keeps runs reproducible)", min_value=0, value=42, step=1)
run_sim = st.sidebar.button("Run Simulation")

st.sidebar.markdown("---")
st.sidebar.write(f"Orders: {len(orders_df)}")
st.sidebar.write(f"Vehicles total in sheet: {len(vehicles_df)}")
st.sidebar.write(f"Hubs: {len(hubs_df)}")

# -------------------------
# ROUTING / ASSIGNMENT (simple deterministic round-robin)
# -------------------------
def assign_round_robin(orders, vehicles_sel):
    vids = list(vehicles_sel["vehicle_id"].astype(str))
    assignment = {v: [] for v in vids}
    for i, (_, row) in enumerate(orders.iterrows()):
        vid = vids[i % len(vids)]
        assignment[vid].append(row.to_dict())
    return assignment

# -------------------------
# TRACE GENERATION (synchronous)
# -------------------------
def generate_traces(assignment, vehicles_sel, hubs, traffic, speed_kmph, service_min, timestep_min):
    pos_traces = {}   # vid -> list of {minute, lat, lon, order_id, status}
    event_log = []    # events per delivery
    sim_end = 0

    for vid, orders in assignment.items():
        vrow = vehicles_sel[vehicles_sel["vehicle_id"] == vid]
        if vrow.empty:
            vrow = vehicles_sel.iloc[[0]]
        vrow = vrow.iloc[0]
        hub_id = vrow["start_hub"]
        hub_row = hubs[hubs[hubs.columns[0]] == hub_id]
        if hub_row.empty:
            hub_lat = float(hubs.iloc[0]["lat"]); hub_lon = float(hubs.iloc[0]["lon"])
        else:
            hub_lat = float(hub_row.iloc[0]["lat"]); hub_lon = float(hub_row.iloc[0]["lon"])

        cur_lat, cur_lon = hub_lat, hub_lon
        t = 0
        trace = [{"minute": t, "lat": cur_lat, "lon": cur_lon, "order_id": "", "status": "at_hub_start"}]

        for order in orders:
            dest_lat = float(order["order_lat"]); dest_lon = float(order["order_lon"])
            oid = order["order_id"]

            dist = haversine(cur_lat, cur_lon, dest_lat, dest_lon)
            mult = traffic_mult(traffic, t)
            travel_min = (dist / speed_kmph) * 60.0 / mult
            travel_min = max(travel_min, timestep_min)

            steps = max(1, int(math.ceil(travel_min / timestep_min)))
            for s in range(1, steps+1):
                frac = s/steps
                lat = cur_lat + (dest_lat - cur_lat) * frac
                lon = cur_lon + (dest_lon - cur_lon) * frac
                t += timestep_min
                trace.append({"minute": int(t), "lat": lat, "lon": lon, "order_id": oid, "status": "moving"})

            # service time
            for _ in range(service_min):
                t += timestep_min
                trace.append({"minute": int(t), "lat": dest_lat, "lon": dest_lon, "order_id": oid, "status": "servicing"})

            event_log.append({"vehicle_id": vid, "order_id": oid, "arrival_minute": int(t - service_min), "finish_minute": int(t), "distance_km": dist})
            cur_lat, cur_lon = dest_lat, dest_lon

        # return to hub
        dist = haversine(cur_lat, cur_lon, hub_lat, hub_lon)
        mult = traffic_mult(traffic, t)
        travel_min = (dist / speed_kmph) * 60.0 / mult
        travel_min = max(travel_min, timestep_min)
        steps = max(1, int(math.ceil(travel_min / timestep_min)))
        for s in range(1, steps+1):
            frac = s/steps
            lat = cur_lat + (hub_lat - cur_lat) * frac
            lon = cur_lon + (hub_lon - cur_lon) * frac
            t += timestep_min
            trace.append({"minute": int(t), "lat": lat, "lon": lon, "order_id": "", "status": "returning"})

        trace.append({"minute": int(t), "lat": hub_lat, "lon": hub_lon, "order_id": "", "status": "at_hub_end"})
        pos_traces[str(vid)] = trace
        sim_end = max(sim_end, int(t))

    return pos_traces, event_log, sim_end

# -------------------------
# Run button logic
# -------------------------
if run_sim:
    # set seed
    random.seed(int(seed)); np.random.seed(int(seed))

    vehicles_sel = vehicles_df.head(vehicle_count).copy()
    assignment = assign_round_robin(orders_df, vehicles_sel)

    with st.spinner("Running simulation and preparing traces..."):
        pos_traces, event_log, sim_end = generate_traces(assignment, vehicles_sel, hubs_df, traffic_df, base_speed, service_time, TIME_STEP_MIN)

    # Save into session_state for persistence across reruns/Play
    st.session_state["pos_traces"] = pos_traces
    st.session_state["event_log"] = event_log
    st.session_state["sim_end"] = sim_end
    st.session_state["vehicles_sel"] = vehicles_sel
    st.session_state["assignment"] = assignment
    st.session_state["current_minute"] = 0
    st.session_state["play"] = False

    st.success("Simulation prepared. Use Play to animate and observe vehicles moving.")

# -------------------------
# If there is a simulation in session_state, show controls + map
# -------------------------
if "pos_traces" in st.session_state:
    pos_traces = st.session_state["pos_traces"]
    event_log = st.session_state["event_log"]
    sim_end = st.session_state["sim_end"]
    vehicles_sel = st.session_state["vehicles_sel"]

    # top KPI row & live clock
    total_deliveries = len(event_log)
    total_distance = sum([e["distance_km"] for e in event_log])
    avg_service = np.mean([e["finish_minute"] - e["arrival_minute"] for e in event_log]) if total_deliveries > 0 else None

    c1, c2, c3, c4 = st.columns([1.2,1.2,1.2,1.4])
    c1.metric("Total deliveries (simulated)", total_deliveries)
    c2.metric("Total distance (approx km)", f"{total_distance:.2f}")
    c3.metric("Avg service (min)", f"{avg_service:.1f}" if avg_service else "N/A")
    # live clock
    cur_min = st.session_state.get("current_minute", 0)
    sim_time = SIM_START + timedelta(minutes=int(cur_min))
    c4.metric("Simulated clock", sim_time.strftime("%Y-%m-%d %H:%M"))

    # play / pause buttons
    colp1, colp2, colp3 = st.columns([1,1,4])
    if colp1.button("▶ Play"):
        st.session_state["play"] = True
    if colp2.button("⏸ Pause"):
        st.session_state["play"] = False

    # Map + vehicle list + event log
    st.markdown("### Map: vehicles moving from hubs → customers (orange dots).")
    st.markdown("**Legend:** orange dot = vehicle; orange line = full planned route for that vehicle.")

    # Build pos_df for current minute (latest position <= current_minute for each vehicle)
    def build_pos_df_at(minute):
        rows = []
        for vid, trace in pos_traces.items():
            # pick last trace point with minute <= minute
            trace_le = [p for p in trace if p["minute"] <= minute]
            if len(trace_le) == 0:
                pt = trace[0]
            else:
                pt = trace_le[-1]
            rows.append({"vehicle_id": vid, "lat": pt["lat"], "lon": pt["lon"], "order_id": pt["order_id"], "status": pt["status"]})
        return pd.DataFrame(rows)

    pos_df_cur = build_pos_df_at(st.session_state["current_minute"])

    # route lines data
    route_records = []
    for vid, trace in pos_traces.items():
        path = [[p["lon"], p["lat"]] for p in trace]
        route_records.append({"vehicle_id": vid, "path": path})

    # path layer (orange)
    path_layer = pdk.Layer(
        "PathLayer",
        data=route_records,
        get_path="path",
        get_width=3,
        get_color=[255,140,0],
    )

    # scatter layer for vehicles (orange)
    if not pos_df_cur.empty:
        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=pos_df_cur,
            get_position=["lon", "lat"],
            get_radius=120,
            radius_min_pixels=6,
            radius_max_pixels=60,
            get_fill_color=[255,140,0],
            pickable=True,
        )
    else:
        scatter_layer = pdk.Layer("ScatterplotLayer", data=[{"lon":77.2090,"lat":28.6139}], get_position=["lon","lat"], get_radius=100, get_fill_color=[255,140,0])

    # initial view centered on mean of all traces (Delhi area)
    all_lats = []
    all_lons = []
    for trace in pos_traces.values():
        for p in trace:
            all_lats.append(p["lat"]); all_lons.append(p["lon"])
    if all_lats:
        center_lat = float(np.mean(all_lats)); center_lon = float(np.mean(all_lons))
    else:
        center_lat, center_lon = 28.6139, 77.2090

    view = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=12, pitch=0)

    deck = pdk.Deck(layers=[path_layer, scatter_layer], initial_view_state=view, tooltip={"text":"Vehicle: {vehicle_id}\nStatus: {status}\nOrder: {order_id}"})
    map_placeholder = st.empty()
    map_placeholder.pydeck_chart(deck)

    # show vehicle table and event log
    st.markdown("### Vehicles (current locations & status)")
    st.dataframe(pos_df_cur)

    st.markdown("### Event log (deliveries)")
    st.dataframe(pd.DataFrame(event_log).sort_values(["vehicle_id"]).reset_index(drop=True).head(200))

    # autoplay loop logic (advances current_minute while play True)
    if st.session_state.get("play", False):
        # Advance by 1 minute per frame, sleep a bit to animate
        # Stop when reaching sim_end
        next_min = st.session_state.get("current_minute", 0) + 1
        if next_min > st.session_state["sim_end"]:
            st.session_state["play"] = False
        else:
            st.session_state["current_minute"] = next_min
            # small pause for animation (adjust as needed)
            time.sleep(0.12)
            # rerun to update UI
            st.experimental_rerun()

    # Downloads
    st.markdown("### Downloads")
    st.download_button("Download event log (CSV)", data=pd.DataFrame(event_log).to_csv(index=False).encode("utf-8"), file_name="event_log.csv")
    # also export current positions
    st.download_button("Download current positions (CSV)", data=pos_df_cur.to_csv(index=False).encode("utf-8"), file_name="positions_now.csv")

else:
    st.info("No simulation prepared. Configure parameters in the sidebar and click **Run Simulation**.")
    st.write("Orders preview (first 5):")
    st.dataframe(orders_df.head(5))
    st.write("Vehicles preview (all):")
    st.dataframe(vehicles_df)
