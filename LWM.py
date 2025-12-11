"""
last_mile_interactive_sim.py

Interactive last-mile simulation + map animation (Option A: bike emoji)
- Uses your Excel: last_mile_dataset_with_coords.xlsx
- 1-minute resolution by default
- Streamlit + PyDeck visualization
- SimPy-style synchronous per-vehicle simulation (generates interpolated traces)
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
import random
import pydeck as pdk
import time
from datetime import datetime, timedelta

# -----------------------
# CONFIGURATION / PARAMS
# -----------------------
EXCEL_PATH = "last_mile_dataset_with_coords.xlsx"  # ensure file in repo
SIM_START = datetime(2025, 2, 1, 8, 0)             # displayed baseline start
BASE_SPEED_KMPH = 30.0                             # default base speed km/h (modifiable from UI)
SERVICE_TIME_MIN = 4                               # service time per stop in minutes
TIME_STEP_MIN = 1                                  # animation resolution in minutes (1 recommended)
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

st.set_page_config(layout="wide", page_title="Interactive Last-mile Sim (India)")

# -----------------------
# HELPERS
# -----------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def traffic_multiplier_for_minute(traffic_df, minute_since_start):
    # Convert minute to HH:MM
    tm = (SIM_START + timedelta(minutes=int(minute_since_start))).strftime("%H:%M")
    for _, r in traffic_df.iterrows():
        slot = str(r["time_slot"])
        mult = float(r["speed_multiplier"])
        try:
            s, e = slot.split("-")
            if s <= tm <= e:
                return max(0.15, mult + random.uniform(-0.03, 0.03))
        except Exception:
            continue
    return 1.0

# -----------------------
# LOAD DATA
# -----------------------
@st.cache_data
def load_all(path=EXCEL_PATH):
    sheets = pd.read_excel(path, sheet_name=None)
    return sheets

try:
    sheets = load_all(EXCEL_PATH)
except Exception as e:
    st.error(f"Failed to load {EXCEL_PATH}. Make sure the file is in repo root. Error: {e}")
    st.stop()

orders_df = sheets["Orders"].copy()
customers_df = sheets["Customers"].copy()
hubs_df = sheets["Micro_Hubs"].copy()
vehicles_df = sheets["Vehicles"].copy()
traffic_df = sheets["Traffic_Profile"].copy()

# ensure correct column names exist
required_orders_cols = ["order_id", "customer_id", "order_lat", "order_lon"]
for c in required_orders_cols:
    if c not in orders_df.columns:
        st.error(f"Missing column '{c}' in Orders sheet. Found columns: {list(orders_df.columns)}")
        st.stop()

# -----------------------
# UI: top description / challenge / method
# -----------------------
st.title("🚚 Interactive Last-mile Delivery Simulation — India (Delhi)")

with st.expander("Project overview (click to expand)", expanded=True):
    st.markdown("""
**Challenge:** Urban last-mile delivery is expensive and inefficient due to traffic congestion, suboptimal consolidation and routing.  
**Goal:** Build an interactive simulation + visualization to test fleet sizes, traffic scenarios and consolidation strategies, and to show stakeholder impact.
    
**Method chosen:** Discrete-event delivery simulation (minute-resolution) implemented in Python, with interactive visualization using Streamlit + PyDeck.  
**Visualization style (Option A):** Bike emoji represents each vehicle; vehicles move along straight-line interpolated paths between hubs and customers.  
    """)

# -----------------------
# Sidebar controls
# -----------------------
st.sidebar.header("Simulation Controls")
vehicle_count = st.sidebar.slider("Number of vehicles to use (select first N vehicles from Vehicles sheet)", 1, max(1, len(vehicles_df)), value=min(3, max(1, len(vehicles_df))))
base_speed_kmph = st.sidebar.slider("Base vehicle speed (km/h)", 10, 60, value=int(BASE_SPEED_KMPH))
service_time_min = st.sidebar.slider("Service time per delivery (min)", 1, 10, value=SERVICE_TIME_MIN)
time_step_min = st.sidebar.selectbox("Animation resolution", options=["1 minute", "10 seconds"], index=0)
if time_step_min == "1 minute":
    TIME_STEP = 1
else:
    TIME_STEP = 1/6  # 10 seconds -> 1/6 minute

seed_input = st.sidebar.number_input("Random seed", min_value=0, value=RANDOM_SEED, step=1)

run_button = st.sidebar.button("Run Simulation")

st.sidebar.markdown("---")
st.sidebar.write(f"Orders loaded: {len(orders_df)}")
st.sidebar.write(f"Vehicles available: {len(vehicles_df)}")
st.sidebar.write(f"Hubs: {len(hubs_df)}")

# -----------------------
# Assignment / routing (simple but deterministic)
# -----------------------
def assign_round_robin(orders_df, vehicles_selected):
    vehicles = list(vehicles_selected[vehicles_selected.columns[0]].astype(str))
    assign = {v: [] for v in vehicles}
    for i, (_, r) in enumerate(orders_df.iterrows()):
        v = vehicles[i % len(vehicles)]
        assign[v].append(r.to_dict())
    return assign

# -----------------------
# Simulation engine (synchronous trace generation)
# -----------------------
def generate_traces(assignment, vehicles_df_sel, hubs_df, traffic_df, base_speed_kmph, service_time_min, time_step_min):
    """
    For each vehicle, generate per-minute (or per TIME_STEP) position trace and event log.
    Returns:
      position_traces: dict vehicle_id -> list of {minute, lat, lon, order_id, status}
      event_log: list of {vehicle_id, order_id, arrival_minute, finish_minute, distance_km}
      sim_duration_min: int
    """
    position_traces = {}
    event_log = []
    sim_duration_min = 0

    for vid, orders in assignment.items():
        # vehicle row
        vrow = vehicles_df_sel[vehicles_df_sel[vehicles_df_sel.columns[0]] == vid]
        if vrow.empty:
            vrow = vehicles_df_sel.iloc[[0]]
        vrow = vrow.iloc[0]
        hub_id = vrow["start_hub"]
        hub_row = hubs_df[hubs_df[hubs_df.columns[0]] == hub_id]
        if hub_row.empty:
            hub_lat = float(hubs_df.iloc[0]["lat"]); hub_lon = float(hubs_df.iloc[0]["lon"])
        else:
            hub_lat = float(hub_row.iloc[0]["lat"]); hub_lon = float(hub_row.iloc[0]["lon"])

        cur_lat, cur_lon = hub_lat, hub_lon
        t_min = 0
        trace = []
        trace.append({"minute": t_min, "lat": cur_lat, "lon": cur_lon, "order_id": "", "status": "at_hub_start"})

        for order in orders:
            dest_lat = float(order["order_lat"])
            dest_lon = float(order["order_lon"])
            order_id = order["order_id"]
            # compute distance
            dist_km = haversine(cur_lat, cur_lon, dest_lat, dest_lon)
            mult = traffic_multiplier_for_minute(traffic_df, t_min)
            travel_time_min = (dist_km / base_speed_kmph) * 60.0 / mult
            travel_time_min = max( max( (time_step_min if time_step_min>=1 else 1/6) , travel_time_min), 0.5)

            # number of steps
            steps = max(1, int(math.ceil(travel_time_min / TIME_STEP)))
            for s in range(1, steps+1):
                frac = s/steps
                lat = cur_lat + (dest_lat - cur_lat) * frac
                lon = cur_lon + (dest_lon - cur_lon) * frac
                t_min += TIME_STEP
                trace.append({"minute": int(round(t_min)), "lat": lat, "lon": lon, "order_id": order_id, "status": "enroute"})

            # service time
            service_steps = max(1, int(math.ceil(service_time_min / TIME_STEP)))
            for s in range(service_steps):
                t_min += TIME_STEP
                trace.append({"minute": int(round(t_min)), "lat": dest_lat, "lon": dest_lon, "order_id": order_id, "status": "servicing"})

            event_log.append({"vehicle_id": vid, "order_id": order_id, "arrival_minute": int(round(t_min - service_time_min)), "finish_minute": int(round(t_min)), "distance_km": dist_km})
            cur_lat, cur_lon = dest_lat, dest_lon

        # return to hub
        dist_km = haversine(cur_lat, cur_lon, hub_lat, hub_lon)
        mult = traffic_multiplier_for_minute(traffic_df, t_min)
        travel_time_min = (dist_km / base_speed_kmph) * 60.0 / mult
        travel_time_min = max( (time_step_min if time_step_min>=1 else 1/6), travel_time_min)
        steps = max(1, int(math.ceil(travel_time_min / TIME_STEP)))
        for s in range(1, steps+1):
            frac = s/steps
            lat = cur_lat + (hub_lat - cur_lat) * frac
            lon = cur_lon + (hub_lon - cur_lon) * frac
            t_min += TIME_STEP
            trace.append({"minute": int(round(t_min)), "lat": lat, "lon": lon, "order_id": "", "status": "returning"})

        trace.append({"minute": int(round(t_min)), "lat": hub_lat, "lon": hub_lon, "order_id": "", "status": "at_hub_end"})
        position_traces[str(vid)] = trace
        sim_duration_min = max(sim_duration_min, int(round(t_min)))

    return position_traces, event_log, sim_duration_min

# -----------------------
# Run simulation on button
# -----------------------
if run_button:
    # set params
    random.seed(int(seed_input)); np.random.seed(int(seed_input))
    base_speed = float(base_speed_kmph)
    service_time = int(service_time_min)
    TIME_STEP = TIME_STEP  # already set

    # select first N vehicles
    vehicles_df_sel = vehicles_df.head(vehicle_count).copy()
    assignment = assign_round_robin(orders_df, vehicles_df_sel)

    with st.spinner("Running simulation and generating traces..."):
        pos_traces, event_log, sim_duration_min = generate_traces(assignment, vehicles_df_sel, hubs_df, traffic_df, base_speed, service_time, TIME_STEP)

    # build DataFrame of positions
    records = []
    for vid, trace in pos_traces.items():
        for p in trace:
            records.append({"vehicle_id": vid, "minute": p["minute"], "lat": p["lat"], "lon": p["lon"], "order_id": p["order_id"], "status": p["status"]})
    pos_df = pd.DataFrame.from_records(records)

    # KPIs
    total_deliveries = len(event_log)
    total_distance = sum([e["distance_km"] for e in event_log])
    avg_service = np.mean([e["finish_minute"]-e["arrival_minute"] for e in event_log]) if total_deliveries>0 else None

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Deliveries (simulated)", total_deliveries)
    col2.metric("Total Distance (approx km)", f"{total_distance:.2f}")
    col3.metric("Avg service duration (min)", f"{avg_service:.1f}" if avg_service else "N/A")

    # Time slider + Play
    t_min = st.slider("Animation time (minute)", min_value=0, max_value=sim_duration_min, value=0, step=1)
    play = st.button("Play animation")

    # handle play via session state loop (basic)
    if "play" not in st.session_state:
        st.session_state.play = False
    if play:
        st.session_state.play = True
    if st.sidebar.button("Pause"):
        st.session_state.play = False

    # Autoplay loop (will rerun page while playing)
    if st.session_state.play:
        for tt in range(t_min, sim_duration_min+1):
            t_min = tt
            time.sleep(0.08)  # short sleep to animate (0.08s per frame)
            st.session_state.current_time = t_min
            st.experimental_rerun()  # rerun script to update slider & map

    # use slider value or session_state
    if "current_time" in st.session_state:
        t_min = st.session_state.current_time if st.session_state.current_time <= sim_duration_min else t_min

    # build df for map at time t_min
    df_t = pos_df[pos_df["minute"] <= t_min].sort_values(["vehicle_id", "minute"]).groupby("vehicle_id").tail(1).reset_index(drop=True)

    # Build route lines for visualization (full path per vehicle)
    route_lines = []
    for vid, trace in pos_traces.items():
        coords = [[p["lon"], p["lat"]] for p in trace]
        route_lines.append({"vehicle_id": vid, "path": coords})

    # PyDeck layers
    # 1) PathLayer for route polylines
    path_layer = pdk.Layer(
        "PathLayer",
        data=route_lines,
        get_path="path",
        get_width=4,
        pickable=True,
        width_min_pixels=2,
    )

    # 2) TextLayer for bike emoji at vehicle positions
    if not df_t.empty:
        df_t["text"] = "🏍️"
        text_layer = pdk.Layer(
            "TextLayer",
            data=df_t,
            get_position=["lon", "lat"],
            get_text="text",
            get_size=32,
            get_angle=0,
            get_text_anchor="'middle'",
            get_alignment_baseline="'center'",
        )
    else:
        text_layer = None

    # initial view centered on mean position within India (Delhi area)
    if not pos_df.empty:
        lat0 = float(pos_df["lat"].mean())
        lon0 = float(pos_df["lon"].mean())
    else:
        lat0 = 28.6139; lon0 = 77.2090  # Delhi fallback

    view = pdk.ViewState(latitude=lat0, longitude=lon0, zoom=12, pitch=0)

    layers = [path_layer]
    if text_layer is not None:
        layers.append(text_layer)

    deck = pdk.Deck(layers=layers, initial_view_state=view, tooltip={"text": "Vehicle: {vehicle_id}"})
    st.pydeck_chart(deck)

    # Display event log and positions
    st.write("### Event log (sample)")
    st.dataframe(pd.DataFrame(event_log).head(50))

    st.write("### Position traces (sample)")
    st.dataframe(pos_df.sort_values(["vehicle_id", "minute"]).head(200))

    # Downloads
    st.download_button("Download event log CSV", data=pd.DataFrame(event_log).to_csv(index=False).encode("utf-8"), file_name="event_log.csv")
    st.download_button("Download position traces CSV", data=pos_df.to_csv(index=False).encode("utf-8"), file_name="position_traces.csv")

else:
    st.info("Configure parameters in the sidebar and click 'Run Simulation' to start.")
    st.write("Orders preview:")
    st.dataframe(orders_df.head())
    st.write("Vehicles preview:")
    st.dataframe(vehicles_df.head())
