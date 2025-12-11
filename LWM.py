# last_mile_interactive_sim_final.py
# Single-file Streamlit app: improved interactive last-mile simulation
# - Continuous playback loop (one-click animation)
# - Adjustable Playback speed
# - Layman's explanation added

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

st.set_page_config(layout="wide", page_title="Last-mile Interactive Simulation")

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
# UI: Title & Explanation
# -------------------------
st.title("🚚 Last-mile Delivery — Interactive Simulation")

# --- NEW: EXPLANATION SECTION ---
with st.expander("ℹ️ How to read this simulation (Click to expand)", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **1. The Scenario**
        We are simulating a delivery fleet in Delhi. Vehicles start at "Micro Hubs" (warehouses), visit assigned customers to drop off packages, and return to the hub.
        
        **2. The Visuals**
        * 🟠 **Orange Dots:** Delivery vehicles moving in real-time.
        * ➖ **Orange Lines:** The planned path for the day.
        * 🗺️ **The Map:** A realistic view of the delivery zone.
        """)
    with col2:
        st.markdown("""
        **3. What affects the simulation?**
        * **Traffic:** Vehicles move slower during peak hours (e.g., 9 AM - 11 AM) based on the "Traffic Profile" data.
        * **Service Time:** When a vehicle reaches a customer, it stops for a few minutes (default: 4 mins) to hand over the package.
        
        **4. The Goal**
        This tool visualizes fleet efficiency. Watch the **"Deliveries Completed"** counter to see how quickly the fleet serves the orders.
        """)

st.markdown("---")

# -------------------------
# SIDEBAR CONTROLS
# -------------------------
st.sidebar.header("1. Simulation Parameters")
vehicle_count = st.sidebar.slider("Number of vehicles", 1, max(1,len(vehicles_df)), value=min(3, len(vehicles_df)))
base_speed = st.sidebar.slider("Base speed (km/h)", 10, 60, value=int(BASE_SPEED_DEFAULT))
service_time = st.sidebar.slider("Service time per delivery (min)", 1, 10, value=int(SERVICE_TIME_DEFAULT))
seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)

# Playback Speed Control
st.sidebar.markdown("---")
st.sidebar.header("2. Animation Settings")
speed_option = st.sidebar.select_slider("Playback speed", options=["1x", "2x", "5x", "10x", "20x", "Max"], value="5x")

# Convert speed option to sleep time
speed_map = {
    "1x": 0.5,
    "2x": 0.25,
    "5x": 0.1,
    "10x": 0.05,
    "20x": 0.01,
    "Max": 0.001
}
sleep_delay = speed_map[speed_option]

st.sidebar.markdown("---")
run_sim = st.sidebar.button("Run Simulation", type="primary")

st.sidebar.write(f"Total Orders: {len(orders_df)}")
st.sidebar.write(f"Total Vehicles Available: {len(vehicles_df)}")

# -------------------------
# ROUTING / ASSIGNMENT
# -------------------------
def assign_round_robin(orders, vehicles_sel):
    vids = list(vehicles_sel["vehicle_id"].astype(str))
    assignment = {v: [] for v in vids}
    for i, (_, row) in enumerate(orders.iterrows()):
        vid = vids[i % len(vids)]
        assignment[vid].append(row.to_dict())
    return assignment

# -------------------------
# TRACE GENERATION
# -------------------------
def generate_traces(assignment, vehicles_sel, hubs, traffic, speed_kmph, service_min, timestep_min):
    pos_traces = {}   
    event_log = []    
    sim_end = 0

    for vid, orders in assignment.items():
        vrow = vehicles_sel[vehicles_sel["vehicle_id"] == vid]
        if vrow.empty: vrow = vehicles_sel.iloc[[0]]
        vrow = vrow.iloc[0]
        
        # Get start hub
        hub_id = vrow["start_hub"]
        hub_row = hubs[hubs[hubs.columns[0]] == hub_id]
        if hub_row.empty:
            hub_lat, hub_lon = float(hubs.iloc[0]["lat"]), float(hubs.iloc[0]["lon"])
        else:
            hub_lat, hub_lon = float(hub_row.iloc[0]["lat"]), float(hub_row.iloc[0]["lon"])

        cur_lat, cur_lon = hub_lat, hub_lon
        t = 0
        trace = [{"minute": t, "lat": cur_lat, "lon": cur_lon, "order_id": "", "status": "at_hub_start"}]

        for order in orders:
            dest_lat, dest_lon = float(order["order_lat"]), float(order["order_lon"])
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
# Run button logic (Preparation)
# -------------------------
if run_sim:
    random.seed(int(seed)); np.random.seed(int(seed))
    vehicles_sel = vehicles_df.head(vehicle_count).copy()
    assignment = assign_round_robin(orders_df, vehicles_sel)

    with st.spinner("Calculating routes and traffic impact..."):
        pos_traces, event_log, sim_end = generate_traces(assignment, vehicles_sel, hubs_df, traffic_df, base_speed, service_time, TIME_STEP_MIN)

    # Store in session state
    st.session_state["pos_traces"] = pos_traces
    st.session_state["event_log"] = event_log
    st.session_state["sim_end"] = sim_end
    st.session_state["vehicles_sel"] = vehicles_sel
    st.session_state["assignment"] = assignment
    # We reset current minute to 0 on a new run
    st.session_state["current_minute"] = 0
    
    st.success(f"Simulation ready! Total shift duration: {sim_end} minutes.")

# -------------------------
# MAIN ANIMATION UI
# -------------------------
if "pos_traces" in st.session_state:
    pos_traces = st.session_state["pos_traces"]
    event_log = st.session_state["event_log"]
    sim_end = st.session_state["sim_end"]
    
    # 1. Start Button
    col_btn, col_txt = st.columns([1, 4])
    start_btn = col_btn.button("▶ Start Animation")
    if not start_btn:
        col_txt.info("Click Start to watch the vehicles move.")
    
    # 2. Placeholders for dynamic content
    #    We create these EMPTY slots now, and update them inside the loop later.
    kpi_placeholder = st.empty()
    map_placeholder = st.empty()
    table_placeholder = st.empty()
    status_text = st.empty()

    # Pre-calculate Map Center
    all_lats = []
    all_lons = []
    for trace in pos_traces.values():
        for p in trace:
            all_lats.append(p["lat"]); all_lons.append(p["lon"])
    center_lat = float(np.mean(all_lats)) if all_lats else 28.6139
    center_lon = float(np.mean(all_lons)) if all_lons else 77.2090

    # Route Lines (Static background layer)
    route_records = []
    for vid, trace in pos_traces.items():
        path = [[p["lon"], p["lat"]] for p in trace]
        route_records.append({"vehicle_id": vid, "path": path})

    path_layer = pdk.Layer(
        "PathLayer",
        data=route_records,
        get_path="path",
        get_width=3,
        get_color=[255, 140, 0, 100], # Orange with transparency
        pickable=False
    )

    # FUNCTION TO RENDER ONE FRAME
    def render_frame(minute):
        # A. Filter positions for this minute
        rows = []
        for vid, trace in pos_traces.items():
            # Get the point at 'minute', or the last known point if simulation ended for this vehicle
            valid_pts = [p for p in trace if p["minute"] <= minute]
            if valid_pts:
                pt = valid_pts[-1]
                rows.append({
                    "vehicle_id": vid, 
                    "lat": pt["lat"], 
                    "lon": pt["lon"], 
                    "order_id": pt["order_id"], 
                    "status": pt["status"]
                })
        pos_df_cur = pd.DataFrame(rows)

        # B. Calculate KPIs up to this minute
        # Filter event log for events that finished <= minute
        current_events = [e for e in event_log if e["finish_minute"] <= minute]
        count_del = len(current_events)
        dist_so_far = sum([e["distance_km"] for e in current_events])
        sim_time = SIM_START + timedelta(minutes=int(minute))

        # C. Render KPIs
        with kpi_placeholder.container():
            c1, c2, c3 = st.columns(3)
            c1.metric("🕒 Simulated Clock", sim_time.strftime("%Y-%m-%d %H:%M"))
            c2.metric("📦 Deliveries Completed", f"{count_del}")
            c3.metric("📏 Distance Covered", f"{dist_so_far:.2f} km")

        # D. Render Map
        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=pos_df_cur,
            get_position=["lon", "lat"],
            get_radius=150,  # size of vehicle dot
            get_fill_color=[255, 140, 0, 255], # Solid Orange
            get_line_color=[0, 0, 0],
            get_line_width=20,
            pickable=True,
        )

        view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=11, pitch=0)
        
        # Tooltip
        tooltip = {"text": "Vehicle: {vehicle_id}\nStatus: {status}\nOrder: {order_id}"}

        deck = pdk.Deck(
            layers=[path_layer, scatter_layer],
            initial_view_state=view_state,
            tooltip=tooltip
        )
        map_placeholder.pydeck_chart(deck)

        # E. Render Table
        with table_placeholder.container():
            st.markdown("### Vehicle Status (Live)")
            st.dataframe(pos_df_cur, hide_index=True, use_container_width=True)

    # -------------------------
    # ANIMATION LOOP
    # -------------------------
    if start_btn:
        # Loop from 0 to sim_end
        for minute in range(0, sim_end + 1):
            render_frame(minute)
            time.sleep(sleep_delay) # Control speed here
        
        status_text.success("✅ Simulation Complete!")
    else:
        # Show specific static frame (start) if not playing
        render_frame(0)

else:
    st.info("👈 **Step 1:** Configure parameters in the Sidebar.\n\n👇 **Step 2:** Click **Run Simulation**.")
