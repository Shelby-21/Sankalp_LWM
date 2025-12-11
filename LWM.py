# last_mile_interactive_sim_final_v2.py
# Final interactive last-mile simulation (auto-play with pause + speed control + explanation + summary)
# Uses: last_mile_dataset_with_coords.xlsx in repo root

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
DEFAULT_BASE_SPEED = 30.0               # km/h
DEFAULT_SERVICE_MIN = 4                 # minutes per delivery
TIME_STEP_MIN = 1                       # simulation granularity in minutes

# Real-time baseline chosen by you (Option B): 1 simulated minute = 0.20 real seconds
BASE_REAL_SECONDS_PER_SIM_MIN = 0.20

# Icon URL fallback (we use scatter orange dots; icon URL kept if you prefer icons)
ICON_URL = "https://i.imgur.com/YZ9c9xQ.png"

st.set_page_config(layout="wide", page_title="Last-mile Interactive Simulation — Final V2")

# -------------------------
# HELPERS
# -------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
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
    st.error(f"Could not load {EXCEL_PATH}. Place the file in repo root. Error: {e}")
    st.stop()

orders_df = sheets["Orders"].copy()
vehicles_df = sheets["Vehicles"].copy()
hubs_df = sheets["Micro_Hubs"].copy()
traffic_df = sheets["Traffic_Profile"].copy()

# Validate columns
required_orders_cols = ["order_id","customer_id","order_lat","order_lon"]
for c in required_orders_cols:
    if c not in orders_df.columns:
        st.error(f"Orders sheet missing required column: {c}")
        st.stop()

# -------------------------
# UI: header & method for layman
# -------------------------
st.title("🚚 Last-mile Delivery — Interactive Simulation (Delhi)")

st.markdown("""
**What you are seeing (simple explanation):**

- We simulate delivery vehicles that start from micro-hubs in Delhi, visit assigned customers, and return to their hubs.
- Each vehicle follows a planned path (straight-line approximation between stops). The orange line shows the full planned route for that vehicle.
- The orange dot represents the vehicle's current location as time advances.
- Travel time depends on distance, a base speed (km/h) you set, and traffic multipliers taken from the Traffic_Profile sheet (time-of-day effects).
- Increase the number of vehicles or base speed to see how total completion time and average delivery time change.

**Why this is useful:** managers can quickly test "what if" scenarios — e.g., increase base speed (faster vehicles or priority lanes) or increase fleet size — and observe the impact on delivery completion time and distance.
""")

# -------------------------
# SIDEBAR controls
# -------------------------
st.sidebar.header("Simulation Controls")
vehicle_count = st.sidebar.slider("Number of vehicles (first N vehicles from Vehicles sheet)", 1, max(1, len(vehicles_df)), value=min(3, len(vehicles_df)))
base_speed = st.sidebar.slider("Base speed (km/h)", min_value=10, max_value=60, value=int(DEFAULT_BASE_SPEED))
service_time = st.sidebar.slider("Service time per delivery (min)", min_value=1, max_value=10, value=int(DEFAULT_SERVICE_MIN))
seed_val = st.sidebar.number_input("Random seed (keeps runs reproducible)", min_value=0, value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.write(f"Orders: {len(orders_df)}")
st.sidebar.write(f"Vehicles in sheet: {len(vehicles_df)}")
st.sidebar.write(f"Hubs: {len(hubs_df)}")

# Playback speed control (affects real-time sleep)
playback_choice = st.sidebar.selectbox("Playback speed", options=["0.5× (slow)","1× (normal)","1.25×","1.5×","2×"], index=1)
playback_map = {"0.5× (slow)":0.5, "1× (normal)":1.0, "1.25×":1.25, "1.5×":1.5, "2×":2.0}
playback_factor = playback_map[playback_choice]

run_btn = st.sidebar.button("Run Simulation")

# -------------------------
# ROUTING (simple round-robin deterministic)
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
    """
    Returns:
      pos_traces: dict vehicle_id -> list of {minute, lat, lon, order_id, status}
      event_log: list of deliveries with distances
      sim_end_min: int
    """
    pos_traces = {}
    event_log = []
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
                frac = s / steps
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
        dist_back = haversine(cur_lat, cur_lon, hub_lat, hub_lon)
        mult = traffic_mult(traffic, t)
        travel_min = (dist_back / speed_kmph) * 60.0 / mult
        travel_min = max(travel_min, timestep_min)
        steps = max(1, int(math.ceil(travel_min / timestep_min)))
        for s in range(1, steps+1):
            frac = s / steps
            lat = cur_lat + (hub_lat - cur_lat) * frac
            lon = cur_lon + (hub_lon - cur_lon) * frac
            t += timestep_min
            trace.append({"minute": int(t), "lat": lat, "lon": lon, "order_id": "", "status": "returning"})

        trace.append({"minute": int(t), "lat": hub_lat, "lon": hub_lon, "order_id": "", "status": "at_hub_end"})
        pos_traces[str(vid)] = trace
        sim_end = max(sim_end, int(t))

    return pos_traces, event_log, int(sim_end)

# -------------------------
# When user clicks Run Simulation
# -------------------------
if run_btn:
    random.seed(int(seed_val)); np.random.seed(int(seed_val))
    vehicles_sel = vehicles_df.head(vehicle_count).copy()
    assignment = assign_round_robin(orders_df, vehicles_sel)

    with st.spinner("Generating simulation traces..."):
        pos_traces, event_log, sim_end = generate_traces(assignment, vehicles_sel, hubs_df, traffic_df, float(base_speed), int(service_time), TIME_STEP_MIN)

    # store in session_state
    st.session_state["pos_traces"] = pos_traces
    st.session_state["event_log"] = event_log
    st.session_state["sim_end"] = sim_end
    st.session_state["vehicles_sel"] = vehicles_sel
    st.session_state["assignment"] = assignment
    st.session_state["current_minute"] = 0
    st.session_state["play"] = True   # auto-start
    st.session_state["playback_factor"] = playback_factor
    st.success("Simulation ready — autoplay started. Use Pause to stop.")

# -------------------------
# If a simulation exists, show controls + map + explanation + summary
# -------------------------
if "pos_traces" in st.session_state:
    pos_traces = st.session_state["pos_traces"]
    event_log = st.session_state["event_log"]
    sim_end = st.session_state["sim_end"]
    vehicles_sel = st.session_state["vehicles_sel"]
    playback_factor = st.session_state.get("playback_factor", playback_factor)

    # KPIs & live clock
    total_deliveries = len(event_log)
    total_distance = sum([e["distance_km"] for e in event_log])
    avg_service = np.mean([e["finish_minute"] - e["arrival_minute"] for e in event_log]) if total_deliveries>0 else None

    c1, c2, c3, c4 = st.columns([1.2,1.2,1.2,1.4])
    c1.metric("Deliveries (simulated)", total_deliveries)
    c2.metric("Total distance (km)", f"{total_distance:.2f}")
    c3.metric("Avg service (min)", f"{avg_service:.1f}" if avg_service else "N/A")
    current_min = st.session_state.get("current_minute", 0)
    sim_time = SIM_START + timedelta(minutes=int(current_min))
    c4.metric("Simulated time", sim_time.strftime("%Y-%m-%d %H:%M"))

    # Explanation above map (brief layman text)
    st.markdown("### Simulation explanation")
    st.markdown("""
    - **Orange dot** = vehicle current location (updates as time advances).  
    - **Orange line** = vehicle's planned route (hub → customers → return).  
    - Travel time uses base speed (set in sidebar) and traffic multipliers (Traffic_Profile) — so same distance can take longer at peak times.  
    - Use **Pause** to stop autoplay anytime. Change playback speed in the sidebar (0.5× → 2×).
    """)

    # Play / Pause controls
    colp1, colp2, colp3 = st.columns([1,1,8])
    if colp1.button("⏸ Pause"):
        st.session_state["play"] = False
    if colp2.button("▶ Play"):
        st.session_state["play"] = True

    # Build current positions DataFrame (last point <= current_min for each vehicle)
    def build_current_pos(minute):
        rows = []
        for vid, trace in pos_traces.items():
            candidates = [p for p in trace if p["minute"] <= minute]
            if candidates:
                pt = candidates[-1]
            else:
                pt = trace[0]
            rows.append({"vehicle_id": vid, "lat": pt["lat"], "lon": pt["lon"], "status": pt["status"], "order_id": pt["order_id"]})
        return pd.DataFrame(rows)

    pos_df_now = build_current_pos(st.session_state.get("current_minute", 0))

    # Create route records for PathLayer
    route_records = []
    for vid, trace in pos_traces.items():
        coords = [[p["lon"], p["lat"]] for p in trace]
        route_records.append({"vehicle_id": vid, "path": coords})

    # PyDeck layers: PathLayer + ScatterplotLayer (orange dots)
    path_layer = pdk.Layer(
        "PathLayer",
        data=route_records,
        get_path="path",
        get_width=4,
        get_color=[255,140,0],
        pickable=False
    )

    if not pos_df_now.empty:
        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=pos_df_now,
            get_position=["lon","lat"],
            get_radius=120,
            radius_min_pixels=6,
            radius_max_pixels=60,
            get_fill_color=[255,140,0],
            pickable=True,
        )
    else:
        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=[{"lon":77.2090,"lat":28.6139}],
            get_position=["lon","lat"],
            get_radius=120,
            get_fill_color=[255,140,0],
        )

    # Center view to mean of all coords (keeps India/Delhi focus)
    all_lats, all_lons = [], []
    for trace in pos_traces.values():
        for p in trace:
            all_lats.append(p["lat"]); all_lons.append(p["lon"])
    if all_lats:
        center_lat, center_lon = float(np.mean(all_lats)), float(np.mean(all_lons))
    else:
        center_lat, center_lon = 28.6139, 77.2090

    view = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=12, pitch=0)
    deck = pdk.Deck(layers=[path_layer, scatter_layer], initial_view_state=view, tooltip={"text":"Vehicle: {vehicle_id}\nStatus: {status}\nOrder: {order_id}"})

    st.pydeck_chart(deck)

    # Vehicle table + event log
    st.markdown("### Vehicles (current)")
    st.dataframe(pos_df_now)

    st.markdown("### Event log (deliveries)")
    st.dataframe(pd.DataFrame(event_log).sort_values(["vehicle_id"]).reset_index(drop=True).head(200))

    # Autoplay advancement logic (advance by 1 simulated minute each step)
    if st.session_state.get("play", False):
        # compute real seconds per simulated minute per user choice
        base_real = BASE_REAL_SECONDS_PER_SIM_MIN
        sleep_time = base_real / playback_factor  # faster playback -> smaller sleep
        # advance time by 1 minute
        next_min = st.session_state.get("current_minute", 0) + 1
        if next_min > st.session_state["sim_end"]:
            # simulation finished
            st.session_state["play"] = False
            # show summary after completion
            st.success("Simulation completed.")
            # Simulation summary block (vehicle-wise stats)
            st.markdown("## Simulation Summary & Insights")
            st.markdown(f"- **Total simulated minutes:** {st.session_state['sim_end']} min")
            st.markdown(f"- **Total deliveries simulated:** {total_deliveries}")
            st.markdown(f"- **Total distance (approx km):** {total_distance:.2f}")
            # vehicle-wise summary
            veh_stats = []
            for vid, trace in pos_traces.items():
                # compute total path distance from trace by summing segment haversine
                seg_dist = 0.0
                for i in range(1, len(trace)):
                    seg_dist += haversine(trace[i-1]["lat"], trace[i-1]["lon"], trace[i]["lat"], trace[i]["lon"])
                veh_stats.append((vid, seg_dist, trace[-1]["status"]))
            df_vs = pd.DataFrame(veh_stats, columns=["vehicle_id","route_length_km","final_status"]).sort_values("route_length_km", ascending=False)
            st.write("Vehicle summary (route length km, final status)")
            st.dataframe(df_vs)
            # Short observations (automated)
            st.markdown("### Observations")
            if total_deliveries > 0:
                avg_finish = np.mean([e["finish_minute"] - e["arrival_minute"] for e in event_log])
                st.markdown(f"- Average finish time per delivery (service window): **{avg_finish:.1f} minutes**")
                st.markdown("- Recommendation: Increasing base speed or adding vehicles reduces overall completion time. Try changing Base speed or Number of vehicles and re-run.")
            else:
                st.markdown("- No deliveries recorded.")
        else:
            st.session_state["current_minute"] = next_min
            time.sleep(sleep_time)
            st.experimental_rerun()

    # If paused or not playing, we still show summary if finished
    if not st.session_state.get("play", False) and st.session_state.get("current_minute",0) >= st.session_state.get("sim_end",0):
        st.markdown("## Simulation Summary & Insights (completed)")
        st.markdown(f"- **Total simulated minutes:** {st.session_state.get('sim_end',0)} min")
        st.markdown(f"- **Total deliveries:** {total_deliveries}")
        st.markdown(f"- **Total distance (approx km):** {total_distance:.2f}")

        # vehicle-wise summary as above
        veh_stats = []
        for vid, trace in pos_traces.items():
            seg_dist = 0.0
            for i in range(1, len(trace)):
                seg_dist += haversine(trace[i-1]["lat"], trace[i-1]["lon"], trace[i]["lat"], trace[i]["lon"])
            veh_stats.append((vid, seg_dist, trace[-1]["status"]))
        df_vs = pd.DataFrame(veh_stats, columns=["vehicle_id","route_length_km","final_status"]).sort_values("route_length_km", ascending=False)
        st.write("Vehicle summary (route length km, final status)")
        st.dataframe(df_vs)

    # Downloads
    st.markdown("### Downloads")
    st.download_button("Download event log (CSV)", data=pd.DataFrame(event_log).to_csv(index=False).encode("utf-8"), file_name="event_log.csv")
    pos_now = build_current_pos(st.session_state.get("current_minute",0))
    st.download_button("Download current positions (CSV)", data=pos_now.to_csv(index=False).encode("utf-8"), file_name="positions_now.csv")

else:
    st.info("Configure parameters and click **Run Simulation** to start. Use the playback speed selector in the sidebar to control animation speed.")
    st.write("Orders preview (first 5 rows):")
    st.dataframe(orders_df.head(5))
    st.write("Vehicles (all):")
    st.dataframe(vehicles_df)
