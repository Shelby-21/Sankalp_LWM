"""
last_mile_streamlit_sim.py
Single-file Streamlit app that:
 - loads last_mile_dataset_with_coords.xlsx
 - runs a simple SimPy-based delivery simulation (straight-line travel, traffic multipliers)
 - constructs per-minute vehicle position traces (interpolated)
 - shows interactive PyDeck map with time slider to animate vehicles
 - shows KPIs and event table

Usage in Jupyter:
 - Run `!pip install simpy pandas streamlit pydeck numpy openpyxl` (one-time)
 - Paste this file into a cell, run it, then save as last_mile_streamlit_sim.py
 - Push to GitHub and deploy via Streamlit Cloud (or run locally with `streamlit run last_mile_streamlit_sim.py`)
"""

import simpy
import pandas as pd
import numpy as np
import math
import random
import pydeck as pdk
import streamlit as st
from datetime import datetime, timedelta

# -------------------------
# CONFIG / PARAMETERS
# -------------------------
EXCEL_PATH = "last_mile_dataset_with_coords.xlsx"  # your uploaded file name
SIM_START = datetime(2025, 2, 1, 8, 0)   # reference start (not strictly required, used for display)
BASE_SPEED_KMPH = 30.0      # base vehicle speed (km/h) - average across fleet for straight-line
SERVICE_TIME_MIN = 4        # service/delivery time at customer in minutes
TIME_STEP_MIN = 1           # resolution for animation traces (minutes)
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

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
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def find_traffic_multiplier(traffic_df, current_minute):
    """
    traffic_df has columns: time_slot (like "08:00-10:00") and speed_multiplier.
    current_minute is minutes since start (we convert to HH:MM).
    """
    sim_time = (SIM_START + timedelta(minutes=int(current_minute))).strftime("%H:%M")
    for _, r in traffic_df.iterrows():
        slot = str(r["time_slot"])
        mult = float(r["speed_multiplier"])
        try:
            start_s, end_s = slot.split("-")
            if start_s <= sim_time <= end_s:
                # small random noise to avoid flatness
                return max(0.1, mult + random.uniform(-0.03, 0.03))
        except Exception:
            continue
    return 1.0

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data(path=EXCEL_PATH):
    x = pd.read_excel(path, sheet_name=None)
    # expected sheets: Orders, Customers, Micro_Hubs, Lockers, Vehicles, Traffic_Profile
    # We'll ensure column names exist
    return x

data = load_data(EXCEL_PATH)

orders_df = data["Orders"].copy()
customers_df = data["Customers"].copy()
hubs_df = data["Micro_Hubs"].copy()
vehicles_df = data["Vehicles"].copy()
traffic_df = data["Traffic_Profile"].copy()

# Confirm important columns exist:
# Orders: order_id, customer_id, order_lat, order_lon
# Vehicles: vehicle_id, type, max_capacity_kg, start_hub
# Hubs: hub_id, lat, lon
# Traffic_Profile: time_slot, speed_multiplier

# -------------------------
# ROUTING: simple assignment
# -------------------------
def assign_orders_to_vehicles(orders_df, vehicles_df):
    """
    Simple round-robin assignment across vehicles (keeps things deterministic & simple).
    Returns a dict: vehicle_id -> list of order rows (as dicts).
    """
    vehicles = list(vehicles_df["vehicle_id"].astype(str))
    if len(vehicles) == 0:
        raise ValueError("No vehicles in vehicles_df")
    assign = {v: [] for v in vehicles}
    for i, (_, row) in enumerate(orders_df.iterrows()):
        v = vehicles[i % len(vehicles)]
        assign[v].append(row.to_dict())
    return assign

# -------------------------
# SIMPY ENV (minute resolution)
# -------------------------
class SimpleSimEnv:
    def __init__(self, assignment, vehicles_df, hubs_df, traffic_df):
        self.env = simpy.Environment()
        self.assignment = assignment          # dict vehicle -> list(order dicts)
        self.vehicles_df = vehicles_df
        self.hubs_df = hubs_df
        self.traffic_df = traffic_df
        self.position_traces = {}             # vehicle_id -> list of dicts {minute, lat, lon, order_id, status}
        self.event_log = []                   # event-level log
        self.sim_duration_min = 0

    def get_hub_coord(self, hub_id):
        h = self.hubs_df[self.hubs_df[self.hubs_df.columns[0]] == hub_id]
        if not h.empty:
            return float(h.iloc[0]["lat"]), float(h.iloc[0]["lon"])
        # fallback: first hub
        return float(self.hubs_df.iloc[0]["lat"]), float(self.hubs_df.iloc[0]["lon"])

    def run_vehicle(self, vehicle_id, orders_for_vehicle):
        """
        SimPy process per vehicle: depart from start_hub, visit each assigned order in sequence,
        return to hub. We record per-minute interpolated positions.
        """
        # get vehicle row
        vrow = self.vehicles_df[self.vehicles_df[self.vehicles_df.columns[0]] == vehicle_id]
        if vrow.empty:
            vrow = self.vehicles_df.iloc[[0]]
            hub_id = vrow.iloc[0]["start_hub"]
        else:
            vrow = vrow.iloc[0]
            hub_id = vrow["start_hub"]

        hub_lat, hub_lon = self.get_hub_coord(hub_id)
        cur_lat, cur_lon = hub_lat, hub_lon
        t_min = 0  # minutes since sim start for this vehicle

        trace = []
        # mark start
        trace.append({"minute": t_min, "lat": cur_lat, "lon": cur_lon, "order_id": None, "status": "at_hub"})

        for order in orders_for_vehicle:
            # destination coordinates
            dest_lat = float(order["order_lat"])
            dest_lon = float(order["order_lon"])
            order_id = order["order_id"]

            # distance km
            dist_km = haversine(cur_lat, cur_lon, dest_lat, dest_lon)
            # get traffic multiplier based on current minute
            mult = find_traffic_multiplier(self.traffic_df, t_min)
            # travel time in minutes = (dist / speed_kmph) * 60 / multiplier
            travel_minutes = (dist_km / BASE_SPEED_KMPH) * 60.0 / mult
            # ensure at least 1 minute
            travel_minutes = max(1.0, travel_minutes)

            # interpolate per TIME_STEP_MIN minutes
            steps = max(1, int(math.ceil(travel_minutes / TIME_STEP_MIN)))
            for s in range(1, steps + 1):
                frac = s / steps
                lat = cur_lat + (dest_lat - cur_lat) * frac
                lon = cur_lon + (dest_lon - cur_lon) * frac
                t_min += TIME_STEP_MIN
                trace.append({"minute": t_min, "lat": lat, "lon": lon, "order_id": order_id, "status": "enroute"})

            # arrive and service time
            service_minutes = SERVICE_TIME_MIN
            for s in range(1, int(service_minutes) + 1):
                t_min += 1
                trace.append({"minute": t_min, "lat": dest_lat, "lon": dest_lon, "order_id": order_id, "status": "servicing"})

            # event log
            self.event_log.append({
                "vehicle_id": vehicle_id,
                "order_id": order_id,
                "arrival_minute": t_min - int(service_minutes),
                "finish_minute": t_min,
                "distance_km": dist_km
            })

            # update current pos
            cur_lat, cur_lon = dest_lat, dest_lon

        # return to hub
        dist_km = haversine(cur_lat, cur_lon, hub_lat, hub_lon)
        mult = find_traffic_multiplier(self.traffic_df, t_min)
        travel_minutes = (dist_km / BASE_SPEED_KMPH) * 60.0 / mult
        travel_minutes = max(1.0, travel_minutes)
        steps = max(1, int(math.ceil(travel_minutes / TIME_STEP_MIN)))
        for s in range(1, steps + 1):
            frac = s / steps
            lat = cur_lat + (hub_lat - cur_lat) * frac
            lon = cur_lon + (hub_lon - cur_lon) * frac
            t_min += TIME_STEP_MIN
            trace.append({"minute": t_min, "lat": lat, "lon": lon, "order_id": None, "status": "returning"})

        # final position at hub
        trace.append({"minute": t_min, "lat": hub_lat, "lon": hub_lon, "order_id": None, "status": "at_hub_end"})
        self.position_traces[vehicle_id] = trace
        # update sim duration if needed
        if t_min > self.sim_duration_min:
            self.sim_duration_min = int(t_min)

    def run(self):
        # start processes for all vehicles
        for vid, orders in self.assignment.items():
            # schedule each vehicle as a process (no concurrency blocking needed here)
            self.env.process(self.run_vehicle_proc_wrapper(vid, orders))
        # run until all processes finish
        self.env.run(until=1000000)  # large until - but processes manage their own time via t_min
        # Note: Because our vehicle processes are synchronous calculations (no yields), env.run completes immediately.
        # We do not rely on simpy's time progression for this simple implementation.

    def run_vehicle_proc_wrapper(self, vid, orders):
        # Wrapper to call the run_vehicle synchronously but still satisfy SimPy process API
        def _proc(env):
            self.run_vehicle(vid, orders)
            yield env.timeout(0)
        return _proc(self.env)

# -------------------------
# PREPARE ASSIGNMENT & RUN SIM
# -------------------------
st.set_page_config(layout="wide", page_title="Last-mile Sim & Map Animation")
st.title("🚚 Last-mile Delivery — SimPy + PyDeck Map Animation")

st.markdown("This app runs a simple delivery simulation and shows vehicle positions on a map. "
            "Vehicles travel in straight lines from their start hub to assigned customers and back. "
            "Use the slider to animate by minute.")

# Controls
vehicle_count = st.sidebar.slider("Vehicles (use 1..)", min_value=1, max_value=int(max(1, len(vehicles_df))), value=min(3, max(1, len(vehicles_df))))
seed_input = st.sidebar.number_input("Random seed", value=RANDOM_SEED, step=1)
run_button = st.sidebar.button("Run Simulation")

# Show dataset info
st.sidebar.write("Dataset summary:")
st.sidebar.write(f"Orders: {len(orders_df)} | Vehicles: {len(vehicles_df)} | Hubs: {len(hubs_df)}")

if run_button:
    RANDOM_SEED = int(seed_input)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # We will select up to vehicle_count vehicles from vehicles_df
    selected_vehicles = vehicles_df.iloc[:vehicle_count].copy()
    # assign orders to these selected vehicles in round-robin
    assignment = assign_orders_to_vehicles(orders_df, selected_vehicles)

    # Build simulation environment
    sim = SimpleSimEnv(assignment, selected_vehicles, hubs_df, traffic_df)
    # run simulation (this uses our synchronous per-vehicle routines)
    sim.run()

    # Build position points DataFrame for animation / plotting
    records = []
    for vid, trace in sim.position_traces.items():
        for p in trace:
            records.append({
                "vehicle_id": str(vid),
                "minute": int(p["minute"]),
                "lat": float(p["lat"]),
                "lon": float(p["lon"]),
                "order_id": p["order_id"] if p["order_id"] is not None else "",
                "status": p["status"]
            })
    pos_df = pd.DataFrame.from_records(records)
    if pos_df.empty:
        st.error("No position data generated.")
    else:
        max_minute = int(pos_df["minute"].max())
        st.sidebar.write(f"Simulation duration (minutes): {max_minute}")

        # KPI panel
        total_distance = sum([e["distance_km"] for e in sim.event_log])
        total_deliveries = len(sim.event_log)
        avg_time_per_delivery = None
        if total_deliveries > 0:
            avg_time_per_delivery = np.mean([e["finish_minute"] - e["arrival_minute"] for e in sim.event_log])

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Deliveries (simulated)", total_deliveries)
        col2.metric("Total Distance (approx km)", f"{total_distance:.2f}")
        col3.metric("Avg service time (min)", f"{avg_time_per_delivery:.1f}" if avg_time_per_delivery else "N/A")

        # Time slider for animation
        t = st.slider("Animation time (minute)", min_value=0, max_value=max_minute, value=0, step=1)

        # For the selected time t, pick latest position per vehicle at or before t
        df_t = pos_df[pos_df["minute"] <= t].sort_values(["vehicle_id", "minute"]).groupby("vehicle_id").tail(1)

        # Map: lines = route polylines per vehicle; points = vehicle current positions
        # Prepare route lines (full route)
        route_lines = []
        for vid, trace in sim.position_traces.items():
            coords = [[p["lon"], p["lat"]] for p in trace]
            route_lines.append({"vehicle_id": str(vid), "path": coords})

        # PyDeck layers
        # 1) route lines (solid)
        route_layer = pdk.Layer(
            "ArcLayer",
            data=route_lines,
            get_source_position="path[0]",
            get_target_position="path[-1]",
            get_width=3,
            get_tilt=15,
            pickable=True,
            auto_highlight=True,
            get_source_color=[0, 128, 200],
            get_target_color=[200, 30, 0],
        )

        # 2) vehicle points (current)
        points_df = df_t.copy()
        if points_df.empty:
            # fallback: show hubs
            hubs_points = hubs_df.rename(columns={hubs_df.columns[1]: "lat", hubs_df.columns[2]: "lon"})
            points_layer = pdk.Layer(
                "ScatterplotLayer",
                data=hubs_points,
                get_position=["lon", "lat"],
                get_radius=80,
                get_fill_color=[0, 200, 0],
                pickable=True
            )
            initial_view = pdk.ViewState(latitude=hubs_points.iloc[0]["lat"], longitude=hubs_points.iloc[0]["lon"], zoom=12)
            deck = pdk.Deck(layers=[points_layer, route_layer], initial_view_state=initial_view)
            st.pydeck_chart(deck)
        else:
            points_layer = pdk.Layer(
                "ScatterplotLayer",
                data=points_df,
                get_position=["lon", "lat"],
                get_fill_color=[255, 140, 0],
                get_radius=80,
                pickable=True
            )
            # create a center for initial view: mean of points
            center_lat = float(points_df["lat"].mean())
            center_lon = float(points_df["lon"].mean())
            initial_view = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=12)
            deck = pdk.Deck(layers=[route_layer, points_layer], initial_view_state=initial_view)
            st.pydeck_chart(deck)

        # show event log and pos_df slices
        st.write("### Event Log (sample)")
        st.dataframe(pd.DataFrame(sim.event_log).head(50))

        st.write("### Vehicle Positions (sample)")
        st.dataframe(pos_df.sort_values(["vehicle_id", "minute"]).head(200))

        # Optionally allow CSV export
        csv_positions = pos_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download position traces CSV", data=csv_positions, file_name="position_traces.csv", mime="text/csv")
        csv_events = pd.DataFrame(sim.event_log).to_csv(index=False).encode("utf-8")
        st.download_button("Download event log CSV", data=csv_events, file_name="event_log.csv", mime="text/csv")

else:
    st.info("Adjust controls in the sidebar and click 'Run Simulation' to start.")
    st.write("You can also preview some data.")

    st.write("Orders (first 5 rows):")
    st.dataframe(orders_df.head())

    st.write("Vehicles (all):")
    st.dataframe(vehicles_df)

    st.write("Micro hubs (all):")
    st.dataframe(hubs_df)
