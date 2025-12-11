import streamlit as st
import pandas as pd
import numpy as np
import math
import random
import pydeck as pdk
import time
from datetime import datetime, timedelta

# --------------------------------------------
# CONFIGURATION
# --------------------------------------------
EXCEL_PATH = "last_mile_dataset_with_coords.xlsx"
SIM_START = datetime(2025, 2, 1, 8, 0)

BASE_SPEED_KMPH = 30
SERVICE_TIME_MIN = 4
TIME_STEP_MIN = 1  # 1-minute resolution recommended

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

st.set_page_config(layout="wide", page_title="Interactive Last Mile Simulation (India)")


# --------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c


def traffic_multiplier_for_minute(traffic_df, minute_since_start):
    tm = (SIM_START + timedelta(minutes=int(minute_since_start))).strftime("%H:%M")
    for _, r in traffic_df.iterrows():
        slot = str(r["time_slot"])
        mult = float(r["speed_multiplier"])
        try:
            s, e = slot.split("-")
            if s <= tm <= e:
                return max(0.2, mult + random.uniform(-0.02, 0.02))
        except:
            continue
    return 1.0


# --------------------------------------------
# LOAD DATA
# --------------------------------------------
@st.cache_data
def load_data(path):
    return pd.read_excel(path, sheet_name=None)

sheets = load_data(EXCEL_PATH)
orders_df = sheets["Orders"]
vehicles_df = sheets["Vehicles"]
hubs_df = sheets["Micro_Hubs"]
traffic_df = sheets["Traffic_Profile"]


# --------------------------------------------
# ROUTING (SIMPLE ROUND ROBIN)
# --------------------------------------------
def assign_orders(orders_df, vehicles_df_sel):
    vehicles = list(vehicles_df_sel["vehicle_id"])
    assignment = {v: [] for v in vehicles}
    for i, (_, row) in enumerate(orders_df.iterrows()):
        assignment[vehicles[i % len(vehicles)]].append(row.to_dict())
    return assignment


# --------------------------------------------
# SIMULATION ENGINE (GENERATES POSITION TRACES)
# --------------------------------------------
def generate_traces(assignment, vehicles_df_sel, hubs_df, traffic_df, speed, service_time, timestep):
    pos_traces = {}
    event_log = []
    overall_end = 0

    for vid, orders in assignment.items():
        vrow = vehicles_df_sel[vehicles_df_sel["vehicle_id"] == vid].iloc[0]
        hub_id = vrow["start_hub"]
        hub_row = hubs_df[hubs_df["hub_id"] == hub_id].iloc[0]

        cur_lat = float(hub_row["lat"])
        cur_lon = float(hub_row["lon"])

        t = 0
        trace = [{"minute": t, "lat": cur_lat, "lon": cur_lon, "order_id": "", "status": "start"}]

        for order in orders:
            dest_lat = float(order["order_lat"])
            dest_lon = float(order["order_lon"])
            oid = order["order_id"]

            dist = haversine(cur_lat, cur_lon, dest_lat, dest_lon)
            mult = traffic_multiplier_for_minute(traffic_df, t)
            travel_min = (dist / speed) * 60 / mult
            travel_min = max(travel_min, timestep)

            steps = max(1, int(travel_min / timestep))
            for s in range(1, steps + 1):
                frac = s / steps
                lat = cur_lat + (dest_lat - cur_lat) * frac
                lon = cur_lon + (dest_lon - cur_lon) * frac
                t += timestep
                trace.append({"minute": int(t), "lat": lat, "lon": lon, "order_id": oid, "status": "moving"})

            for _ in range(service_time):
                t += timestep
                trace.append({"minute": int(t), "lat": dest_lat, "lon": dest_lon, "order_id": oid, "status": "servicing"})

            event_log.append({"vehicle_id": vid, "order_id": oid, "distance_km": dist})
            cur_lat, cur_lon = dest_lat, dest_lon

        # RETURN TO HUB
        dist = haversine(cur_lat, cur_lon, float(hub_row["lat"]), float(hub_row["lon"]))
        mult = traffic_multiplier_for_minute(traffic_df, t)
        travel_min = (dist / speed) * 60 / mult
        travel_min = max(travel_min, timestep)

        steps = max(1, int(travel_min / timestep))
        for s in range(1, steps + 1):
            frac = s / steps
            lat = cur_lat + (hub_row["lat"] - cur_lat) * frac
            lon = cur_lon + (hub_row["lon"] - cur_lon) * frac
            t += timestep
            trace.append({"minute": int(t), "lat": lat, "lon": lon, "order_id": "", "status": "returning"})

        pos_traces[str(vid)] = trace
        overall_end = max(overall_end, t)

    return pos_traces, event_log, int(overall_end)


# --------------------------------------------
# UI LAYOUT HEADER
# --------------------------------------------
st.title("🚚 Interactive Last-Mile Delivery Simulation — Delhi (India)")

st.markdown("""
### **Simulation Overview**
- Vehicles start from hubs  
- Travel straight-line paths  
- Affects speed based on traffic  
- Service time at each order  
- Movement shown with **orange bike icons**  
""")


# --------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------
vehicle_count = st.sidebar.slider("Number of vehicles", 1, len(vehicles_df), 3)
speed_input = st.sidebar.slider("Base Speed (km/h)", 10, 60, BASE_SPEED_KMPH)
service_input = st.sidebar.slider("Service time (min)", 1, 10, SERVICE_TIME_MIN)
seed_input = st.sidebar.number_input("Random Seed", value=RANDOM_SEED)

run_btn = st.sidebar.button("Run Simulation")


# --------------------------------------------
# RUN SIMULATION
# --------------------------------------------
if run_btn:
    random.seed(seed_input)
    np.random.seed(seed_input)

    vehicles_df_sel = vehicles_df.head(vehicle_count)
    assignment = assign_orders(orders_df, vehicles_df_sel)

    st.success("Simulation running...")

    pos_traces, event_log, sim_end = generate_traces(
        assignment, vehicles_df_sel, hubs_df, traffic_df,
        speed_input, service_input, TIME_STEP_MIN
    )

    # Build DataFrame
    rows = []
    for vid, tr in pos_traces.items():
        for p in tr:
            rows.append({
                "vehicle_id": vid,
                "minute": p["minute"],
                "lat": p["lat"],
                "lon": p["lon"],
                "order_id": p["order_id"],
                "status": p["status"]
            })

    pos_df = pd.DataFrame(rows)

    # ------------------------------
    # Animation State Handling
    # ------------------------------
    if "current_time" not in st.session_state:
        st.session_state.current_time = 0

    # Time Slider
    t = st.slider("Animation Time", 0, sim_end, st.session_state.current_time)

    # Play Button
    if st.button("▶ Play Animation"):
        for i in range(t, sim_end + 1):
            st.session_state.current_time = i
            time.sleep(0.12)
            st.experimental_rerun()

    st.session_state.current_time = t

    # ------------------------------
    # MAP LAYER — ORANGE BIKE ICON
    # ------------------------------
    df_t = pos_df[pos_df["minute"] == t]

    # ICON DATA
    icon_url = "https://i.imgur.com/YZ9c9xQ.png"  # bright orange bike PNG

    df_t["icon_data"] = [{
        "url": icon_url,
        "width": 128,
        "height": 128,
        "anchorY": 128
    }] * len(df_t)

    icon_layer = pdk.Layer(
        "IconLayer",
        data=df_t,
        get_icon="icon_data",
        get_size=4,
        size_scale=18,
        get_position=["lon", "lat"],
    )

    # PATH LAYER (ROUTE)
    route_data = []
    for vid, trace in pos_traces.items():
        coords = [[p["lon"], p["lat"]] for p in trace]
        route_data.append({"path": coords})

    path_layer = pdk.Layer(
        "PathLayer",
        data=route_data,
        get_path="path",
        get_width=4,
        width_min_pixels=2,
        get_color=[255, 140, 0]
    )

    # View centered on Delhi
    view = pdk.ViewState(
        latitude=28.6139,
        longitude=77.2090,
        zoom=11,
        pitch=0
    )

    deck = pdk.Deck(
        layers=[path_layer, icon_layer],
        initial_view_state=view
    )

    st.pydeck_chart(deck)

    st.markdown("### Event Log Preview")
    st.dataframe(pd.DataFrame(event_log).head(20))

else:
    st.info("Click **Run Simulation** from sidebar to begin.")
