"""
River Pollutant Dispersion Model (3D)
=====================================

Description:
------------
This Streamlit application simulates the 3D dispersion of pollutants in a river channel 
using the Advection-Dispersion Equation. It provides an interactive platform to 
visualize how a pollutant cloud evolves over time and space under various flow 
and dispersion conditions.

Capabilities:
-------------
1.  3D Visualization: Uses Plotly for volumetric rendering of pollutant concentration clouds.
2.  Release Modes:
    * Instantaneous (Pulse): Simulates a single dump of pollutant (Gaussian Puff).
    * Continuous (Plume): Simulates a steady discharge over time (Gaussian Plume via integration).
3.  Visualization Scenarios:
    * Space-Time Plots (Time vs Length vs Depth/Breadth).
    * 3D Spatial Snapshot (Length vs Breadth vs Depth at a fixed time).
4.  Interactive Parameters: Real-time adjustment of river geometry, flow velocities (u, v, w), 
    dispersion coefficients (Dx, Dy, Dz), and source terms.

Metadata:
---------
* Author: Subodh Purohit
* Motivation: Dr. Abhradeep Majumder, Ph.D
* Purpose: Educational use only.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

# --- 1. Configuration & Layout Setup ---
st.set_page_config(page_title="River Pollutant Dispersion Model", layout="wide")

st.title("3D River Pollutant Dispersion Model")
st.markdown("""
This application simulates the transport of a pollutant in a river channel using the **Advection-Dispersion Equation**.
You can visualize the concentration $C(x,y,z,t)$ for an **Instantaneous** (Puff) release.
""")

# --- 2. Sidebar Inputs: Simulation Parameters ---
st.sidebar.header("1. River Geometry & Flow")
L = st.sidebar.number_input("River Length (X) [m]", value=100.0, min_value=10.0)
W = st.sidebar.number_input("River Breadth (Y) [m]", value=20.0, min_value=1.0)
H = st.sidebar.number_input("River Depth (Z) [m]", value=5.0, min_value=0.5)

st.sidebar.subheader("Velocities [m/s]")
u = st.sidebar.number_input("Advection u (Flow X)", value=0.5, step=0.1, format="%.2f")
v = st.sidebar.number_input("Advection v (Transverse Y)", value=0.0, step=0.01, format="%.2f")
w_vel = st.sidebar.number_input("Advection w (Vertical Z)", value=0.0, step=0.01, format="%.2f") 

st.sidebar.header("2. Dispersion Coefficients")

# Allow 0.0 input so we can catch it with a custom error message
Dx = st.sidebar.number_input("Dx (Longitudinal) [m²/s]", value=1.0, min_value=0.0, step=0.1, format="%.6f")
Dy = st.sidebar.number_input("Dy (Transverse) [m²/s]", value=0.1, min_value=0.0, step=0.01, format="%.6f")
Dz = st.sidebar.number_input("Dz (Vertical) [m²/s]", value=0.01, min_value=0.0, step=0.01, format="%.6f")

# Validation Check: Stop if any dispersion coefficient is 0
if Dx == 0 or Dy == 0 or Dz == 0:
    st.sidebar.error("⚠️ Dispersion Coefficients (Dx, Dy, Dz) cannot be zero, please see the theory tab for reference.")
    st.error("Please set non-zero values for Dispersion Coefficients (Dx, Dy, Dz) in the sidebar to proceed.")
    st.stop()

st.sidebar.header("3. Source Term")
# Removed Release Type Selector and Continuous logic
M = st.sidebar.number_input("Total Mass M [kg]", value=10.0, step=1.0)

st.sidebar.subheader("Source Location (Release Point)")
x0 = st.sidebar.slider("x₀ (Length)", 0.0, L, 5.0)
y0 = st.sidebar.slider("y₀ (Breadth)", 0.0, W, W/2)
z0 = st.sidebar.slider("z₀ (Depth)", 0.0, H, H/2)


# --- 3. Calculation Core (Physics Engine) ---

def calculate_concentration_instantaneous(t, x, y, z, M, u, v, w, Dx, Dy, Dz, x0, y0, z0):
    """
    Analytical solution for Instantaneous Point Source in 3D.
    Uses vector-safe operations to handle both scalar and array 't' inputs.
    """
    # Safe handling for t to avoid division by zero
    t_safe = np.maximum(t, 1e-6)
    
    # Note: Dx, Dy, Dz are guaranteed > 0 by the sidebar check, so we don't need silent max() here.
    
    term1 = M / ((4 * np.pi * t_safe)**1.5 * np.sqrt(Dx * Dy * Dz))
    exp_x = -((x - x0 - u*t_safe)**2) / (4 * Dx * t_safe)
    exp_y = -((y - y0 - v*t_safe)**2) / (4 * Dy * t_safe)
    exp_z = -((z - z0 - w*t_safe)**2) / (4 * Dz * t_safe)
    
    C = term1 * np.exp(exp_x + exp_y + exp_z)

    # Physically, if t <= 0, concentration is 0. 
    C = np.where(t > 0, C, 0.0)
    
    return C


# --- 4. Main App Layout (Tabs) ---

tab1, tab2 = st.tabs(["📊 Simulation & Visualization", "📚 Theory & References"])

# ==========================================
# TAB 1: VISUALIZATION
# ==========================================
with tab1:
    st.subheader("Visualization Settings")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        plot_mode = st.selectbox(
            "Select Plot Configuration",
            [
                "1. Space-Time: Time (X) vs Length (Y) vs Depth (Z)",
                "2. Space-Time: Time (X) vs Length (Y) vs Breadth (Z)",
                "3. Space-Time: Time (X) vs Depth (Y) vs Breadth (Z)",
                "4. 3D Spatial Snapshot: Length (X) vs Breadth (Y) vs Depth (Z) at fixed Time"
            ]
        )
    with col2:
        # Resolution control
        res = st.select_slider("Resolution (Grid Density)", options=["Low", "Medium", "High"], value="Medium")
        if res == "Low": n_pts = 20
        elif res == "Medium": n_pts = 35
        else: n_pts = 50

    # --- Plot Generation Block ---

    if "4. 3D Spatial Snapshot" in plot_mode:
        # === Mode 4: Real 3D Space (X, Y, Z) at fixed T ===
        st.info("Visualizing the pollutant cloud in the river channel at a specific moment in time.")
        
        sim_time = st.slider("Time T [s]", 0.0, 3600.0, 60.0, step=10.0)
        
        # Create spatial grid
        x_vals = np.linspace(0, L, n_pts)
        y_vals = np.linspace(0, W, n_pts)
        z_vals = np.linspace(0, H, n_pts)
        X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
        
        with st.spinner('Calculating 3D Field...'):
            # Always calculate Instantaneous
            C = calculate_concentration_instantaneous(sim_time, X, Y, Z, M, u, v, w_vel, Dx, Dy, Dz, x0, y0, z0)

        # Visualization Setup
        c_max = np.max(C)
        
        fig = go.Figure(data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=C.flatten(),
            isomin=c_max * 0.05, 
            isomax=c_max,
            opacity=0.3, 
            surface_count=15, 
            colorscale='Jet',
            caps=dict(x_show=False, y_show=False, z_show=False)
        ))

        fig.update_layout(
            scene=dict(
                xaxis_title='Length (X) [m]',
                yaxis_title='Breadth (Y) [m]',
                zaxis_title='Depth (Z) [m]',
                aspectmode='data'
            ),
            title=f"Concentration Field at T = {sim_time}s (Instantaneous Pulse)",
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"**Max Concentration:** {c_max:.4e} kg/m³")

    else:
        # === Modes 1, 2, 3: Space-Time Plots (T + 2 Spatial Dims) ===
        st.info("Visualizing dispersion over Time (X-axis) and two spatial dimensions.")
        
        # 1. Setup Time Axis
        t_max = st.slider("Max Time for Simulation [s]", 60.0, 7200.0, 600.0)
        t_vals = np.linspace(1, t_max, n_pts) 
        
        # 2. Determine Axis mapping based on selection
        if "1. Space-Time" in plot_mode:
            # X: Time, Y: Length, Z: Depth (Breadth fixed)
            slice_loc = st.slider(f"Fixed Breadth (y) slice [m]", 0.0, W, y0)
            
            y_grid = slice_loc
            x_vals = np.linspace(0, L, n_pts)
            z_vals = np.linspace(0, H, n_pts)
            
            T_grid, X_grid, Z_grid = np.meshgrid(t_vals, x_vals, z_vals, indexing='ij')
            Y_grid = np.full_like(T_grid, y_grid)
            
            labels = {'x': 'Time [s]', 'y': 'River Length [m]', 'z': 'River Depth [m]'}
            plot_x, plot_y, plot_z = T_grid, X_grid, Z_grid

        elif "2. Space-Time" in plot_mode:
            # X: Time, Y: Length, Z: Breadth (Depth fixed)
            slice_loc = st.slider(f"Fixed Depth (z) slice [m]", 0.0, H, z0)
            
            z_grid = slice_loc
            x_vals = np.linspace(0, L, n_pts)
            y_vals = np.linspace(0, W, n_pts)
            
            T_grid, X_grid, Y_grid = np.meshgrid(t_vals, x_vals, y_vals, indexing='ij')
            Z_grid = np.full_like(T_grid, z_grid)
            
            labels = {'x': 'Time [s]', 'y': 'River Length [m]', 'z': 'River Breadth [m]'}
            plot_x, plot_y, plot_z = T_grid, X_grid, Y_grid

        elif "3. Space-Time" in plot_mode:
            # X: Time, Y: Depth, Z: Breadth (Length fixed)
            slice_loc = st.slider(f"Fixed Length (x) slice [m]", 0.0, L, L/2)
            
            x_grid = slice_loc
            z_vals = np.linspace(0, H, n_pts)
            y_vals = np.linspace(0, W, n_pts)
            
            T_grid, Z_grid, Y_grid = np.meshgrid(t_vals, z_vals, y_vals, indexing='ij')
            X_grid = np.full_like(T_grid, x_grid)
            
            labels = {'x': 'Time [s]', 'y': 'River Depth [m]', 'z': 'River Breadth [m]'}
            plot_x, plot_y, plot_z = T_grid, Z_grid, Y_grid

        # 3. Calculate Concentration
        with st.spinner('Calculating Space-Time Field...'):
            # Always calculate Instantaneous
            C = calculate_concentration_instantaneous(T_grid, X_grid, Y_grid, Z_grid, M, u, v, w_vel, Dx, Dy, Dz, x0, y0, z0)

        # 4. Plot
        c_max = np.max(C)
        
        st.markdown(f"**Visualizing:** {labels['x']} vs {labels['y']} vs {labels['z']}")
        
        fig = go.Figure(data=go.Volume(
            x=plot_x.flatten(),
            y=plot_y.flatten(),
            z=plot_z.flatten(),
            value=C.flatten(),
            isomin=c_max * 0.05,
            isomax=c_max,
            opacity=0.2,
            surface_count=20,
            colorscale='Turbo',
            caps=dict(x_show=False, y_show=False, z_show=False)
        ))

        fig.update_layout(
            scene=dict(
                xaxis_title=labels['x'],
                yaxis_title=labels['y'],
                zaxis_title=labels['z'],
            ),
            title="Space-Time Dispersion (Instantaneous Pulse)",
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"**Peak Concentration in this domain:** {c_max:.4e} kg/m³")


# ==========================================
# TAB 2: THEORY & REFERENCES
# ==========================================
with tab2:
    st.header("Theory & Background")
    
    st.markdown(r"""
    ### Instantaneous Point Source (Pulse)
    The concentration $C$ at position $(x,y,z)$ and time $t$ for a mass $M$ released at $(x_0, y_0, z_0)$ is given by the fundamental Gaussian Puff equation:
    
    $$
    C(x,y,z,t) = \frac{M}{(4\pi t)^{3/2} \sqrt{D_x D_y D_z}} \exp\left[ -\frac{(x-x_0-ut)^2}{4D_x t} -\frac{(y-y_0-vt)^2}{4D_y t} -\frac{(z-z_0-wt)^2}{4D_z t} \right]
    $$
    
    Where:
    * $M$: Total mass of pollutant released [kg]
    * $u, v, w$: Mean velocities in x, y, z directions [m/s]
    * $D_x, D_y, D_z$: Dispersion coefficients [m²/s]
    
    *Note: This simulation assumes an unbounded domain for simplicity. In a real river, reflection boundary conditions would be applied at the river bed and banks.*
    """)
    
    st.markdown("---")
    st.subheader("References & Credits")
    st.markdown("""
        ***Primary Source Reference:*** For further study and reference on Environmental Engineering and dispersion modeling, you may consult the work and class notes of:

        **Dr. Abhradeep Majumder, Ph.D.** Assistant Professor, Department of Civil Engineering, BITS Pilani-Hyderabad Campus  
        Academic Profiles: [Scopus](https://www.scopus.com/authid/detail.uri?authorId=57191504507), [ORCID](https://orcid.org/0000-0002-0186-6450), [Google Scholar](https://scholar.google.co.in/citations?user=mnJ5zdwAAAAJ&hl=en&oi=ao), [LinkedIn](https://linkedin.com/in/abhradeep-majumder-36503777/)

        ---
        ###### Application Development

        This interactive Pollutant Dispersion Model application was developed by **Subodh Purohit** ([LinkedIn](https://www.linkedin.com/in/subodhpurohit/)) as an educational tool.
        """, unsafe_allow_html=True)
