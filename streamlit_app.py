import streamlit as st

import pandas as pd

import numpy as np

import io

import plotly.graph_objects as go

 

# IEC 60891 correction constants (example values, adjust as needed)

alpha_Isc = 0.0005  # A/°C

beta_Voc = -0.0023  # V/°C

gamma_Pmax = -0.0045  # 1/°C

 

st.title("IEC 60891 IV Curve Correction for PV Modules")

 

# Upload CSV file

uploaded_file = st.file_uploader("Upload IV Data CSV", type=["csv"])

 

if uploaded_file:

    # Read CSV

    df = pd.read_csv(uploaded_file)

    st.write("Uploaded Data Preview:", df.head())

 

    if "Voltage" in df.columns and "Current" in df.columns:

        # Sidebar inputs with sliders

        st.sidebar.header("Correction Parameters")

        T_meas = st.sidebar.slider("Measured Temperature (°C)", min_value=0.0, max_value=80.0, value=45.0)

        T_ref = st.sidebar.slider("Reference Temperature (°C)", min_value=0.0, max_value=80.0, value=25.0)

        G_meas = st.sidebar.slider("Measured Irradiance (W/m²)", min_value=200.0, max_value=1200.0, value=800.0)

        G_ref = st.sidebar.slider("Reference Irradiance (W/m²)", min_value=200.0, max_value=1200.0, value=1000.0)

 

        # Method selection

        method = st.sidebar.selectbox("Select IEC 60891 Method", ["Method 1", "Method 2", "Method 3"])

 

        V = df["Voltage"].values

        I = df["Current"].values

 

        delta_T = T_ref - T_meas

        delta_G = G_ref / G_meas

 

        # Apply correction based on selected method

        if method == "Method 1":

            I_corr = I * delta_G + alpha_Isc * delta_T

            V_corr = V + beta_Voc * delta_T

 

        elif method == "Method 2":

            I_corr = I * (G_ref / G_meas) * (1 + alpha_Isc * delta_T)

            V_corr = V * (1 + beta_Voc * delta_T)

 

        elif method == "Method 3":

            coeffs = np.polyfit(V, I, 3)  # cubic fit

            poly = np.poly1d(coeffs)

            V_corr = V * (1 + beta_Voc * delta_T)

            I_corr = poly(V_corr) * (G_ref / G_meas) * (1 + alpha_Isc * delta_T)

 

        # Create corrected DataFrame

        corrected_df = pd.DataFrame({"Voltage": V_corr, "Current": I_corr})

 

        # Calculate metrics for original curve

        Isc_orig = I[np.argmin(np.abs(V))]  # closest to V=0

        Voc_orig = V[np.argmin(np.abs(I))]  # closest to I=0

        Pmax_orig = np.max(V * I)

 

        # Calculate metrics for corrected curve

        Isc_corr = I_corr[np.argmin(np.abs(V_corr))]

        Voc_corr = V_corr[np.argmin(np.abs(I_corr))]

        Pmax_corr = np.max(V_corr * I_corr)

 

        # Plot using Plotly

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=V, y=I, mode='lines', name='Original IV', line=dict(color='blue')))

        fig.add_trace(go.Scatter(x=V_corr, y=I_corr, mode='lines', name=f'Corrected IV ({method})', line=dict(color='red', dash='dash')))

 

        fig.update_layout(

            title=f"IV Curve Correction - {method}",

            xaxis_title="Voltage (V)",

            yaxis_title="Current (A)",

            hovermode="x unified"

        )

 

        st.plotly_chart(fig, use_container_width=True)

 

        # Display metrics

        st.subheader("Performance Metrics")

        col1, col2 = st.columns(2)

        with col1:

            st.write("**Original Curve**")

            st.metric("Isc (A)", f"{Isc_orig:.3f}")

            st.metric("Voc (V)", f"{Voc_orig:.3f}")

            st.metric("Pmax (W)", f"{Pmax_orig:.3f}")

        with col2:

            st.write("**Corrected Curve**")

            st.metric("Isc (A)", f"{Isc_corr:.3f}")

            st.metric("Voc (V)", f"{Voc_corr:.3f}")

            st.metric("Pmax (W)", f"{Pmax_corr:.3f}")

 

        # Download corrected data

        csv_buffer = io.StringIO()

        corrected_df.to_csv(csv_buffer, index=False)

        st.download_button("Download Corrected Data", csv_buffer.getvalue(), "corrected_IV.csv", "text/csv")

    else:

        st.error("CSV must contain 'Voltage' and 'Current' columns.")
