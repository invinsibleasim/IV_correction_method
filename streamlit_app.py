import streamlit as st

import pandas as pd

import numpy as np

import io

import plotly.graph_objects as go

from scipy.interpolate import CubicSpline

from sklearn.metrics import mean_squared_error, mean_absolute_error

 

st.title("IEC 60891 IV Curve Correction - Procedures 1, 2, and 3")

 

# Sidebar: Select correction procedure

procedure = st.sidebar.selectbox("Select Correction Procedure", ["Procedure 1", "Procedure 2", "Procedure 3"])

 

# Common sidebar inputs

G_target = st.sidebar.slider("Target Irradiance (W/m²)", min_value=200.0, max_value=1200.0, value=1000.0)

T_target = st.sidebar.slider("Target Temperature (°C)", min_value=0.0, max_value=80.0, value=25.0)

num_points = st.sidebar.slider("Number of points in corrected CSV", min_value=10, max_value=1000, value=500)

interp_type = st.sidebar.selectbox("Interpolation Type", ["Linear", "Spline", "Polynomial"])

 

if procedure in ["Procedure 1", "Procedure 2"]:

    uploaded_file = st.file_uploader("Upload IV Data CSV", type=["csv"])

    if uploaded_file:

        df = pd.read_csv(uploaded_file)

        st.write("Uploaded Data Preview:", df.head())

 

        if "Voltage" in df.columns and "Current" in df.columns:

            V1 = df["Voltage"].values

            I1 = df["Current"].values

 

            # Sidebar inputs for measured conditions

            G_meas = st.sidebar.slider("Measured Irradiance G1 (W/m²)", min_value=200.0, max_value=1200.0, value=800.0)

            T_meas = st.sidebar.slider("Measured Temperature T1 (°C)", min_value=0.0, max_value=80.0, value=45.0)

 

            if procedure == "Procedure 1":

                # Additional inputs for Procedure 1

                st.sidebar.subheader("Correction Coefficients")

                alpha_Isc = st.sidebar.number_input("Alpha (Isc Temp Coefficient) [A/°C]", value=0.0005, step=0.0001, format="%.6f")

                beta_Voc = st.sidebar.number_input("Beta (Voc Temp Coefficient) [V/°C]", value=-0.0023, step=0.0001, format="%.6f")

                RS = st.sidebar.number_input("Series Resistance RS (Ω)", value=0.2, step=0.01, format="%.3f")

                kappa = st.sidebar.number_input("Curve Correction Factor κ", value=0.0001, step=0.0001, format="%.6f")

                ISC1 = st.sidebar.number_input("Measured Short-Circuit Current ISC1 (A)", value=8.0, step=0.1, format="%.2f")

 

                # Apply Procedure 1 formula

                I2 = I1 + ISC1 * ((G_target / G_meas) - 1) + alpha_Isc * (T_target - T_meas)

                V2 = V1 - RS * (I2 - I1) - kappa * I2 * (T_target - T_meas) + beta_Voc * (T_target - T_meas)

 

            elif procedure == "Procedure 2":

                # Simplified linear correction for Procedure 2

                I2 = I1 * (G_target / G_meas) * (1 + 0.0005 * (T_target - T_meas))

                V2 = V1 * (1 + (-0.0023) * (T_target - T_meas))

 

            # Interpolation for corrected curve

            V_new = np.linspace(V2.min(), V2.max(), num_points)

            if interp_type == "Linear":

                I_new = np.interp(V_new, V2, I2)

            elif interp_type == "Spline":

                spline = CubicSpline(V2, I2)

                I_new = spline(V_new)

            else:

                poly_fit = np.polyfit(V2, I2, 3)

                I_new = np.poly1d(poly_fit)(V_new)

 

            corrected_df = pd.DataFrame({"Voltage": V_new, "Current": I_new})

 

            # Metrics

            Isc_orig = I1[np.argmin(np.abs(V1))]

            Voc_orig = V1[np.argmin(np.abs(I1))]

            Pmax_orig = np.max(V1 * I1)

            Isc_corr = I_new[np.argmin(np.abs(V_new))]

            Voc_corr = V_new[np.argmin(np.abs(I_new))]

            Pmax_corr = np.max(V_new * I_new)

 

            # Error metrics

            I2_interp = np.interp(V_new, V2, I2)

            rmse = np.sqrt(mean_squared_error(I2_interp, I_new))

            mae = mean_absolute_error(I2_interp, I_new)

 

            # Plot

            fig = go.Figure()

            fig.add_trace(go.Scatter(x=V1, y=I1, mode='lines', name='Original IV', line=dict(color='blue')))

            fig.add_trace(go.Scatter(x=V_new, y=I_new, mode='lines', name=f'Corrected IV ({procedure})', line=dict(color='red', dash='dash')))

            fig.update_layout(title=f"IV Curve Correction - {procedure}", xaxis_title="Voltage (V)", yaxis_title="Current (A)")

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

 

            st.subheader("Interpolation Error Metrics")

            st.write(f"RMSE: {rmse:.6f}")

            st.write(f"MAE: {mae:.6f}")

 

            # Download corrected data

            csv_buffer = io.StringIO()

            corrected_df.to_csv(csv_buffer, index=False)

            st.download_button("Download Corrected Data", csv_buffer.getvalue(), "corrected_IV.csv", "text/csv")

 

elif procedure == "Procedure 3":

    st.write("Upload two IV curves for interpolation/extrapolation")

    file1 = st.file_uploader("Upload IV Data CSV for Condition 1", type=["csv"], key="file1")

    file2 = st.file_uploader("Upload IV Data CSV for Condition 2", type=["csv"], key="file2")

 

    if file1 and file2:

        df1 = pd.read_csv(file1)

        df2 = pd.read_csv(file2)

        st.write("Curve 1 Preview:", df1.head())

        st.write("Curve 2 Preview:", df2.head())

 

        if "Voltage" in df1.columns and "Current" in df1.columns and "Voltage" in df2.columns and "Current" in df2.columns:

            V1 = df1["Voltage"].values

            I1 = df1["Current"].values

            V2 = df2["Voltage"].values

            I2 = df2["Current"].values

 

            # Sidebar inputs for measured conditions

            G1 = st.sidebar.slider("Measured Irradiance G1 (W/m²)", min_value=200.0, max_value=1200.0, value=800.0)

            G2 = st.sidebar.slider("Measured Irradiance G2 (W/m²)", min_value=200.0, max_value=1200.0, value=1000.0)

            T1 = st.sidebar.slider("Measured Temperature T1 (°C)", min_value=0.0, max_value=80.0, value=45.0)

            T2 = st.sidebar.slider("Measured Temperature T2 (°C)", min_value=0.0, max_value=80.0, value=25.0)

 

            # Compute interpolation factor a

            a = (G_target - G1) / (G2 - G1)

 

            # Align lengths by interpolation

            V_common = np.linspace(max(V1.min(), V2.min()), min(V1.max(), V2.max()), num_points)

            I1_interp = np.interp(V_common, V1, I1)

            I2_interp = np.interp(V_common, V2, I2)

 

            # Apply Procedure 3 formula

            V3 = V_common + a * (V_common - V_common)  # Voltage difference is zero since aligned

            I3 = I1_interp + a * (I2_interp - I1_interp)

 

            corrected_df = pd.DataFrame({"Voltage": V_common, "Current": I3})

 

            # Metrics

            Isc_orig = I1[np.argmin(np.abs(V1))]

            Voc_orig = V1[np.argmin(np.abs(I1))]

            Pmax_orig = np.max(V1 * I1)

            Isc_corr = I3[np.argmin(np.abs(V_common))]

            Voc_corr = V_common[np.argmin(np.abs(I3))]

            Pmax_corr = np.max(V_common * I3)

 

            # Plot

            fig = go.Figure()

            fig.add_trace(go.Scatter(x=V1, y=I1, mode='lines', name='Curve 1', line=dict(color='blue')))

            fig.add_trace(go.Scatter(x=V2, y=I2, mode='lines', name='Curve 2', line=dict(color='green')))

            fig.add_trace(go.Scatter(x=V_common, y=I3, mode='lines', name='Corrected IV (Procedure 3)', line=dict(color='red', dash='dash')))

            fig.update_layout(title="IV Curve Correction - Procedure 3", xaxis_title="Voltage (V)", yaxis_title="Current (A)")

            st.plotly_chart(fig, use_container_width=True)

 

            # Display metrics

            st.subheader("Performance Metrics")

            col1, col2 = st.columns(2)

            with col1:

                st.write("**Curve 1**")
