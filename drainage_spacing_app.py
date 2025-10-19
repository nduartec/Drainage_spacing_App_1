import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Drainage Spacing Hub", layout="wide")

st.title("Subsurface Drainage Spacing – Methods & Plots")
st.markdown("""
This app gathers classical formulas for estimating **drain spacing (S)** or analyzing
the **response of the water table** for subsurface drainage. Units are **SI** (m, d, m/day).
Use with engineering judgement. Field calibration and local guidelines prevail.
""")

def plot_S_vs_q(q_vals, S_vals, title, note=None):
    fig, ax = plt.subplots()
    ax.plot(q_vals, S_vals)
    ax.set_xlabel("Recharge rate q (m/day)")
    ax.set_ylabel("Spacing S (m)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    if note:
        ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=9, va="bottom")
    st.pyplot(fig)

def plot_time_series(t, y, ylabel, title):
    fig, ax = plt.subplots()
    ax.plot(t, y)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    st.pyplot(fig)

st.sidebar.header("Global plotting range")
qmin = st.sidebar.number_input("q min (m/day)", value=0.002, min_value=0.0, step=0.001, format="%.5f")
qmax = st.sidebar.number_input("q max (m/day)", value=0.02, min_value=0.0001, step=0.001, format="%.5f")
npts = st.sidebar.slider("Points", min_value=10, max_value=300, value=80)
q_range = np.linspace(qmin, qmax, npts)

tabs = st.tabs([
    "Dupuit–Forchheimer (steady)",
    "Hooghoudt (steady)",
    "Ernst (anisotropic steady)",
    "Glover–Dumm (transient)",
    "Hooghoudt (transient) – stub",
    "Boussinesq – stub",
    "Glover (steady) – stub",
    "Prevedello – stub",
    "Van Schilfgaarde – stub"
])

# Dupuit–Forchheimer (steady)
with tabs[0]:
    st.subheader("Dupuit–Forchheimer (steady)")
    col1, col2, col3 = st.columns(3)
    with col1:
        K = st.number_input("K (m/day)", value=0.8, min_value=0.0001, step=0.05)
    with col2:
        H = st.number_input("H (m) – water table height at midpoint above drain level", value=0.5, min_value=0.01, step=0.05)
    with col3:
        mode = st.selectbox("Solve for", ["Spacing S for q","Recharge q for S"])
    if mode == "Spacing S for q":
        S_vals = np.sqrt((4.0*K*H**2) / q_range)
        plot_S_vs_q(q_range, S_vals, "Dupuit–Forchheimer: S vs q",
                    note="q = (4 K H²)/S² ; planar flow, no entrance/radial losses.")
    else:
        S = st.number_input("Given spacing S (m)", value=40.0, min_value=1.0, step=1.0)
        q = (4.0*K*H**2) / (S**2)
        st.metric("Recharge q (m/day)", f"{q:.5f}")
        st.info("Formula: q = (4 K H²) / S²")

# Hooghoudt (steady)
with tabs[1]:
    st.subheader("Hooghoudt – Steady (permanent flow)")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        K = st.number_input("K (m/day)", value=0.8, min_value=0.0001, step=0.05, key="hhK")
    with col2:
        H = st.number_input("H (m)", value=0.5, min_value=0.01, step=0.05, key="hhH")
    with col3:
        de = st.number_input("Equivalent depth de (m)", value=0.7, min_value=0.001, step=0.05)
    with col4:
        mode = st.selectbox("Solve for", ["Spacing S for q","Recharge q for S"], key="hhmode")
    if mode == "Spacing S for q":
        S_vals = np.sqrt((4.0*K*H*(2.0*de + H)) / q_range)
        plot_S_vs_q(q_range, S_vals, "Hooghoudt steady: S vs q",
                    note="q = [4 K H (2 de + H)] / S² ; de from Hooghoudt tables.")
    else:
        S = st.number_input("Given spacing S (m)", value=40.0, min_value=1.0, step=1.0, key="hhS")
        q = (4.0*K*H*(2.0*de + H)) / (S**2)
        st.metric("Recharge q (m/day)", f"{q:.5f}")
        st.caption("Includes radial flow via equivalent depth de.")

# Ernst (anisotropic steady)
with tabs[2]:
    st.subheader("Ernst – Anisotropic Steady Flow")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        Kh = st.number_input("Kh (m/day)", value=0.8, min_value=0.0001, step=0.05)
    with col2:
        Kv = st.number_input("Kv (m/day)", value=0.2, min_value=0.00001, step=0.01)
    with col3:
        H = st.number_input("H (m)", value=0.5, min_value=0.01, step=0.05, key="ernH")
    with col4:
        de = st.number_input("Equivalent depth de (m)", value=0.7, min_value=0.001, step=0.05, key="ernde")
    mode = st.selectbox("Solve for", ["Spacing S for q","Recharge q for S"], key="ernmode")
    if mode == "Spacing S for q":
        S_vals = np.sqrt((4.0*Kh*H**2 + 8.0*Kv*de*H) / q_range)
        plot_S_vs_q(q_range, S_vals, "Ernst (anisotropic): S vs q",
                    note="q = [4 Kh H² + 8 Kv de H] / S²")
    else:
        S = st.number_input("Given spacing S (m)", value=40.0, min_value=1.0, step=1.0, key="ernS")
        q = (4.0*Kh*H**2 + 8.0*Kv*de*H) / (S**2)
        st.metric("Recharge q (m/day)", f"{q:.5f}")
        st.caption("Ernst combines planar (Kh) and vertical/radial (Kv, de) terms.")

# Glover–Dumm (transient midpoint)
with tabs[3]:
    st.subheader("Glover–Dumm – Transient (midpoint drawdown)")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        K = st.number_input("K (m/day)", value=0.8, min_value=0.0001, step=0.05, key="gdK")
    with col2:
        Sy = st.number_input("Specific yield Sy (–)", value=0.12, min_value=0.001, max_value=0.6, step=0.01)
    with col3:
        H0 = st.number_input("Initial H0 at midpoint (m)", value=0.6, min_value=0.01, step=0.05)
    with col4:
        Hf = st.number_input("Target Hf at midpoint (m)", value=0.3, min_value=0.001, step=0.05)
    t_days = st.number_input("Drainage time available t (days)", value=3.0, min_value=0.1, step=0.5)
    if Hf >= H0:
        st.error("Hf must be less than H0 to have drawdown.")
    else:
        S = np.pi * np.sqrt((K * t_days) / (Sy * np.log(H0 / Hf)))
        st.metric("Required spacing S (m)", f"{S:.2f}")
        t = np.linspace(0, t_days, 200)
        Ht = H0 * np.exp(-(np.pi**2 * K * t) / (Sy * S**2))
        plot_time_series(t, Ht, "H(t) at midpoint (m)",
                         "Glover–Dumm midpoint response with computed S")

def stub_panel(label, refs):
    st.info(f"{label} – Placeholder. Add your preferred formulation and calibration. References: {refs}")

with tabs[4]:
    stub_panel("Hooghoudt (non-steady / transient)",
               "van Beers tables; Ritzema (Drainage Principles and Applications).")
with tabs[5]:
    stub_panel("Boussinesq (general)",
               "Boussinesq equation for unconfined aquifer; see Bear (Hydraulics of Groundwater).")
with tabs[6]:
    stub_panel("Glover (steady)",
               "Glover & Balmer steady approximations.")
with tabs[7]:
    stub_panel("Prevedello",
               "Prevedello analytical developments for specific boundary conditions.")
with tabs[8]:
    stub_panel("Van Schilfgaarde",
               "USDA/ARS transient drainage formulations.")
