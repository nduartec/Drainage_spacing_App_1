import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import pandas as pd

# ---------- Config ----------
st.set_page_config(page_title="Drainage Spacing Hub", layout="wide")

# ---------- Utilidades ----------
def df_to_csv_bytes(df: pd.DataFrame, fname: str = "data.csv"):
    bio = io.BytesIO()
    df.to_csv(bio, index=False)
    return bio.getvalue()

def plot_xy(x, y, xlabel, ylabel, title, note=None):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
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

# ---------- Ayuda global ----------
with st.sidebar.expander("ℹ️ Ayuda / Definiciones", expanded=False):
    st.markdown("""
**Símbolos (SI):**  
`K, Kh, Kv` (m/día), `H` (m), `dₑ` (m), `Sy` (–), `q` (m/día), `S` (m), `t` (días).

**Atajos:**  
- Dupuit–Forchheimer: flujo planar (sin término radial).  
- Hooghoudt (perm.): planar + radial (vía `dₑ`).  
- Ernst: anisotrópico (usa `Kh` y `Kv`).  
- Glover–Dumm: transitorio (decaimiento exponencial en el punto medio).
- Van Schilfgaarde / Prevedello: transitorios parametrizados.
""")

with st.sidebar.expander("📚 Bibliografía (sugerida)", expanded=False):
    st.markdown("""
- Ritzema (ed.). *Drainage Principles and Applications* (ILRI).  
- van Beers. *Calculation of drain spacings*.  
- USDA NRCS, *Subsurface Drainage Design*.  
- Bear, *Hydraulics of Groundwater* (Boussinesq, acuífero libre).  
- FAO Irrigation & Drainage Papers (Hooghoudt/Van Schilfgaarde).
""")

st.title("Subsurface Drainage Spacing – Methods & Plots")

# Rango global para gráficas S–q
st.sidebar.header("Rango global para S–q")
qmin = st.sidebar.number_input("q min (m/day)", value=0.002, min_value=0.0, step=0.001, format="%.5f")
qmax = st.sidebar.number_input("q max (m/day)", value=0.02, min_value=0.0001, step=0.001, format="%.5f")
npts = st.sidebar.slider("Puntos", min_value=10, max_value=300, value=80)
q_range = np.linspace(qmin, qmax, npts)

tabs = st.tabs([
    "Dupuit–Forchheimer (perm.)",
    "Hooghoudt (perm.)",
    "Ernst (anisotrópico, perm.)",
    "Glover–Dumm (no perm.)",
    "Hooghoudt (no perm.)",
    "Boussinesq (perm.)",
    "Glover (perm.)",
    "Prevedello (no perm.)",
    "Van Schilfgaarde (no perm.)"
])

# -------------------------------------------------
# Dupuit–Forchheimer (permanente)
# q = (4 K H^2) / S^2  ->  S = sqrt( (4 K H^2) / q )
# -------------------------------------------------
with tabs[0]:
    st.subheader("Dupuit–Forchheimer (flujo permanente)")
    st.caption(r"$q=\dfrac{4\,K\,H^2}{S^2}\;\Rightarrow\;S=\sqrt{\dfrac{4\,K\,H^2}{q}}$")
    c1, c2, c3 = st.columns(3)
    with c1: K = st.number_input("K (m/día)", value=0.8, min_value=0.0001, step=0.05)
    with c2: H = st.number_input("H (m)", value=0.5, min_value=0.01, step=0.01)
    with c3: modo = st.selectbox("Resolver", ["S vs q", "q dado S"])
    if modo == "S vs q":
        S_vals = np.sqrt((4.0*K*H**2)/q_range)
        plot_xy(q_range, S_vals, "q (m/día)", "S (m)",
                "Dupuit–Forchheimer: S vs q", "Flujo planar (sin término radial).")
        df = pd.DataFrame({"q_m_per_day": q_range, "S_m": S_vals})
        st.download_button("⬇️ Descargar CSV S–q (Dupuit–Forchheimer)", df_to_csv_bytes(df), file_name="dupuit_S_vs_q.csv", mime="text/csv")
    else:
        S = st.number_input("S (m)", value=40.0, min_value=1.0, step=1.0)
        q = (4.0*K*H**2)/(S**2)
        st.metric("q (m/día)", f"{q:.5f}")

# -------------------------------------------------
# Hooghoudt (permanente)
# q = (4 K H (2 de + H)) / S^2
# -------------------------------------------------
with tabs[1]:
    st.subheader("Hooghoudt – permanente")
    st.caption(r"$q=\dfrac{4\,K\,H\,(2\,d_e+H)}{S^2}\;\Rightarrow\;S=\sqrt{\dfrac{4\,K\,H\,(2\,d_e+H)}{q}}$")
    c1, c2, c3, c4 = st.columns(4)
    with c1: K = st.number_input("K (m/día)", value=0.8, min_value=0.0001, step=0.05, key="hhK")
    with c2: H = st.number_input("H (m)", value=0.5, min_value=0.01, step=0.01, key="hhH")
    with c3: de = st.number_input("dₑ (m)", value=0.7, min_value=0.001, step=0.01)
    with c4: modo = st.selectbox("Resolver", ["S vs q", "q dado S"], key="hhmode")
    if modo == "S vs q":
        S_vals = np.sqrt((4.0*K*H*(2.0*de + H))/q_range)
        plot_xy(q_range, S_vals, "q (m/día)", "S (m)", "Hooghoudt (perm.): S vs q", "Incluye flujo radial vía dₑ.")
        df = pd.DataFrame({"q_m_per_day": q_range, "S_m": S_vals})
        st.download_button("⬇️ Descargar CSV S–q (Hooghoudt perm.)", df_to_csv_bytes(df), file_name="hooghoudt_perm_S_vs_q.csv", mime="text/csv")
    else:
        S = st.number_input("S (m)", value=40.0, min_value=1.0, step=1.0, key="hhS")
        q = (4.0*K*H*(2.0*de + H))/ (S**2)
        st.metric("q (m/día)", f"{q:.5f}")

# -------------------------------------------------
# Ernst (anisotrópico permanente)
# q = [4 Kh H^2 + 8 Kv de H] / S^2
# -------------------------------------------------
with tabs[2]:
    st.subheader("Ernst – anisotrópico (permanente)")
    st.caption(r"$q=\dfrac{4\,K_h\,H^2+8\,K_v\,d_e\,H}{S^2}$")
    c1, c2, c3, c4 = st.columns(4)
    with c1: Kh = st.number_input("Kh (m/día)", value=0.8, min_value=0.0001, step=0.05)
    with c2: Kv = st.number_input("Kv (m/día)", value=0.2, min_value=0.00001, step=0.01)
    with c3: H  = st.number_input("H (m)", value=0.5, min_value=0.01, step=0.01, key="ernH")
    with c4: de = st.number_input("dₑ (m)", value=0.7, min_value=0.001, step=0.01, key="ernde")
    modo = st.selectbox("Resolver", ["S vs q", "q dado S"], key="ernmode")
    if modo == "S vs q":
        S_vals = np.sqrt((4.0*Kh*H**2 + 8.0*Kv*de*H)/q_range)
        plot_xy(q_range, S_vals, "q (m/día)", "S (m)", "Ernst (anisotrópico): S vs q",
                "Combina flujo planar (Kh) y radial (Kv, dₑ).")
        df = pd.DataFrame({"q_m_per_day": q_range, "S_m": S_vals})
        st.download_button("⬇️ Descargar CSV S–q (Ernst)", df_to_csv_bytes(df), file_name="ernst_S_vs_q.csv", mime="text/csv")
    else:
        S = st.number_input("S (m)", value=40.0, min_value=1.0, step=1.0, key="ernS")
        q = (4.0*Kh*H**2 + 8.0*Kv*de*H)/(S**2)
        st.metric("q (m/día)", f"{q:.5f}")

# -------------------------------------------------
# Glover–Dumm (no permanente, punto medio)
# H(t) = H0 * exp( - (pi^2 * K * t) / (Sy * S^2) )
# S = pi * sqrt( K t / (Sy * ln(H0/Hf)) )
# -------------------------------------------------
with tabs[3]:
    st.subheader("Glover–Dumm – no permanente (punto medio)")
    st.caption(r"$H(t)=H_0\,\exp\!\left(-\dfrac{\pi^2 K t}{S_y S^2}\right)$  ;  "
               r"$S=\pi\,\sqrt{\dfrac{K\,t}{S_y\,\ln(H_0/H_f)}}$")
    c1, c2, c3, c4 = st.columns(4)
    with c1: K  = st.number_input("K (m/día)", value=0.8, min_value=0.0001, step=0.05, key="gdK")
    with c2: Sy = st.number_input("Sy (–)", value=0.12, min_value=0.001, max_value=0.6, step=0.01)
    with c3: H0 = st.number_input("H₀ (m)", value=0.6, min_value=0.01, step=0.01)
    with c4: Hf = st.number_input("Hf (m)", value=0.3, min_value=0.001, step=0.01)
    t_days = st.number_input("t (días)", value=3.0, min_value=0.1, step=0.1)
    if Hf >= H0:
        st.error("Hf debe ser menor que H0.")
    else:
        S = np.pi*np.sqrt((K*t_days)/(Sy*np.log(H0/Hf)))
        st.metric("S requerido (m)", f"{S:.2f}")
        t = np.linspace(0, t_days, 200)
        Ht = H0*np.exp(-(np.pi**2*K*t)/(Sy*S**2))
        plot_time_series(t, Ht, "H(t) en punto medio (m)", "Glover–Dumm: evolución H(t)")
        df = pd.DataFrame({"t_days": t, "H_midpoint_m": Ht})
        st.download_button("⬇️ Descargar CSV H(t) (Glover–Dumm)", df_to_csv_bytes(df), file_name="glover_dumm_Ht.csv", mime="text/csv")

# -------------------------------------------------
# Hooghoudt (no permanente) – Aproximación docente
# -------------------------------------------------
with tabs[4]:
    st.subheader("Hooghoudt – no permanente (aprox. docente)")
    st.caption("Se aproxima H(t) con Glover–Dumm y se usa en Hooghoudt-permanente.")
    c1, c2, c3, c4 = st.columns(4)
    with c1: K  = st.number_input("K (m/día)", value=0.8, min_value=0.0001, step=0.05, key="hhtK")
    with c2: Sy = st.number_input("Sy (–)", value=0.12, min_value=0.001, max_value=0.6, step=0.01, key="hhtSy")
    with c3: de = st.number_input("dₑ (m)", value=0.7, min_value=0.001, step=0.01, key="hhtde")
    with c4: q  = st.number_input("q (m/día)", value=0.01, min_value=0.00001, step=0.001, format="%.5f")
    c5, c6, c7 = st.columns(3)
    with c5: H0 = st.number_input("H₀ (m)", value=0.6, min_value=0.01, step=0.01, key="hhtH0")
    with c6: Hf = st.number_input("H objetivo Hf (m)", value=0.3, min_value=0.001, step=0.01, key="hhtHf")
    with c7: t_days = st.number_input("t (días)", value=3.0, min_value=0.1, step=0.1, key="hhtt")
    if Hf >= H0:
        st.error("Hf debe ser menor que H0.")
    else:
        Sgd = np.pi*np.sqrt((K*t_days)/(Sy*np.log(H0/Hf)))
        Heff = H0*np.exp(-(np.pi**2*K*t_days)/(Sy*Sgd**2))
        S_hh = np.sqrt((4.0*K*Heff*(2.0*de + Heff))/q)
        st.metric("S (m) Hooghoudt-no perm. (aprox.)", f"{S_hh:.2f}")
        df = pd.DataFrame({
            "K_m_per_day":[K], "Sy":[Sy], "de_m":[de], "q_m_per_day":[q],
            "H0_m":[H0], "Hf_m":[Hf], "t_days":[t_days],
            "S_glover_dumm_m":[Sgd], "H_eff_m":[Heff], "S_hooghoudt_np_m":[S_hh]
        })
        st.download_button("⬇️ Descargar CSV (Hooghoudt no perm. aprox.)", df_to_csv_bytes(df),
                           file_name="hooghoudt_transient_approx.csv", mime="text/csv")

# -------------------------------------------------
# Boussinesq (perm.) – forma operativa (≈ Dupuit)
# -------------------------------------------------
with tabs[5]:
    st.subheader("Boussinesq – permanente (forma operativa)")
    st.caption("Bajo hipótesis de Dupuit, coincide con Dupuit–Forchheimer.")
    c1, c2, c3 = st.columns(3)
    with c1: K = st.number_input("K (m/día)", value=0.8, min_value=0.0001, step=0.05, key="bK")
    with c2: H = st.number_input("H (m)", value=0.5, min_value=0.01, step=0.01, key="bH")
    with c3: modo = st.selectbox("Resolver", ["S vs q", "q dado S"], key="bmodo")
    if modo == "S vs q":
        S_vals = np.sqrt((4.0*K*H**2)/q_range)
        plot_xy(q_range, S_vals, "q (m/día)", "S (m)", "Boussinesq (perm.): S vs q",
                "Forma operativa (equivalente a Dupuit en régimen perm.).")
        df = pd.DataFrame({"q_m_per_day": q_range, "S_m": S_vals})
        st.download_button("⬇️ Descargar CSV S–q (Boussinesq perm.)", df_to_csv_bytes(df), file_name="boussinesq_perm_S_vs_q.csv", mime="text/csv")
    else:
        S = st.number_input("S (m)", value=40.0, min_value=1.0, step=1.0, key="bS")
        q = (4.0*K*H**2)/(S**2)
        st.metric("q (m/día)", f"{q:.5f}")

# -------------------------------------------------
# Glover (perm.) – forma operativa
# -------------------------------------------------
with tabs[6]:
    st.subheader("Glover – permanente (forma operativa)")
    st.caption("Bajo supuestos comunes, la forma permanente converge a Dupuit/Hooghoudt.")
    c1, c2, c3 = st.columns(3)
    with c1: K = st.number_input("K (m/día)", value=0.8, min_value=0.0001, step=0.05, key="gK")
    with c2: H = st.number_input("H (m)", value=0.5, min_value=0.01, step=0.01, key="gH")
    with c3: modo = st.selectbox("Resolver", ["S vs q", "q dado S"], key="gmodo")
    if modo == "S vs q":
        S_vals = np.sqrt((4.0*K*H**2)/q_range)
        plot_xy(q_range, S_vals, "q (m/día)", "S (m)", "Glover (perm.): S vs q",
                "Forma operativa para comparación con Dupuit/Hooghoudt.")
        df = pd.DataFrame({"q_m_per_day": q_range, "S_m": S_vals})
        st.download_button("⬇️ Descargar CSV S–q (Glover perm.)", df_to_csv_bytes(df), file_name="glover_perm_S_vs_q.csv", mime="text/csv")
    else:
        S = st.number_input("S (m)", value=40.0, min_value=1.0, step=1.0, key="gS")
        q = (4.0*K*H**2)/(S**2)
        st.metric("q (m/día)", f"{q:.5f}")

# -------------------------------------------------
# Prevedello (no perm.) – parametrización operativa
# S = pi * sqrt( K t / (Sy * Phi) )
# -------------------------------------------------
with tabs[7]:
    st.subheader("Prevedello – no permanente (parametrización operativa)")
    st.caption(r"$S=\pi\sqrt{\dfrac{K\,t}{S_y\,\Phi}}$  con  $\Phi$ adimensional (usuario).")
    c1, c2, c3, c4 = st.columns(4)
    with c1: K  = st.number_input("K (m/día)", value=0.8, min_value=0.0001, step=0.05, key="pK")
    with c2: Sy = st.number_input("Sy (–)", value=0.12, min_value=0.001, max_value=0.6, step=0.01, key="pSy")
    with c3: t_days = st.number_input("t (días)", value=3.0, min_value=0.1, step=0.1, key="pt")
    with c4: Phi = st.number_input("Φ (adimensional)", value=1.0, min_value=0.0001, step=0.05,
                                   help="Ej.: Φ=ln(H0/Hf) o Φ ajustada según tu referencia.")
    S = np.pi*np.sqrt((K*t_days)/(Sy*Phi))
    st.metric("S (m)", f"{S:.2f}")
    df = pd.DataFrame({"K_m_per_day":[K], "Sy":[Sy], "t_days":[t_days], "Phi":[Phi], "S_m":[S]})
    st.download_button("⬇️ Descargar CSV (Prevedello no perm.)", df_to_csv_bytes(df), file_name="prevedello_transient.csv", mime="text/csv")

# -------------------------------------------------
# Van Schilfgaarde (no perm.) – parametrización operativa
# S = pi * sqrt( K t / (Sy * Psi) )
# -------------------------------------------------
with tabs[8]:
    st.subheader("Van Schilfgaarde – no permanente (parametrización operativa)")
    st.caption(r"$S=\pi\sqrt{\dfrac{K\,t}{S_y\,\Psi}}$  con  $\Psi$ adimensional (usuario).")
    c1, c2, c3, c4 = st.columns(4)
    with c1: K  = st.number_input("K (m/día)", value=0.8, min_value=0.0001, step=0.05, key="vK")
    with c2: Sy = st.number_input("Sy (–)", value=0.12, min_value=0.001, max_value=0.6, step=0.01, key="vSy")
    with c3: t_days = st.number_input("t (días)", value=3.0, min_value=0.1, step=0.1, key="vt")
    with c4: Psi = st.number_input("Ψ (adimensional)", value=1.0, min_value=0.0001, step=0.05,
                                   help="Ajusta Ψ según la variante de Van Schilfgaarde que utilices.")
    S = np.pi*np.sqrt((K*t_days)/(Sy*Psi))
    st.metric("S (m)", f"{S:.2f}")
    df = pd.DataFrame({"K_m_per_day":[K], "Sy":[Sy], "t_days":[t_days], "Psi":[Psi], "S_m":[S]})
    st.download_button("⬇️ Descargar CSV (Van Schilfgaarde no perm.)", df_to_csv_bytes(df),
                       file_name="van_schilfgaarde_transient.csv", mime="text/csv")
