import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pygad
import time

st.set_page_config(page_title="CL Slag Cu Counterfactuals", layout="wide")

# -------------------------
# Load artifacts
# -------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("cu_rf_model.pkl")      # trained RF (unscaled training)
    scaler = joblib.load("cu_scaler.pkl")      # scaler (fit on training X_train)
    X_train = np.load("X_train.npy")           # unscaled X_train (shape n_samples, n_features)
    y_train = np.load("y_train.npy")           # training labels
    FEATURE_ORDER = joblib.load("feature_order.pkl")
    return model, scaler, X_train, y_train, FEATURE_ORDER

try:
    model, scaler, X_train, y_train, FEATURE_ORDER = load_artifacts()
except Exception as e:
    st.error("Error loading artifacts. Ensure cu_rf_model.pkl, cu_scaler.pkl, X_train.npy, y_train.npy, feature_order.pkl exist.")
    st.stop()

n_features = len(FEATURE_ORDER)

# -------------------------
# UI: Inputs grouped (same as earlier)
# -------------------------
st.title("CL Slag Cu Counterfactuals")

st.markdown("Enter process inputs:")

# handy to fetch training means for defaults
train_means = X_train.mean(axis=0)
train_mins = X_train.min(axis=0)
train_maxs = X_train.max(axis=0)

# Build grouped inputs (expanders)
with st.expander("Blend Composition", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        Fe = st.number_input("Fe (%)", value=float(train_means[FEATURE_ORDER.index('Fe')]))
        SiO2 = st.number_input("SiO₂ (%)", value=float(train_means[FEATURE_ORDER.index('SiO2')]))
    with col2:
        CaO = st.number_input("CaO (%)", value=float(train_means[FEATURE_ORDER.index('CaO')]))
        MgO = st.number_input("MgO (%)", value=float(train_means[FEATURE_ORDER.index('MgO')]))
    with col3:
        Al2O3 = st.number_input("Al₂O₃ (%)", value=float(train_means[FEATURE_ORDER.index('Al2O3')]))
        S_Cu = st.number_input("S/Cu Ratio", value=float(train_means[FEATURE_ORDER.index('S/Cu')]))

with st.expander("S-Furnace Parameters", expanded=False):
    col4, col5 = st.columns(2)
    with col4:
        conc_feed = st.number_input("CONC. FEED RATE", value=float(train_means[FEATURE_ORDER.index('CONC. FEED RATE')]))
        silica_feed = st.number_input("SILICA FEED RATE", value=float(train_means[FEATURE_ORDER.index('SILICA FEED RATE ')]))
        cslag_feed = st.number_input("C-SLAG FEED RATE - S Furnace", value=float(train_means[FEATURE_ORDER.index('C-SLAG FEED RATE - S Furnace')]))
    with col5:
        s_air = st.number_input("S-FURNACE AIR", value=float(train_means[FEATURE_ORDER.index('S-FURNACE AIR')]))
        s_oxygen = st.number_input("S-FURNACE OXYGEN", value=float(train_means[FEATURE_ORDER.index('S-FURNACE OXYGEN')]))

with st.expander("Fe/SiO2 & CLS", expanded=False):
    fe_sio2_ratio = st.number_input("Fe/SiO2", value=float(train_means[FEATURE_ORDER.index('Fe/SiO2')]))
    fe3o4_cls = st.number_input("Fe3O4_Cls", value=float(train_means[FEATURE_ORDER.index('Fe3O4_Cls')]))

with st.expander("Matte Grade", expanded=False):
    matte_grade = st.number_input("Matte Grade", value=float(train_means[FEATURE_ORDER.index('Matte Grade')]))

with st.expander("C-Slag Analysis", expanded=False):
    col6, col7, col8, col9 = st.columns(4)
    with col6:
        cu_cslag = st.number_input("Cu_C_slag", value=float(train_means[FEATURE_ORDER.index('Cu_C_slag')]))
    with col7:
        fe_cslag = st.number_input("Fe_C_slag", value=float(train_means[FEATURE_ORDER.index('Fe_C_slag')]))
    with col8:
        cao_cslag = st.number_input("CaO_C_slag", value=float(train_means[FEATURE_ORDER.index('CaO_C_slag')]))
    with col9:
        fe3o4_cslag = st.number_input("Fe3O4_C_slag", value=float(train_means[FEATURE_ORDER.index('Fe3O4_C_slag')]))

# assemble input_map and vector in FEATURE_ORDER
input_map = {
    'Fe': Fe, 'SiO2': SiO2, 'Al2O3': Al2O3, 'CaO': CaO, 'MgO': MgO, 'S/Cu': S_Cu,
    'CONC. FEED RATE': conc_feed, 'SILICA FEED RATE ': silica_feed, 'C-SLAG FEED RATE - S Furnace': cslag_feed,
    'S-FURNACE AIR': s_air, 'S-FURNACE OXYGEN': s_oxygen,
    'Fe/SiO2': fe_sio2_ratio, 'Fe3O4_Cls': fe3o4_cls, 'Matte Grade': matte_grade,
    'Cu_C_slag': cu_cslag, 'Fe_C_slag': fe_cslag, 'CaO_C_slag': cao_cslag, 'Fe3O4_C_slag': fe3o4_cslag
}
try:
    input_vector = np.array([[ input_map[f] for f in FEATURE_ORDER ]], dtype=float)
except KeyError as ke:
    st.error(f"Feature {ke} not found — check FEATURE_ORDER & input_map keys.")
    st.stop()

# prediction display (modified)
st.markdown("---")
st.subheader("Predict Cl Slag Cu Class")
scaled_in = scaler.transform(input_vector)
pred_class = model.predict(scaled_in)[0]
pred_proba = model.predict_proba(scaled_in)[0]
if pred_class == 1:
    st.success(f"(0.70–0.75 Cu%) — Probability: {pred_proba[1]:.2f}")
else:
    st.error(f"(0.80–0.85 Cu%) — Probability: {pred_proba[0]:.2f}")

# -------------------------
# GA controls: lock features, permitted ranges, GA hyperparams
# -------------------------
st.markdown("---")
st.subheader("Counterfactual Search")

locked_features = st.multiselect("Select features to keep constant:", options=FEATURE_ORDER, default=[])
# features_to_vary is complement
features_to_vary = [f for f in FEATURE_ORDER if f not in locked_features]

# choose desired class
target_choice = st.radio("Target for counterfactuals:", options=["Same as current prediction", "[0.70–0.75 Cu%]", "[0.80–0.85 Cu%]"])
if target_choice == "Same as current prediction":
    desired_class = int(pred_class)
elif target_choice == "[0.70–0.75 Cu%]":
    desired_class = 1
else:
    desired_class = 0

# permitted ranges: use training min/max or allow custom
use_train_ranges = st.checkbox("Use training min/max as permitted ranges", value=True)
permitted_range = {}
if use_train_ranges:
    min_vals = train_mins
    max_vals = train_maxs
    for i, f in enumerate(FEATURE_ORDER):
        permitted_range[f] = [float(min_vals[i]), float(max_vals[i])]
else:
    st.info("You can enter custom min/max ranges")
    for f in FEATURE_ORDER:
        lo = st.number_input(f"Min for {f}", value=float(train_mins[FEATURE_ORDER.index(f)]), key=f"min_{f}")
        hi = st.number_input(f"Max for {f}", value=float(train_maxs[FEATURE_ORDER.index(f)]), key=f"max_{f}")
        permitted_range[f] = [float(lo), float(hi)]

# discretization / integer rounding dictionary (optional)
#st.markdown("Optional: indicate features that should be integers (will be rounded).")
#integer_features = st.multiselect("Features to round to integer:", options=FEATURE_ORDER, default=[])

# GA hyperparameters
st.markdown("GA hyperparameters:")
pop_size = st.slider("Population size", min_value=10, max_value=200, value=40, step=10)
num_generations = st.slider("Generations", min_value=10, max_value=500, value=60, step=10)
mutation_percent_genes = st.slider("Mutation % genes", min_value=1, max_value=50, value=10)
desired_prob = st.slider("Desired probability threshold for target class", min_value=0.5, max_value=0.99, value=0.9, step=0.01)
n_cfs_to_return = st.slider("Number of counterfactual solutions to return (top N)", min_value=1, max_value=5, value=1)

# show-only-changes option
show_only_changes = st.checkbox("Show only changed parameters", value=True)

# -------------------------
# Fitness function for GA
# -------------------------
# We'll maximize: fitness = prob_target - lambda * (normalized_distance)
# so higher prob and smaller distance => higher fitness
LAMBDA = st.number_input("Distance penalty weight λ (higher => smaller changes prioritized)", value=0.5, step=0.1)

# prepare original vector and indexes for convenience
orig = input_vector.flatten()
feat_index = {f: i for i, f in enumerate(FEATURE_ORDER)}

def fitness_func(ga_instance, solution, solution_idx):

    # Convert solution vector → feature dictionary
    sol = pd.DataFrame([solution], columns=FEATURE_ORDER)

    # Scale using your existing scaler
    scaled = scaler.transform(sol)

    # Predict probability for the target class
    proba = model.predict_proba(scaled)[0][desired_class]

    # Apply constraints penalty  
    penalty = 0
    for i, feat in enumerate(FEATURE_ORDER):
        min_v, max_v = FEATURE_BOUNDS[feat]
        val = solution[i]
        if val < min_v:
            penalty += (min_v - val)**2
        if val > max_v:
            penalty += (val - max_v)**2

    # Apply feature-locking penalty
    for i, feat in enumerate(FEATURE_ORDER):
        if feat in locked_features:
            penalty += (solution[i] - original_input[feat])**2 * 200

    # Final fitness = class probability − penalties
    fitness = proba - penalty

    return fitness

# -------------------------
# Build gene_space for pygad (bounds per feature). For locked features put exact value list [val]
# -------------------------
gene_space = []
for i, f in enumerate(FEATURE_ORDER):
    if f in locked_features:
        # locked: fix gene to original value
        gene_space.append([float(orig[i])])
    else:
        lo, hi = permitted_range[f]
        # If integer -> allow integer gene space by giving range and noting later to round in fitness evaluation
        gene_space.append({'low': float(lo), 'high': float(hi)})

# -------------------------
# Run GA when button pressed
# -------------------------
if st.button("Geneting counterfactual(s)"):
    start_time = time.time()
    st.info("Finding counterfactual(s)")
    # pygad expects a fitness function that takes solution and returns fitness
    def pygad_fitness_func(solution, solution_idx):
        return candidate_fitness_func(solution)

    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=max(2, pop_size//4),
        fitness_func=fitness_func,
        sol_per_pop=pop_size,
        num_genes=n_features,
        gene_space=gene_space,
        parent_selection_type="sss",
        keep_parents=2,
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=mutation_percent_genes,
        random_seed=42,
        suppress_warnings=True
    )

    ga_instance.run()
    # collect best solutions (top K unique)
    solutions = []
    pop_solutions = ga_instance.population
    pop_fitness = ga_instance.last_generation_fitness
    # sort by fitness desc
    idx_sorted = np.argsort(pop_fitness)[::-1]
    for idx in idx_sorted:
        sol = pop_solutions[idx]
        # round integer features
        sol_arr = sol.copy()
        for f in integer_features:
            sol_arr[feat_index[f]] = round(sol_arr[feat_index[f]])
        # clamp to permitted ranges
        for i, f in enumerate(FEATURE_ORDER):
            lo, hi = permitted_range[f]
            sol_arr[i] = min(max(sol_arr[i], lo), hi)
        # compute prob and distance
        sol_scaled = scaler.transform(sol_arr.reshape(1,-1))
        probs = model.predict_proba(sol_scaled)[0]
        prob_target = float(probs[desired_class])
        dist = np.linalg.norm(sol_arr - orig)
        solutions.append({'solution': sol_arr, 'fitness': float(pop_fitness[idx]), 'prob_target': prob_target, 'distance': dist, 'probs': probs})
        if len(solutions) >= n_cfs_to_return:
            break

    elapsed = time.time() - start_time
    st.success(f"GA finished in {elapsed:.1f}s — returning top {len(solutions)} solution(s)")

    # display solutions
    for i, soldic in enumerate(solutions):
        sol = soldic['solution']
        probs = soldic['probs']
        prob_target = soldic['prob_target']
        st.write(f"### Counterfactual #{i+1}  — prob_target={prob_target:.4f}, distance={soldic['distance']:.4f}")
        # build dataframe original vs counterfactual vs delta
        df = pd.DataFrame({
            'feature': FEATURE_ORDER,
            'original': orig,
            'counterfactual': sol,
            'delta': sol - orig
        })
        # optionally show only changes
        if show_only_changes:
            df_show = df[df['delta'].abs() > 1e-6].reset_index(drop=True)
            if df_show.shape[0] == 0:
                st.info("No changes were required (counterfactual equals original).")
            else:
                st.dataframe(df_show.style.format({"original":"{:.4f}","counterfactual":"{:.4f}","delta":"{:.4f}"}))
        else:
            st.dataframe(df.style.format({"original":"{:.4f}","counterfactual":"{:.4f}","delta":"{:.4f}"}))

        # predicted probs for this CF
        st.write("Predicted class probabilities for this counterfactual:")
        probs_df = pd.DataFrame([probs], columns=[f"prob_class_{j}" for j in range(len(probs))])
        st.dataframe(probs_df.style.format("{:.4f}"))

st.markdown("---")
#st.caption("Notes: GA searches for candidates within permitted ranges. Locking a feature forces GA to keep it equal to the provided value. Tune population/generations for quality vs. runtime trade-off.")
