import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# =====================================================
# Session State (CRITICAL FIX)
# =====================================================
if "has_results" not in st.session_state:
    st.session_state.has_results = False

if "results_bundle" not in st.session_state:
    st.session_state.results_bundle = None

if "submission_content" not in st.session_state:
    st.session_state.submission_content = None

# =====================================================
# Page Config
# =====================================================
st.set_page_config(page_title="LiFin Health Insurance Lab", layout="wide")

st.title("🏥 LiFin Health Insurance Lab — Developed by Prof. Shalini Velappan, IIM Trichy")

st.caption("""
This is a **teaching simulator** to understand:

• Family floater vs separate parents policies  
• Room rent proportional deduction (Indian reality)  
• Deductibles, co-pay, sub-limits  
• Super top-up layers  
• No Claim Bonus (NCB)  
• Zonal pricing (city cost differences)  
• Risk distributions & tail risk over 30 years  
• Probability of financial ruin relative to income & savings  

It is **not** an actuarial pricing engine.
""")

# =====================================================
# Sidebar: Financial Capacity
# =====================================================
st.sidebar.header("💰 Household Financial Capacity")

income = st.sidebar.number_input("Annual Household Income (₹)", value=1_200_000, step=100_000)
savings = st.sidebar.number_input("Current Savings (₹)", value=800_000, step=100_000)

ruin_multiple = st.sidebar.slider("Ruin if Medical Cost Exceeds (X × Income + Savings)", 1, 15, 5)
ruin_threshold = savings + ruin_multiple * income

st.sidebar.markdown(f"**Ruin Threshold = ₹ {ruin_threshold:,.0f}**")

# =====================================================
# Sidebar: Family Structure
# =====================================================
st.sidebar.header("👨‍👩‍👧‍👦 Family Structure")

include_spouse = st.sidebar.checkbox("Include Spouse", value=True)
num_children = st.sidebar.selectbox("Number of Children", [0, 1, 2], index=1)
num_parents = st.sidebar.selectbox("Number of Parents", [0, 1, 2], index=2)

members = ["Self"]
if include_spouse:
    members.append("Spouse")
for i in range(num_children):
    members.append(f"Child {i+1}")
for i in range(num_parents):
    members.append(f"Parent {i+1}")

st.sidebar.markdown(f"**Total Covered Members:** {len(members)}")

# =====================================================
# Sidebar: Stress Tests
# =====================================================
st.sidebar.header("🔥 Stress Tests")

stress_parent = st.sidebar.checkbox("Double Parent Risk")
stress_metro = st.sidebar.checkbox("Force Metro Zone")
stress_inflation = st.sidebar.checkbox("High Medical Inflation (12%)")

# =====================================================
# Sidebar: Zone
# =====================================================
st.sidebar.header("🌍 Treatment City Zone")

zone = st.sidebar.selectbox("City Zone", ["A (Metro)", "B (Tier-2)", "C (Small city)"])
zone_multiplier = {"A (Metro)": 1.4, "B (Tier-2)": 1.1, "C (Small city)": 0.9}[zone]

if stress_metro:
    zone_multiplier = 1.4

# =====================================================
# Sidebar: Medical Environment
# =====================================================
st.sidebar.header("📈 Medical Environment")

inflation = st.sidebar.slider("Medical Inflation (%)", 4.0, 15.0, 8.0) / 100
if stress_inflation:
    inflation = 0.12

years = 30

normal_min = st.sidebar.number_input("Normal Event Cost - Min (₹)", value=50_000, step=10_000)
normal_max = st.sidebar.number_input("Normal Event Cost - Max (₹)", value=300_000, step=50_000)

major_min = st.sidebar.number_input("Major Event Cost - Min (₹)", value=500_000, step=100_000)
major_max = st.sidebar.number_input("Major Event Cost - Max (₹)", value=2_500_000, step=100_000)

# =====================================================
# Sidebar: Risk Parameters
# =====================================================
st.sidebar.header("⚠️ Annual Risk Parameters")

p_normal_base = st.sidebar.slider("Prob. Normal Hosp (%) Non-Parents", 0.0, 20.0, 5.0) / 100
p_major_base = st.sidebar.slider("Prob. Major Illness (%) Non-Parents", 0.0, 10.0, 1.0) / 100

st.sidebar.markdown("**Parents Risk (User Controlled)**")
p_normal_parent = st.sidebar.slider("Parent: Prob. Normal Hosp (%)", 0.0, 30.0, 12.0) / 100
p_major_parent = st.sidebar.slider("Parent: Prob. Major Illness (%)", 0.0, 20.0, 4.0) / 100

if stress_parent:
    p_normal_parent *= 2
    p_major_parent *= 2

parent_cost_multiplier = st.sidebar.slider("Parent Cost Multiplier", 1.0, 3.0, 1.6)

# =====================================================
# Preset Policies
# =====================================================
PRESETS = {
    "Starter": {"cover": 500_000, "deductible": 100_000, "copay": 0.2, "room": 3000, "premium": 12_000, "topup": False},
    "Standard": {"cover": 1_000_000, "deductible": 50_000, "copay": 0.1, "room": 5000, "premium": 25_000, "topup": True},
    "Premium": {"cover": 2_500_000, "deductible": 0, "copay": 0.0, "room": 10000, "premium": 55_000, "topup": True},
}

# =====================================================
# Policy UI
# =====================================================
def policy_ui(label, default_preset):
    st.sidebar.header(f"📜 {label}")

    preset = st.sidebar.selectbox(f"{label} Preset", ["Custom", "Starter", "Standard", "Premium"],
                                  index=["Custom","Starter","Standard","Premium"].index(default_preset))

    base = PRESETS[preset] if preset != "Custom" else None

    floater = st.sidebar.checkbox(f"{label}: Family Floater", value=True)

    cover = st.sidebar.number_input(f"{label}: Base Cover (₹)", value=(base["cover"] if base else 1_000_000), step=100_000)
    deductible = st.sidebar.number_input(f"{label}: Deductible (₹)", value=(base["deductible"] if base else 50_000), step=50_000)
    copay = st.sidebar.slider(f"{label}: Co-pay (%)", 0, 50, int((base["copay"]*100) if base else 10)) / 100
    room_limit = st.sidebar.number_input(f"{label}: Room Rent Limit (₹)", value=(base["room"] if base else 5000), step=1000)
    premium = st.sidebar.number_input(f"{label}: Annual Premium (₹)", value=(base["premium"] if base else 25_000), step=2000)

    st.sidebar.markdown(f"**{label}: Super Top-Up**")
    has_topup = st.sidebar.checkbox(f"{label}: Enable Super Top-Up", value=(base["topup"] if base else False))
    topup_cover = st.sidebar.number_input(f"{label}: Top-Up Cover (₹)", value=2_000_000, step=500_000)
    topup_threshold = st.sidebar.number_input(f"{label}: Top-Up Threshold (₹)", value=cover, step=100_000)
    topup_premium = st.sidebar.number_input(f"{label}: Top-Up Premium (₹)", value=10_000, step=1_000)

    st.sidebar.markdown(f"**{label}: No Claim Bonus**")
    ncb_rate = st.sidebar.slider(f"{label}: NCB increase per no-claim year (%)", 0, 50, 10) / 100
    ncb_cap = st.sidebar.slider(f"{label}: NCB Max Cap (% of base cover)", 0, 200, 100) / 100

    return {
        "floater": floater,
        "cover": cover,
        "deductible": deductible,
        "copay": copay,
        "room_limit": room_limit,
        "premium": premium,
        "has_topup": has_topup,
        "topup_cover": topup_cover,
        "topup_threshold": topup_threshold,
        "topup_premium": topup_premium,
        "ncb_rate": ncb_rate,
        "ncb_cap": ncb_cap
    }

policyA = policy_ui("Policy A", "Standard")
policyB = policy_ui("Policy B", "Premium")

# =====================================================
# Simulation Engine
# =====================================================
def simulate_year_for_member(is_parent):
    if is_parent:
        pN, pM, mult = p_normal_parent, p_major_parent, parent_cost_multiplier
    else:
        pN, pM, mult = p_normal_base, p_major_base, 1.0

    u = np.random.rand()
    if u < pM:
        cost = np.random.uniform(major_min, major_max) * mult
    elif u < pM + pN:
        cost = np.random.uniform(normal_min, normal_max) * mult
    else:
        cost = 0

    room = np.random.choice([3000, 5000, 8000, 12000]) if cost > 0 else 0
    return cost * zone_multiplier, room

def apply_policy(policy, total_claim, avg_room_cost, effective_cover):
    if total_claim <= 0:
        return 0

    if avg_room_cost > policy["room_limit"]:
        ratio = policy["room_limit"] / avg_room_cost
    else:
        ratio = 1.0

    admissible = total_claim * ratio
    remaining = max(admissible - policy["deductible"], 0)
    insurer_share = remaining * (1 - policy["copay"])

    paid_by_base = min(insurer_share, effective_cover)
    oop = total_claim - paid_by_base
    return oop

def simulate_lifetime(policy):
    total_oop = 0
    ncb_multiplier = 1.0

    for t in range(years):
        infl = (1 + inflation) ** t
        yearly = 0
        rooms = []

        for m in members:
            is_parent = "Parent" in m
            cost, room = simulate_year_for_member(is_parent)
            yearly += cost * infl
            if cost > 0:
                rooms.append(room)

        effective_cover = policy["cover"] * min(ncb_multiplier, 1 + policy["ncb_cap"])

        if yearly > 0:
            avg_room = np.mean(rooms)
            oop = apply_policy(policy, yearly, avg_room, effective_cover)
            total_oop += oop
            ncb_multiplier = 1.0
        else:
            ncb_multiplier *= (1 + policy["ncb_rate"])

    total_oop += policy["premium"] * years
    return total_oop

# =====================================================
# Run Simulation Button
# =====================================================
st.markdown("## 🎲 Monte Carlo Simulation")

sims = st.slider("Number of Simulation Paths", 500, 4000, 2000, step=500)

if st.button("▶️ Run Simulation"):
    no_ins, polA, polB = [], [], []

    for _ in range(sims):
        no_ins.append(simulate_lifetime({"cover":0,"deductible":0,"copay":0,"room_limit":1e9,"premium":0,"ncb_rate":0,"ncb_cap":0}))
        polA.append(simulate_lifetime(policyA))
        polB.append(simulate_lifetime(policyB))

    st.session_state.results_bundle = {
        "no_ins": np.array(no_ins),
        "polA": np.array(polA),
        "polB": np.array(polB),
    }
    st.session_state.has_results = True

# =====================================================
# Display Results Persistently
# =====================================================
if st.session_state.has_results:

    no_ins = st.session_state.results_bundle["no_ins"]
    polA = st.session_state.results_bundle["polA"]
    polB = st.session_state.results_bundle["polB"]

    def stats(x):
        return np.mean(x), np.percentile(x,95), np.percentile(x,99), np.max(x)

    s0 = stats(no_ins)
    sA = stats(polA)
    sB = stats(polB)

    table = {
        "Metric": ["Mean", "95%", "99%", "Worst"],
        "No Insurance": [f"{s0[0]:,.0f}", f"{s0[1]:,.0f}", f"{s0[2]:,.0f}", f"{s0[3]:,.0f}"],
        "Policy A": [f"{sA[0]:,.0f}", f"{sA[1]:,.0f}", f"{sA[2]:,.0f}", f"{sA[3]:,.0f}"],
        "Policy B": [f"{sB[0]:,.0f}", f"{sB[1]:,.0f}", f"{sB[2]:,.0f}", f"{sB[3]:,.0f}"],
    }

    st.markdown("## 📋 Decision Summary Table")
    st.table(table)

    # =====================================================
    # Assignment
    # =====================================================
    st.markdown("## 📝 Student Reflection")

    name = st.text_input("Name")
    roll = st.text_input("Roll No")

    q1 = st.text_area("1) Which policy would you choose and why?")
    q2 = st.text_area("2) Why is mean misleading?")
    q3 = st.text_area("3) What role does room rent play?")
    q4 = st.text_area("4) Why is super top-up powerful?")
    q5 = st.text_area("5) Explain probability of ruin.")

    if st.button("📄 Generate Submission"):
        st.session_state.submission_content = f"""
Name: {name}
Roll: {roll}

Decision Table:
{table}

Answers:
Q1: {q1}
Q2: {q2}
Q3: {q3}
Q4: {q4}
Q5: {q5}
"""

    if st.session_state.submission_content is not None:
        st.download_button(
            "⬇️ Download Submission",
            st.session_state.submission_content,
            file_name=f"{name}_health_insurance_assignment.txt",
            mime="text/plain"
        )

# =====================================================
# Footer
# =====================================================
st.markdown("""
---
⚠️ This is a **teaching simulator**, not a pricing or underwriting engine.
""")
