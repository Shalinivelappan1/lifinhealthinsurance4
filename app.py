import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# =====================================================
# Session State
# =====================================================
if "has_results" not in st.session_state:
    st.session_state.has_results = False

if "submission_content" not in st.session_state:
    st.session_state.submission_content = None

if "results" not in st.session_state:
    st.session_state.results = None

# =====================================================
# Page Config
# =====================================================
st.set_page_config(page_title="LiFin Health Insurance Lab", layout="wide")

st.title("🏥 LiFin Health Insurance Lab — Developed by Prof. Shalini Velappan, IIM Trichy")

st.caption("""
This is a **teaching simulator** to understand:

• Deductibles, co-pay, room rent limits  
• Family floater vs separate parents policies  
• Super top-up layers  
• No Claim Bonus (NCB)  
• Zonal pricing (city cost differences)  
• Risk distributions & tail risk over 30 years  
• Probability of financial ruin relative to income & savings  

It is **not** an actuarial pricing engine.
""")

# =====================================================
# Teaching Explainers
# =====================================================
with st.expander("📘 How health insurance really works"):
    st.markdown("""
Insurance is about **destroying tail risk**, not eliminating average cost.
""")

with st.expander("📘 What is Probability of Ruin?"):
    st.markdown("""
> Ruin = Lifetime medical cost > (Savings + X × Annual Income)
""")

with st.expander("📘 How to read the charts"):
    st.markdown("""
Mean is misleading. Tails matter.
""")

# =====================================================
# Sidebar: Financial Capacity
# =====================================================
st.sidebar.header("💰 Household Financial Capacity")

income = st.sidebar.number_input("Annual Household Income (₹)", value=1_200_000, step=100_000)
savings = st.sidebar.number_input("Current Savings (₹)", value=800_000, step=100_000)

ruin_multiple = st.sidebar.slider("Ruin if Medical Cost Exceeds (X × Income + Savings)", 1, 15, 5)
ruin_threshold = savings + ruin_multiple * income

# =====================================================
# Family
# =====================================================
st.sidebar.header("👨‍👩‍👧‍👦 Family")

include_spouse = st.sidebar.checkbox("Include Spouse", True)
num_children = st.sidebar.selectbox("Children", [0,1,2], 1)
num_parents = st.sidebar.selectbox("Parents", [0,1,2], 2)

members = ["Self"]
if include_spouse: members.append("Spouse")
for i in range(num_children): members.append(f"Child {i+1}")
for i in range(num_parents): members.append(f"Parent {i+1}")

# =====================================================
# Risk + Environment (shortened, logic unchanged)
# =====================================================
years = 30
inflation = 0.08

p_normal_base = 0.05
p_major_base = 0.01
p_normal_parent = 0.12
p_major_parent = 0.04
parent_cost_multiplier = 1.6

normal_min, normal_max = 50_000, 300_000
major_min, major_max = 500_000, 2_500_000
zone_multiplier = 1.2

# =====================================================
# Dummy Policies (keep simple)
# =====================================================
policyA = {"cover": 1_000_000, "deductible": 50_000, "copay": 0.1, "room_limit": 5000, "premium": 25_000,
           "has_topup": True, "topup_cover": 2_000_000, "topup_threshold": 1_000_000, "topup_premium": 10_000,
           "ncb_rate": 0.1, "ncb_cap": 1.0}

policyB = {"cover": 2_500_000, "deductible": 0, "copay": 0.0, "room_limit": 10000, "premium": 55_000,
           "has_topup": True, "topup_cover": 3_000_000, "topup_threshold": 2_500_000, "topup_premium": 15_000,
           "ncb_rate": 0.1, "ncb_cap": 1.0}

# =====================================================
# Simulation Functions
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

    room = np.random.choice([3000,5000,8000,12000]) if cost>0 else 0
    return cost * zone_multiplier, room

def apply_policy(policy, total_claim, avg_room_cost, effective_cover):
    if total_claim <= 0: return 0
    ratio = min(1, policy["room_limit"]/avg_room_cost) if avg_room_cost>0 else 1
    admissible = total_claim * ratio
    remaining = max(admissible - policy["deductible"], 0)
    insurer_share = remaining * (1 - policy["copay"])
    paid = min(insurer_share, effective_cover)
    return total_claim - paid

def simulate_lifetime(policy):
    total = 0
    for t in range(years):
        infl = (1+inflation)**t
        yearly = 0
        for m in members:
            is_parent = "Parent" in m
            cost,_ = simulate_year_for_member(is_parent)
            yearly += cost*infl
        oop = apply_policy(policy, yearly, 6000, policy["cover"])
        total += oop
    total += policy["premium"]*years
    return total

# =====================================================
# Run Button
# =====================================================
st.markdown("## 🎲 Monte Carlo Simulation")

sims = st.slider("Simulations", 500, 3000, 1500, 500)

if st.button("▶️ Run Simulation"):
    no_ins, polA, polB = [], [], []

    for _ in range(sims):
        no_ins.append(simulate_lifetime({"cover":0,"deductible":0,"copay":0,"room_limit":1e9,"premium":0,"has_topup":False}))
        polA.append(simulate_lifetime(policyA))
        polB.append(simulate_lifetime(policyB))

    no_ins = np.array(no_ins)
    polA = np.array(polA)
    polB = np.array(polB)

    st.session_state.results = (no_ins, polA, polB)
    st.session_state.has_results = True

# =====================================================
# SHOW RESULTS IF AVAILABLE
# =====================================================
if st.session_state.has_results:

    no_ins, polA, polB = st.session_state.results

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

    st.markdown("## 📋 Decision Table")
    st.table(table)

    # =====================================================
    # ASSIGNMENT (NOW WORKS)
    # =====================================================
    st.markdown("## 📝 Student Reflection & Submission")

    name = st.text_input("Name")
    roll = st.text_input("Roll No")
    q1 = st.text_area("1) Which policy would you choose and why?")
    q2 = st.text_area("2) Why is mean misleading?")

    if st.button("📄 Generate Submission"):
        st.session_state.submission_content = f"""
Name: {name}
Roll: {roll}

Decision Table:
{table}

Answers:
Q1: {q1}
Q2: {q2}
"""

    if st.session_state.submission_content is not None:
        st.download_button(
            "⬇️ Download Submission",
            st.session_state.submission_content,
            file_name=f"{name}_health_assignment.txt",
            mime="text/plain"
        )
