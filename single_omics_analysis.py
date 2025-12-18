# After you compute `consistency = soa.compute_consistency_score(...)`

defs = consistency.get("definitions", {})

st.markdown("---")
st.markdown("### Evidence summary")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Evidence Score", f"{consistency['evidence_score']:.1%}")
    st.caption(defs.get("evidence_score", ""))

with c2:
    st.metric("Median AUC (disc)", f"{consistency.get('auc_median_discriminative', np.nan):.3f}")
    st.caption(defs.get("auc_median_discriminative", ""))

with c3:
    st.metric("Direction Agreement", f"{consistency['direction_agreement']:.1%}")
    st.caption(defs.get("direction_agreement", ""))

with c4:
    st.metric("Stability", f"{consistency.get('stability', np.nan):.3f}")
    st.caption(defs.get("stability", ""))

st.markdown("### Score components")

d1, d2, d3, d4 = st.columns(4)
with d1:
    st.metric("Strength", f"{consistency.get('strength', np.nan):.3f}")
    st.caption(defs.get("strength", ""))

with d2:
    st.metric("Study Weight", f"{consistency.get('n_weight', np.nan):.3f}")
    st.caption(defs.get("n_weight", ""))

with d3:
    st.metric("Valid AUROC (n)", f"{consistency.get('n_auc', 0)}")
    st.caption("Number of studies contributing AUROC to the score.")

with d4:
    st.metric("Studies Found", f"{consistency['found_count']}")
    st.caption("Number of studies where the gene was detected (even if AUROC missing).")

st.markdown("### Diagnostics (optional but useful)")

e1, e2, e3 = st.columns(3)
with e1:
    st.metric("Median AUC (raw)", f"{consistency.get('auc_median', np.nan):.3f}")
    st.caption(defs.get("auc_median", ""))

with e2:
    st.metric("Median AUC (oriented)", f"{consistency.get('auc_median_oriented', np.nan):.3f}")
    st.caption(defs.get("auc_median_oriented", ""))

with e3:
    st.metric("disc_full", f"{consistency.get('disc_full', 0.65):.2f}")
    st.caption("Threshold where Strength reaches 1.0.")
