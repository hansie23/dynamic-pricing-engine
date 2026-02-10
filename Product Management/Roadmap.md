# Product Roadmap: Airbnb Dynamic Pricing Engine

## Phase 1: MVP Deployment (Completed - Feb 2026)
*Goal: Establish a functional proof-of-concept for London Airbnb hosts.*
- [x] **Core Optimization Engine:** Implemented simulation-based price optimization using 100-point testing.
- [x] **Interactive Dashboard:** Deployed a Streamlit UI allowing listing lookup and parameter adjustment.
- [x] **Revenue Analytics:** Automated calculation of "Potential Revenue Uplift" with visual delta metrics.
- [x] **Visual Trend Forecasting:** Line charts for price and revenue trends over a 30-day window.
- [x] **Robust Error Handling:** Validations for Listing IDs and system file integrity.

## Phase 2: Market Refinement & Value Validation (Next Iteration)
*Goal: Move from "Technical Success" to "Commercial Viability".*
- [ ] **External Signal Integration:** Incorporate London-specific seasonal events (holidays, festivals) to move beyond historical averages.
- [ ] **Exportable Intelligence:** Allow hosts to download the "Optimized Schedule" as CSV for manual upload to Airbnb.
- [ ] **A/B Performance Tracking:** Enable a basic feature to compare predicted vs. actual bookings for specific listings to prove ROI.
- [ ] **Contextual Benchmarking:** Add a "Neighborhood Average" baseline to the trend charts for better competitive context.

---

## Strategic Outlook: Go/No-Go Decision
**Recommendation:** Proceed to Phase 2, then re-evaluate.

**Reasoning:**
The MVP demonstrates high technical potential, but the "Product Management" challenge is proving actual ROI to non-technical hosts. 

**Proceed if:**
1.  **Metric Lift:** Initial pilot users report a >5% increase in actual bookings or revenue compared to their previous static pricing.
2.  **Usability:** Hosts are able to interpret the "Suggested Price" without assistance.
3.  **Data Quality:** The model remains accurate across different London boroughs (Zone 1 vs. Zone 4).

**Pivot or Stop if:**
1.  **Friction:** The manual step of copying prices from this tool to Airbnb is too high, necessitating an immediate pivot to API automation (high effort/high risk).
2.  **Accuracy Gap:** Predicted revenue uplift does not materialize in real-world bookings due to unmodeled external factors (e.g., rapid changes in transport/regulations).