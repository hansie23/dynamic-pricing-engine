# Prioritization Matrix (RICE)

This document ranks future features for the Airbnb Dynamic Pricing Engine to determine the development order.

| Feature | Reach (R) | Impact (I) | Confidence (C) | Effort (E) | RICE Score | Priority |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seasonal Event Detection** | 10 | 3 | 80% | 2 | **12.0** | High |
| **Host Dashboard / Auth** | 8 | 2 | 90% | 3 | **4.8** | Medium |
| **Direct Airbnb API Sync** | 5 | 5 | 70% | 8 | **2.18** | Medium |
| **Live Competitor Tracking** | 10 | 2 | 60% | 5 | **2.4** | Medium |
| **Multi-Currency Support** | 4 | 1 | 100% | 1 | **4.0** | Medium |
| **Mobile App Wrapper** | 3 | 1 | 90% | 4 | **0.67** | Low |

---

## Score Justifications

### 1. Seasonal Event Detection
*   **Reach (10):** Every listing in the London market is affected by major events (Wimbledon, New Year's Eve, major concerts).
*   **Impact (3):** Massive revenue impact; missing a peak event can mean losing 50%+ of potential monthly revenue.
*   **Confidence (80%):** High confidence that event dates are predictable, though the exact price multiplier requires model tuning.
*   **Effort (2):** Relatively low; primarily involves adding an external "events calendar" dataset to the feature engineering pipeline.

### 2. Host Dashboard / Auth
*   **Reach (8):** Essential for any returning user who wants to manage a portfolio of listings rather than one-off searches.
*   **Impact (2):** High impact on user retention and platform stickiness.
*   **Confidence (90%):** Standard implementation patterns (OAuth/JWT) make this highly predictable.
*   **Effort (3):** Moderate; requires database schema changes and frontend state management.

### 3. Direct Airbnb API Sync
*   **Reach (5):** Only "Power Users" or property managers will initially utilize automated syncing.
*   **Impact (5):** Massive impact on value proposition—it transforms the tool from a "calculator" to an "automated manager."
*   **Confidence (70%):** Lower confidence due to Airbnb's restrictive API access and potential approval delays.
*   **Effort (8):** Very high; requires complex security, error handling for failed syncs, and official API partnership.

### 4. Live Competitor Tracking
*   **Reach (10):** All hosts are interested in what their immediate neighbors are charging.
*   **Impact (2):** Significant, but current model already uses "neighborhood averages" which captures some of this.
*   **Confidence (60%):** Scraping or buying live data can be unreliable or expensive.
*   **Effort (5):** High; involves building a scraping engine or integrating third-party market data providers.

### 5. Multi-Currency Support
*   **Reach (4):** Most current users are London-based and use GBP, but international investors need this.
*   **Impact (1):** Medium; a convenience feature rather than a revenue driver.
*   **Confidence (100%):** Extremely simple to implement using standard exchange rate APIs.
*   **Effort (1):** Negligible; simple UI toggle and a math multiplier.

### 6. Mobile App Wrapper
*   **Reach (3):** Low, as the current Streamlit UI is already mobile-responsive in the browser.
*   **Impact (1):** Low; adds "presence" but minimal new functionality.
*   **Confidence (90%):** Wrapping a web app in a WebView is a solved problem.
*   **Effort (4):** Moderate; requires app store submission processes and platform-specific testing.

---

### Scoring Guide:
- **Reach:** Estimated number of users/listings impacted per month. (1-10)
- **Impact:** How much will this increase revenue/retention? (3 = Massive, 2 = High, 1 = Medium, 0.5 = Low)
- **Confidence:** How sure are we about these estimates? (100% = High, 80% = Medium, 50% = Low)
- **Effort:** Person-months or complexity scale. (1 = Low, 10 = Very High)

**RICE Score Formula:** `(Reach * Impact * Confidence) / Effort`