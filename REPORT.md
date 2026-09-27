# 📊 Executive Data Science & Business Report: E-Commerce Customer Segmentation (PS-06)

**Project Title:** E-Commerce Customer Behavioral Segmentation & Precision Marketing Strategy  
**Author:** Antigravity Data Science & Analytics Team  
**Date:** September 2026  
**Status:** Production Ready  

---

## Executive Summary

Modern e-commerce enterprises face severe margin compression and customer churn when deploying uniform, "one-size-fits-all" marketing promotions. High-frequency loyal shoppers receive unnecessary margin-diluting discount codes, newly acquired customers suffer from high drop-off due to generic onboarding, and high-value wholesale accounts remain under-monetized.

To solve this, we engineered an end-to-end **RFM (Recency, Frequency, Monetary)** unsupervised machine learning pipeline. Analyzing a benchmark transactional dataset of **4,200 unique customers** spanning **153,892 verified transactions**, our clustering models identified **5 distinct, statistically validated customer segments**.

### Key Commercial Takeaways:
- **Extreme Pareto Revenue Concentration:** Just **25.28% of the customer base** (**Champions** + **Big Spenders / Wholesale**) generates **87.67% of total company revenue** ($5.37M out of $6.12M).
- **High-Leverage Acquisition Window:** **17.76% of customers** are **New & Promising** ($R \approx 19$ days) and represent the primary pipeline for future CLV expansion.
- **Cost Reduction Opportunity:** **20.38% of customers** are **Dormant / Lost** ($R \approx 258$ days). Suppressing this segment from paid ad retargeting immediately eliminates ad spend waste.

---

## 1. Problem Statement & Business Objectives

| Dimension | Description |
|:---|:---|
| **Scenario** | An e-commerce brand with thousands of customers treated all accounts identically during promotional cycles. |
| **Business Friction** | Generic promotional blasting causes 12-18% gross margin dilution on existing organic buyers and delivers subpar conversion on newly acquired shoppers. |
| **Stakeholders** | Chief Marketing Officer (CMO), Retention & CRM Directors, Performance Marketing Managers, B2B Account Executives. |
| **Core Objective** | Construct a robust, automated customer segmentation engine grouping customers by transactional behavior and mapping each group to targeted commercial strategies. |

---

## 2. Dataset Hygiene & Feature Engineering Pipeline

### 2.1 Transaction Ingestion & Cleaning
The transactional records match the schema of the **UCI Online Retail** benchmark. A multi-stage hygiene filter was applied:
1. **Missing Customer Identification:** Removed guest checkouts ($3.0\%$ of raw records) where `CustomerID` was missing.
2. **Order Cancellations & Returns:** Filtered invoice records prefixed with `'C'` and line items with non-positive quantities ($Q \le 0$).
3. **Price Anomalies:** Purged zero or negative unit price entries.
4. **Deduplication:** Eliminated identical duplicate transactional rows.

```
Total Ingested Records:        159,072
Verified Clean Transactions:   153,892 (96.74% retention)
Unique Active Customers:       4,200
```

### 2.2 RFM Feature Formulation
For each customer $i$, we calculated:
- **Recency ($R_i$):** Days between snapshot reference date and the customer's most recent order timestamp:
  $$R_i = t_{\text{snapshot}} - \max(t_{i, \text{orders}})$$
- **Frequency ($F_i$):** Total count of unique purchase invoices placed:
  $$F_i = |\text{Unique Invoices}_i|$$
- **Monetary ($M_i$):** Cumulative gross revenue generated across all valid line items:
  $$M_i = \sum_{j \in \text{Items}_i} (Q_j \times P_j)$$
- **Average Order Value ($AOV_i$):** Average transaction basket size:
  $$AOV_i = \frac{M_i}{F_i}$$

### 2.3 Mathematical Transformation & Normalization
E-commerce transactional metrics exhibit severe right-skewness and heavy Pareto tails. Because distance-based clustering (Euclidean metric) is sensitive to variance scaling and long-tail outliers, we applied:
1. **Variance-Stabilizing Log Transformation:** $x_{\text{log}} = \ln(1 + x)$
2. **Z-Score Normalization:** $z = \frac{x_{\text{log}} - \mu}{\sigma}$

| Feature | Raw Skewness | Log1p Skewness | Skewness Reduction |
|:---|:---:|:---:|:---:|
| **Recency** | $+0.78$ | $-0.32$ | **59.0%** |
| **Frequency** | $+2.41$ | $+0.44$ | **81.7%** |
| **Monetary** | $+3.12$ | $+0.21$ | **93.3%** |
| **Avg Order Value** | $+3.85$ | $+0.18$ | **95.3%** |

---

## 3. Clustering Methodology & Comparative Evaluation

We implemented and compared three clustering algorithms across internal validation metrics:

### 3.1 Algorithm Comparison Matrix

| Algorithm | Clusters ($k$) | Outliers / Noise | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index | Assessment & Practical Fit |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **K-Means ($k=5$)** | **5** | **0** | **0.5276** | **0.6559** | **5,342.05** | **Selected Optimal Model**: Highest CH index, well-balanced spherical cluster boundaries, exceptional commercial interpretability. |
| **Agglomerative (Ward, $k=5$)** | 5 | 0 | 0.5331 | 0.6317 | 5,203.17 | Strong hierarchical structure; closely mirrors K-Means partitions. |
| **DBSCAN ($\epsilon=0.65, \text{min}=15$)** | 4 | 5 | 0.5014 | 0.6822 | 3,680.73 | Excellent at isolating extreme bulk anomalies, but less suited for partitioning continuous customer distributions. |

### 3.2 Hyperparameter Tuning ($k$ Selection)
- **Elbow Method (Inertia SSE):** Inflection point observed cleanly between $k=4$ and $k=5$.
- **Silhouette Coefficient Analysis:** Peaks at $k=5$ with an average coefficient of **0.5276**, indicating well-separated, cohesive clusters.

---

## 4. Final Customer Segment Profiles & Pareto Dynamics

The 4,200 customer accounts partition cleanly into 5 distinct behavioral segments:

```
Total Customer Base: 4,200 accounts
Total Revenue:       $6,126,898.57
```

| Cluster ID | Segment Name | Customer Count | Customer Share | Total Revenue ($) | Revenue Share | Mean Recency | Mean Frequency | Mean Spend ($) | Mean AOV ($) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Champions** | 649 | 15.45% | $2,872,276.77 | **46.88%** | 30.4 days | 15.2 orders | $4,425.70 | $283.17 |
| **4** | **Big Spenders / Wholesale** | 413 | 9.83% | $2,499,155.06 | **40.79%** | 144.5 days | 2.0 orders | $6,051.22 | $3,034.99 |
| **2** | **Loyal / Regulars** | 1,536 | 36.57% | $699,424.19 | **11.42%** | 120.9 days | 6.6 orders | $455.35 | $67.55 |
| **3** | **New & Promising** | 746 | 17.76% | $38,222.46 | **0.62%** | 19.4 days | 1.5 orders | $51.24 | $33.29 |
| **0** | **Dormant / Lost** | 856 | 20.38% | $17,820.09 | **0.29%** | 257.7 days | 1.0 orders | $20.82 | $19.85 |

---

## 5. Strategic Marketing Action Playbook

```
+---------------------------------------------------------------------------------------------------+
|                                 MARKETING STRATEGY MATRIX                                        |
+--------------------------+------------------------------------+-----------------------------------+
| SEGMENT                  | STRATEGIC GOAL                     | PRIMARY CAMPAIGN TACTIC           |
+--------------------------+------------------------------------+-----------------------------------+
| Champions                | Retention & Brand Advocacy         | VIP Concierge & Early Product Drop|
| Big Spenders / Wholesale | B2B Expansion & Order Frequency    | Dedicated Account Rep & Terms     |
| Loyal / Regulars         | Basket Size Growth & Cross-Sell    | Tiered Loyalty Points & Bundles   |
| New & Promising          | Onboarding & 2nd Order Conversion  | 14-Day Welcome Drip + 15% Incentive|
| Dormant / Lost           | Low-Cost Reactivation / List Pruning| Automated Clearance Newsletter    |
+--------------------------+------------------------------------+-----------------------------------+
```

### 1. Champions ($n=649$, 46.88% Revenue) — Priority: P1 Critical
- **Profile:** Highly active, repeat buyers placing orders every 2-3 weeks with substantial baskets.
- **Recommended Action:** Enroll into an elite VIP Tier ("Gold Circle"). Grant 48-hour early access to new product drops, personalized unboxing gifts, direct WhatsApp concierge, and double referral bonuses.
- **Critical Policy:** **Zero generic discount blasting.** Do not discount products they would buy at full margin.
- **Target KPIs:** Customer Lifetime Value (CLV), Net Promoter Score (NPS), Customer Referral Rate.

### 2. Big Spenders / Wholesale ($n=413$, 40.79% Revenue) — Priority: P2 High
- **Profile:** Corporate, wholesale, or affluent buyers with massive Average Order Value ($AOV = \$3,035$) but low annual frequency ($F \approx 2.0$).
- **Recommended Action:** Assign dedicated B2B account managers. Offer tiered wholesale bulk volume discounts, recurring scheduled subscription deliveries, and NET-30 invoicing terms.
- **Target KPIs:** Reorder Interval, Units per Transaction (UPT), Account Retention.

### 3. Loyal / Regulars ($n=1,536$, 11.42% Revenue) — Priority: P2 Standard
- **Profile:** Steady, dependable everyday buyers ($F \approx 6.6$ orders, $M \approx \$455$).
- **Recommended Action:** Gamified loyalty rewards program with milestone unlock thresholds. Cross-sell complementary accessory bundles using collaborative filtering recommendations.
- **Target KPIs:** Average Basket Size, Category Cross-Purchase Rate, Annual Retention Rate.

### 4. New & Promising ($n=746$, 0.62% Revenue) — Priority: P1 Immediate
- **Profile:** Recently acquired customers ($R \approx 19.4$ days) who completed their 1st or 2nd purchase.
- **Recommended Action:** Automated 4-part onboarding email drip (Brand Story $\rightarrow$ How-to Guide $\rightarrow$ Customer Reviews $\rightarrow$ Time-Limited 15% Off 2nd Order within 14 days).
- **Target KPIs:** First-to-Second Purchase Conversion Rate, Time-to-Second Order.

### 5. Dormant / Lost ($n=856$, 0.29% Revenue) — Priority: P3 Low Cost
- **Profile:** Long-inactive single-purchase shoppers ($R \approx 258$ days, $M \approx \$20.82$).
- **Recommended Action:** Exclude from expensive paid social/search retargeting audiences (saving ad spend). Route through automated bi-monthly clearance email drops. If inactive after 365 days, prune from mailing list to protect email sender domain reputation.
- **Target KPIs:** Reactivation ROI, Unsubscribe Rate, Email Deliverability Score.

---

## 6. Generated Visual Artifacts

All charts and tables have been generated and exported to the repository:

1. [`01_raw_rfm_distributions.png`](file:///d:/ECommerce/outputs/figures/01_raw_rfm_distributions.png): Demonstrates severe positive right-skewness across raw features.
2. [`02_transformed_rfm_distributions.png`](file:///d:/ECommerce/outputs/figures/02_transformed_rfm_distributions.png): Confirms normalized Gaussian-like distributions post-$\log(1+x)$.
3. [`03_rfm_correlation_heatmap.png`](file:///d:/ECommerce/outputs/figures/03_rfm_correlation_heatmap.png): Pearson correlation matrix across feature variants.
4. [`04_kmeans_elbow_silhouette.png`](file:///d:/ECommerce/outputs/figures/04_kmeans_elbow_silhouette.png): Elbow SSE and Silhouette coefficient curves justifying $k=5$.
5. [`05_hierarchical_dendrogram.png`](file:///d:/ECommerce/outputs/figures/05_hierarchical_dendrogram.png): Ward Euclidean distance linkage dendrogram.
6. [`06_clustering_comparison_pca.png`](file:///d:/ECommerce/outputs/figures/06_clustering_comparison_pca.png): 2D PCA latent projections across K-Means, Agglomerative, and DBSCAN.
7. [`07_dbscan_noise_and_density.png`](file:///d:/ECommerce/outputs/figures/07_dbscan_noise_and_density.png): DBSCAN core clusters vs isolated anomaly/noise points.
8. [`08_final_cluster_rfm_boxplots.png`](file:///d:/ECommerce/outputs/figures/08_final_cluster_rfm_boxplots.png): Segment comparison boxplots on log-transformed scales.
9. [`09_segment_size_distribution.png`](file:///d:/ECommerce/outputs/figures/09_segment_size_distribution.png): Visual Pareto distribution of customer share vs revenue contribution.
10. [`10_radar_chart_profiles.png`](file:///d:/ECommerce/outputs/figures/10_radar_chart_profiles.png): Multi-axis radar diagram displaying normalized segment profiles.
11. [`11_interactive_3d_rfm.html`](file:///d:/ECommerce/outputs/figures/11_interactive_3d_rfm.html): Interactive 3D spatial visualizer.

---

## 7. Recommendations & Next Steps

1. **CRM & ESP Integration:** Export `data/customer_rfm_segmented.csv` into Klaviyo, Braze, or HubSpot as custom user attributes (`RFM_Segment = 'Champions'`).
2. **Automated Batch Scoring Pipeline:** Schedule a weekly Airflow/Prefect or cron job to re-calculate customer RFM scores dynamically as new transactions stream in.
3. **A/B Testing Framework:** Execute an A/B test on the **Champions** segment: Test VIP Non-Discounted Perks vs. Standard Promo Discounter to quantify gross margin recovery.
