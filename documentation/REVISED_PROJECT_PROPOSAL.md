# Project Proposal: Explainable Network Intrusion Detection for Automated SOC Triage

---

## Project Title
**Reducing Alert Fatigue Through Explainable Network Intrusion Detection: An XGBoost-SHAP Framework for Tier 1 Security Operations Center Automation**

---

## Group Members
**Team Group 3:** Lopez, Itzalen; Frankyan, Shahane; Shanbhag, Nethra

**Affiliation:** AI4ALL Explainable AI (XAI) Project

**Date:** November 2025

---

## 1. Project Overview and Research Question

### Research Question
**How can Explainable AI (XAI), specifically SHAP-enhanced XGBoost models, reduce alert fatigue and improve efficiency in Security Operations Center (SOC) Tier 1 network intrusion detection triage through automated, transparent, and actionable threat explanations?**

### Project Summary

Security Operations Centers (SOCs) face severe **alert fatigue**, with Tier 1 analysts manually triaging thousands of network intrusion alerts daily. Traditional machine learning intrusion detection systems (IDS) achieve high accuracy but operate as "black boxes," reducing analyst trust and requiring time-consuming manual investigation of every alert.

This project develops an **Explainable Network Intrusion Detection System (X-IDS)** that combines:
1. **Detection Layer:** XGBoost classifier trained on the CICIDS2017 benchmark dataset (2.8M+ network flow records)
2. **Explainability Layer:** SHAP (SHapley Additive exPlanations) to generate human-readable justifications for each prediction
3. **Automation Layer:** Template-based case report generation for Tier 1 analyst triage

**Core Innovation:** Rather than just predicting "attack" or "benign," our system explains **WHY** it made the prediction using feature contributions (e.g., "High SYN flag count + sequential destination ports + short duration → Port Scan attack"), enabling analysts to quickly validate or challenge the system's reasoning.

**Expected Impact:**
- **90%+ reduction in per-alert triage time** (from 5 minutes manual analysis to 30 seconds XAI report review)
- **95%+ detection accuracy** with <5% false positive rate
- **Transparent decision-making** that maintains analyst trust and enables human oversight
- **Automated handling** of 80%+ of benign traffic, freeing analysts for complex threat hunting

This project directly addresses the gap between high-performing ML models and real-world SOC adoption by making intrusion detection **explainable, trustworthy, and operationally viable**.

---

## 2. Type of Machine Learning Algorithm(s)

### Primary Algorithm: XGBoost (Extreme Gradient Boosting)

**Why XGBoost for Network IDS?**

1. **State-of-the-art Performance on Tabular Data**
   - Network flow data (CICIDS2017) is tabular with 79 numerical features
   - XGBoost consistently outperforms other algorithms on cybersecurity datasets
   - Expected accuracy: 95%+ for binary classification (Benign vs Attack)

2. **Handles Class Imbalance Effectively**
   - CICIDS2017 has ~80% benign, 20% attack traffic
   - XGBoost's `scale_pos_weight` parameter balances classes
   - Alternative: SMOTE (Synthetic Minority Over-sampling) for extreme imbalance

3. **Native Feature Importance**
   - Built-in feature importance rankings
   - Fast training and inference (critical for real-time SOC operations)
   - Robust to outliers and missing values

4. **Ideal for SHAP Explainability**
   - TreeSHAP algorithm provides exact Shapley values for tree-based models
   - Computationally efficient (milliseconds per prediction)
   - Produces consistent, mathematically-grounded explanations

### Explainability Framework: SHAP (SHapley Additive exPlanations)

**Why SHAP for Security?**

1. **Mathematically Rigorous**
   - Based on cooperative game theory (Shapley values)
   - Unique, additive feature attribution
   - Consistent: features that contribute more get higher values

2. **Model-Agnostic with TreeSHAP Optimization**
   - TreeSHAP provides exact SHAP values for tree models in polynomial time
   - No approximation required (unlike LIME)
   - Scales to thousands of predictions

3. **Local and Global Interpretability**
   - **Local:** Explain individual alert predictions ("Why was THIS flow flagged?")
   - **Global:** Identify overall model behavior ("What features matter most for detecting DDoS?")

4. **Security-Relevant Explanations**
   - Maps to SOC analyst mental models (packet counts, flags, timing patterns)
   - Reveals attack signatures in feature space
   - Enables bias detection (over-reliance on weak indicators)

### Model Architecture

```
Input: Network Flow Features (79 dimensions)
    ↓
Preprocessing: StandardScaler normalization
    ↓
XGBoost Classifier (100 trees, max_depth=6, learning_rate=0.1)
    ↓
Prediction: [Benign | Attack] + Confidence Score
    ↓
SHAP TreeExplainer: Calculate feature contributions
    ↓
Case Report Generator: Template-based alert summary
    ↓
Output: Actionable Tier 1 Triage Report
```

---

## 3. Dataset of Interest and Justification

### Dataset: CICIDS2017 (Canadian Institute for Cybersecurity Intrusion Detection System)

**Source:** University of New Brunswick, Canadian Institute for Cybersecurity
**Public Access:** https://www.unb.ca/cic/datasets/ids-2017.html
**Citation:** Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization.* ICISSP.

### Dataset Characteristics

**Size and Scope:**
- **2,830,743 network flow records** (8 CSV files covering Monday-Friday)
- **79 features per flow:** Packet statistics, timing, protocol flags, flow characteristics
- **15 class labels:** 1 benign + 14 attack types

**Attack Types Included:**
- **DDoS** (Distributed Denial of Service)
- **Port Scan** (Network reconnaissance)
- **Brute Force** (FTP-Patator, SSH-Patator)
- **Web Attacks** (SQL Injection, XSS, Infiltration)
- **DoS** (Slowloris, Slowhttptest, Hulk, GoldenEye, Heartbleed)
- **Botnet ARES**

**Temporal Distribution:**
- **Monday:** Benign traffic baseline (529,918 flows)
- **Tuesday:** Benign + FTP/SSH brute force (445,909 flows)
- **Wednesday:** DoS/DDoS attacks (692,703 flows)
- **Thursday AM:** Web attacks (170,366 flows)
- **Thursday PM:** Infiltration attempts (288,602 flows)
- **Friday AM:** Benign traffic (191,033 flows)
- **Friday PM:** DDoS + Port Scans (512,213 flows)

**Class Distribution:**
- Benign: ~80% (2,273,097 flows)
- Attack: ~20% (557,646 flows)
- **Manageable imbalance** (no extreme skew)

### Feature Categories (79 Total)

**1. Flow Identification (5 features)**
- Source IP, Source Port, Destination IP, Destination Port, Protocol

**2. Basic Flow Stats (10 features)**
- Flow Duration, Total Fwd/Bwd Packets, Total Fwd/Bwd Packet Length
- Flow Bytes/s, Flow Packets/s

**3. Packet Length Statistics (12 features)**
- Min/Max/Mean/Std for Forward and Backward packet lengths

**4. Inter-Arrival Time (IAT) Stats (12 features)**
- Flow IAT Mean/Std/Max/Min
- Forward IAT Total/Mean/Std/Max/Min
- Backward IAT Total/Mean/Std/Max/Min

**5. Protocol Flags (9 features)**
- FIN, SYN, RST, PSH, ACK, URG, CWE, ECE Flag Counts
- Down/Up Ratio

**6. Bulk Transfer Stats (6 features)**
- Fwd/Bwd Avg Bytes/Packets/Bulk Rate

**7. Subflow Stats (4 features)**
- Subflow Fwd Packets/Bytes, Subflow Bwd Packets/Bytes

**8. Window and Segment Sizes (4 features)**
- Init_Win_bytes_forward, Init_Win_bytes_backward
- act_data_pkt_fwd, min_seg_size_forward

**9. Active/Idle Time Stats (8 features)**
- Active Mean/Std/Max/Min, Idle Mean/Std/Max/Min

**10. Packet Statistics (9 features)**
- Average Packet Size, Avg Fwd/Bwd Segment Size, Header Lengths, etc.

### Justification for Dataset Selection

**1. Industry-Standard Benchmark**
- CICIDS2017 is widely cited in IDS research (1,000+ citations)
- Enables comparison with state-of-the-art methods
- Realistic network traffic patterns (not synthetic)

**2. Comprehensive Attack Coverage**
- 14 distinct attack types cover major threat vectors
- Includes both classic (Port Scan) and modern (Botnet, Heartbleed) attacks
- Reflects real-world SOC alert diversity

**3. Rich Feature Set**
- 79 features provide comprehensive flow characterization
- Mix of statistical, temporal, and protocol-based features
- Supports deep SHAP analysis (many features to explain)

**4. Appropriate Scale**
- 2.8M records: Large enough for robust training, small enough for single-machine processing
- Stratified sampling possible if memory constraints arise
- Balanced enough to avoid extreme class imbalance

**5. Labeled Ground Truth**
- All flows labeled with attack type or benign
- Enables supervised learning and rigorous evaluation
- Supports multi-class classification (15-way) or binary (Benign vs Attack)

**6. Operational Relevance**
- Network flow data is exactly what SOC analysts see (NetFlow, IPFIX logs)
- Features map directly to SOC tools (Wireshark, Zeek, Suricata)
- Explanations will be interpretable to practitioners

### Data Limitations (Acknowledged)

1. **Simulated Environment:** CICIDS2017 was generated in a controlled lab, not real corporate network
   - *Mitigation:* Still reflects realistic attack signatures; widely accepted in research

2. **Age (2017):** Network threats evolve; some modern attacks missing (e.g., recent ransomware)
   - *Mitigation:* Core attack principles (DDoS, Port Scan) remain consistent; methodology generalizes

3. **Binary/Multi-class Only:** No severity scores or attack stage information
   - *Mitigation:* Sufficient for Tier 1 triage (identify attack type, escalate or dismiss)

**Despite these limitations, CICIDS2017 remains the gold standard for IDS research and is perfectly suited for demonstrating XAI-based automated triage.**

---

## 4. Contribution to the Cybersecurity Field

This project contributes to cybersecurity practice and research in **four key areas**:

### 1. Bridging the ML-Practitioner Gap in SOC Operations

**The Problem:**
- ML-based IDS achieves 95%+ accuracy in research papers
- Yet SOC adoption remains low due to "black box" distrust
- Analysts waste time manually investigating false positives
- High-performing models sit unused because analysts can't validate predictions

**Our Contribution:**
- **X-IDS Framework:** Combines high accuracy (XGBoost) with transparency (SHAP)
- **Proof of concept** that XAI can make ML IDS operationally viable
- **Case reports** that speak the SOC analyst's language (flags, packet patterns, timing)
- **Demonstrates feasibility** of automated Tier 1 triage with human oversight

**Impact:** Provides a blueprint for organizations to deploy trustworthy ML IDS systems

---

### 2. Explainability for Alert Fatigue Reduction

**The Problem:**
- SOC analysts receive 10,000+ alerts per day (Ponemon Institute, 2020)
- Tier 1 analysts spend ~5 minutes per alert on average
- 80%+ are false positives or benign traffic
- **Alert fatigue leads to missed real threats** (2023 IBM Security Report)

**Our Contribution:**
- **Automated triage** of 80%+ benign alerts with XAI justification
- **90%+ reduction in per-alert processing time** (5 min → 30 sec)
- **Prioritization:** High-confidence attacks auto-escalated, low-confidence flagged for review
- **Frees Tier 1 analysts** for proactive threat hunting instead of reactive alert processing

**Impact:** Measurable efficiency gains that can be deployed in real SOCs

---

### 3. XAI Validation in High-Stakes Security Domain

**The Problem:**
- Most XAI research focuses on healthcare, finance, or image classification
- Limited research on **SHAP for cybersecurity** specifically
- Unclear if SHAP explanations are actionable for security analysts
- Need domain-specific evaluation of XAI methods

**Our Contribution:**
- **First comprehensive evaluation** of SHAP for network IDS triage
- **Attack-specific SHAP patterns:** Document what features matter for each attack type
  - Port Scan: Sequential destination ports, high SYN flags
  - DDoS: Flow bytes/s, packet rate, protocol flags
  - Web Attack: Packet length anomalies, header sizes
- **Bias detection:** Use SHAP global importance to identify over-reliance on weak features
- **Case studies:** Show SHAP explanations match known attack signatures

**Impact:** Establishes SHAP as a viable XAI method for security operations

---

### 4. Open Framework for Reproducible XAI Research

**The Problem:**
- Many security ML papers don't release code
- XAI implementation details often omitted
- Difficult to reproduce or build upon prior work

**Our Contribution:**
- **Complete pipeline:** Data preprocessing → Model training → SHAP analysis → Case reports
- **Documented notebooks:** Step-by-step implementation with explanations
- **Evaluation framework:** Metrics for both accuracy AND explainability
- **Reusable templates:** Case report generation applicable to any IDS

**Impact:** Enables other researchers to extend this work to new datasets and attack types

---

### Summary of Contributions

| Area | Contribution | Cybersecurity Impact |
|------|--------------|---------------------|
| **SOC Operations** | Automated Tier 1 triage with XAI | Reduces analyst workload by 80%+ |
| **Alert Fatigue** | SHAP-based case reports | 90%+ reduction in per-alert time |
| **XAI Research** | SHAP validation for network IDS | Establishes best practices |
| **Reproducibility** | Open framework and code | Enables community extension |
| **Trust in ML** | Transparent attack detection | Increases SOC adoption of ML IDS |

---

## 5. Identify Potential Sources of Bias

Machine learning models in security are particularly vulnerable to bias, which can lead to:
- **False positives:** Flagging benign traffic as attacks (alert fatigue)
- **False negatives:** Missing real attacks (security breaches)
- **Discrimination:** Biasing against specific IPs, ports, or protocols
- **Automation bias:** Over-trusting model predictions without validation

We identify **five primary bias sources** and our mitigation strategies:

---

### Bias Source 1: Training Data Labeling Bias

**Risk:**
- CICIDS2017 labels were generated via network simulation, not manual expert labeling
- Potential mislabeling of edge cases (e.g., aggressive legitimate traffic flagged as attack)
- Simulator may not capture all benign behavior patterns (e.g., unusual but legitimate protocols)

**Impact:**
- Model learns incorrect patterns
- Benign traffic from certain applications/protocols falsely flagged
- Erodes analyst trust if too many false positives

**Mitigation Strategy 1: Cross-validation with Domain Knowledge**
- **Action:** Manually review sample predictions and compare SHAP explanations to known attack signatures
- **Validation:** For each attack type, verify top SHAP features align with:
  - NIST attack definitions
  - MITRE ATT&CK framework patterns
  - Wireshark/Snort rule signatures
- **Example:** Port Scan should show high SHAP values for "sequential destination ports" and "high SYN flags" — if not, investigate model logic
- **Threshold:** If SHAP explanations contradict domain expertise in >10% of cases, retrain with adjusted features

**Mitigation Strategy 2: Outlier Analysis**
- **Action:** Identify flows with high prediction confidence but unusual SHAP patterns
- **Tool:** Use SHAP force plots to visualize individual predictions
- **Manual Review:** Inspect 100 random high-confidence predictions per attack type
- **Correction:** If systematic mislabeling detected, relabel those samples and retrain

---

### Bias Source 2: Class Imbalance Bias

**Risk:**
- Benign traffic (80%) vastly outweighs attacks (20%)
- Model may optimize for majority class, leading to high overall accuracy but poor attack detection
- Rare attack types (e.g., Infiltration, Heartbleed) may be underrepresented

**Impact:**
- High false negative rate (missed attacks)
- Model defaults to "benign" prediction
- Rare but critical attacks go undetected

**Mitigation Strategy 1: Class Weighting**
- **Action:** Use XGBoost's `scale_pos_weight` parameter
  ```python
  scale_pos_weight = (# benign flows) / (# attack flows)
  # Example: 2,273,097 / 557,646 = 4.08
  ```
- **Effect:** Penalizes misclassifying attacks 4× more than benign flows
- **Validation:** Ensure recall for attack class >95% (minimize false negatives)

**Mitigation Strategy 2: Stratified Sampling**
- **Action:** Use stratified train/test splits to maintain class proportions
- **Code:**
  ```python
  train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
  ```
- **Validation:** Verify test set has same 80/20 benign/attack ratio as full dataset

**Mitigation Strategy 3: Per-Class Evaluation Metrics**
- **Action:** Report precision, recall, F1-score **per attack type** (not just overall accuracy)
- **Threshold:** Minimum 85% recall for each of the 14 attack types
- **Rare Class Handling:** If specific attack type has <1,000 samples, consider SMOTE upsampling

---

### Bias Source 3: Feature Bias (Over-reliance on Weak Indicators)

**Risk:**
- Model may latch onto spurious correlations
- **Example:** If all DDoS attacks in training data came from specific IP range, model may rely on "source IP" instead of attack behavior
- **Example:** Time-of-day bias (if attacks only simulated at night, model flags all night traffic)

**Impact:**
- Model fails to generalize to new environments
- Brittle detection (easily evaded by changing superficial features)
- False positives when deployed in different network

**Mitigation Strategy 1: SHAP Global Feature Importance Analysis**
- **Action:** Use SHAP summary plots to identify top features
  ```python
  shap.summary_plot(shap_values, X_test, plot_type="bar")
  ```
- **Red Flags:** If top features include:
  - Source/Destination IP addresses (should be network-agnostic)
  - Specific port numbers (attacks use various ports)
  - Time-based features (attacks occur anytime)
- **Threshold:** If any single feature has >30% importance, investigate for overfitting

**Mitigation Strategy 2: Feature Engineering Review**
- **Action:** Prioritize protocol-agnostic features:
  - ✅ Packet size distributions (attackers can't hide this)
  - ✅ Flow timing patterns (inherent to attack behavior)
  - ✅ Flag combinations (protocol-level signatures)
  - ❌ IP addresses (environment-specific)
  - ❌ Absolute port numbers (use port ranges instead)
- **Validation:** Remove IP/port features and retrain; if accuracy drops >5%, model was biased

**Mitigation Strategy 3: Adversarial Testing**
- **Action:** Manually create "adversarial" benign flows with attack-like features
  - Example: Legitimate bulk transfer with high flow bytes/s (looks like DDoS)
- **Test:** Does SHAP explanation correctly identify why model (correctly/incorrectly) flagged it?
- **Refinement:** Add adversarial examples to training if systematic errors found

---

### Bias Source 4: Automation Bias (Over-trust in XAI Explanations)

**Risk:**
- Analysts may blindly trust SHAP explanations without critical thinking
- **SHAP is descriptive, not prescriptive:** It explains what the model learned, not whether the model is correct
- Malicious actors could craft attacks that "look benign" according to SHAP features

**Impact:**
- False sense of security
- Missed attacks that evade SHAP-highlighted features
- Reduced human oversight

**Mitigation Strategy 1: Human-in-the-Loop Validation**
- **Action:** Implement three-tier triage system:
  - **Tier 1 (Automated):** High-confidence benign (SHAP score <0.2) → Auto-close
  - **Tier 2 (Analyst Review):** Medium confidence (SHAP score 0.2-0.8) → Flagged for human validation
  - **Tier 3 (Auto-escalate):** High-confidence attack (SHAP score >0.8) → Immediate escalation
- **Threshold Calibration:** Adjust thresholds based on false positive/negative rates in production

**Mitigation Strategy 2: SHAP Uncertainty Quantification**
- **Action:** Include confidence intervals in case reports
  - Not just: "Port Scan detected (97% confidence)"
  - But: "Port Scan detected (97% confidence, ±3% margin) based on sequential port access pattern"
- **Highlight Ambiguity:** If SHAP values are evenly split between features, flag as "uncertain — human review required"

**Mitigation Strategy 3: Red Team Testing**
- **Action:** Have security experts attempt to evade detection
- **Goal:** Find blind spots where SHAP explanations are plausible but wrong
- **Iteration:** Update model with evasion attempts to improve robustness

---

### Bias Source 5: Temporal Drift (Model Degradation Over Time)

**Risk:**
- Network traffic patterns evolve (new applications, protocols, attack techniques)
- Model trained on 2017 data may not generalize to 2025 threats
- SHAP explanations become outdated

**Impact:**
- Increasing false negatives as new attacks emerge
- False positives as legitimate traffic patterns change

**Mitigation Strategy 1: Model Retraining Schedule**
- **Action:** Establish quarterly retraining on recent traffic data
- **Monitoring:** Track prediction confidence over time; declining confidence signals drift
- **Trigger:** If accuracy drops >5% from baseline, retrain immediately

**Mitigation Strategy 2: SHAP Pattern Monitoring**
- **Action:** Compare SHAP feature importance over time
- **Detection:** If top features shift significantly (e.g., PSH Flag drops from #1 to #5), investigate
- **Alert:** "Model may be drifting — SHAP patterns changing"

---

### Summary: Bias Mitigation Framework

| Bias Source | Mitigation Strategy | Validation Metric |
|-------------|---------------------|-------------------|
| **Labeling Bias** | Cross-validate SHAP with domain expertise | <10% explanation-signature mismatch |
| **Class Imbalance** | scale_pos_weight + stratified sampling | >95% recall for attack class |
| **Feature Bias** | SHAP global importance + adversarial testing | No single feature >30% importance |
| **Automation Bias** | Human-in-the-loop for medium confidence | 0% blind auto-action on uncertain alerts |
| **Temporal Drift** | Quarterly retraining + SHAP monitoring | <5% accuracy degradation |

**By implementing these five mitigation strategies, we ensure our X-IDS system is robust, fair, and maintains high performance while earning analyst trust.**

---

## 6. Expected Outcomes and Success Metrics

### Primary Outcomes

**1. High-Performance Intrusion Detection Model**
- **Target:** 95%+ accuracy (binary classification: Benign vs Attack)
- **Target:** 90%+ accuracy (multi-class: 15 attack types)
- **Target:** <5% false positive rate (critical for alert fatigue reduction)
- **Target:** >95% recall for attack class (minimize missed threats)

**2. Transparent SHAP-Based Explanations**
- **Target:** Generate SHAP explanations for 100% of predictions
- **Target:** <2 seconds inference + explanation time per alert (real-time viable)
- **Target:** Top-5 SHAP features align with known attack signatures (validated against NIST/MITRE)

**3. Automated Tier 1 Triage System**
- **Target:** 80%+ of benign traffic auto-closed with XAI justification
- **Target:** 90%+ reduction in per-alert triage time (5 min → 30 sec)
- **Target:** Template-based case reports for all alerts

**4. Reproducible Research Framework**
- **Target:** Complete Jupyter notebooks with documentation
- **Target:** Reusable code for other IDS datasets
- **Target:** Open-source release (GitHub) for community use

### Evaluation Metrics

**Model Performance Metrics:**
- Accuracy, Precision, Recall, F1-Score (overall and per-class)
- ROC-AUC (Receiver Operating Characteristic - Area Under Curve)
- Confusion Matrix (visualize false positives/negatives)
- Per-attack-type detection rates

**Explainability Metrics:**
- **Consistency:** SHAP feature rankings stable across random seeds
- **Fidelity:** SHAP explanations match model behavior (>95% correlation)
- **Alignment:** Top SHAP features match domain expertise (manual validation)
- **Computational Efficiency:** Inference + SHAP time <2 seconds per alert

**Operational Metrics (Simulated):**
- **Triage Time Reduction:** Baseline (5 min manual) vs Automated (30 sec with XAI)
- **Automation Rate:** % of alerts handled without human intervention
- **False Positive Reduction:** % decrease in benign alerts requiring investigation

### Success Criteria

**Minimum Viable Product (MVP):**
- ✅ XGBoost model trained with 90%+ accuracy
- ✅ SHAP explanations generated for test set
- ✅ Case report template implemented
- ✅ Evaluation metrics calculated

**Full Project Success:**
- ✅ 95%+ accuracy with <5% false positive rate
- ✅ SHAP patterns validated against attack signatures
- ✅ Demonstrated 90%+ triage time reduction
- ✅ Bias mitigation strategies implemented and tested
- ✅ Complete documentation and reproducible code
- ✅ Final report with recommendations for SOC deployment

---

## 7. Project Timeline and Milestones

**Total Duration:** 4 weeks (aggressive timeline, focused scope)

### Week 1: Data Preparation and EDA
- **Days 1-2:** Load CICIDS2017 data, handle infinity/NaN values, encode labels
- **Days 3-4:** Exploratory data analysis (class distribution, feature correlations, visualization)
- **Days 5-7:** Feature selection (reduce from 79 to top 20-30 features), save processed data
- **Deliverable:** Clean dataset ready for modeling, EDA report

### Week 2: Model Training and Evaluation
- **Days 1-2:** Train XGBoost binary classifier (Benign vs Attack), tune hyperparameters
- **Days 3-4:** Train XGBoost multi-class classifier (15 attack types), evaluate per-class metrics
- **Days 5-7:** Handle class imbalance (scale_pos_weight), final model selection and saving
- **Deliverable:** Trained models with 95%+ accuracy, evaluation metrics report

### Week 3: SHAP Explainability Implementation
- **Days 1-2:** Implement TreeSHAP, generate SHAP values for test set
- **Days 3-4:** Create SHAP visualizations (summary plots, force plots, dependence plots)
- **Days 5-7:** Validate SHAP patterns against attack signatures, bias detection analysis
- **Deliverable:** SHAP explanations for all predictions, validation report

### Week 4: Automation, Bias Mitigation, and Documentation
- **Days 1-2:** Build case report generator, simulate triage time reduction
- **Days 3-4:** Implement and test bias mitigation strategies
- **Days 5-7:** Final documentation, presentation preparation, code cleanup
- **Deliverable:** Complete X-IDS system, final report, presentation slides

---

## 8. Tools and Technologies

**Programming Language:** Python 3.8+

**Core Libraries:**
- **Data Processing:** pandas, numpy, scikit-learn
- **Machine Learning:** xgboost
- **Explainability:** shap
- **Visualization:** matplotlib, seaborn, plotly
- **Utilities:** tqdm (progress bars), joblib (model saving)

**Development Environment:**
- **Preferred:** Google Colab Pro (51 GB RAM, free GPU)
- **Alternative:** Local Jupyter Notebook (16 GB+ RAM recommended)

**Version Control:** Git/GitHub (for reproducibility)

---

## 9. Limitations and Future Work

### Acknowledged Limitations

**1. Simulated Data (Not Real SOC Traffic)**
- CICIDS2017 generated in lab environment, not live corporate network
- May not capture all real-world traffic nuances
- *Future Work:* Validate on real SOC data with privacy protections

**2. Dataset Age (2017)**
- Modern attacks (recent ransomware, API exploits) not represented
- Network protocols evolved since 2017
- *Future Work:* Retrain on CICIDS2019 or newer datasets

**3. Binary/Multi-class Classification Only**
- No attack severity scores or kill chain stage detection
- *Future Work:* Extend to multi-label classification (attack + stage)

**4. No Real Analyst User Study**
- Triage time reduction is simulated, not measured with real analysts
- Analyst trust claims are theoretical
- *Future Work:* Deploy in SOC, conduct user study with Likert scales

**5. Single Algorithm (XGBoost)**
- Did not compare to other explainable models (e.g., decision trees, rule-based systems)
- *Future Work:* Comparative study with Random Forest, neural networks + SHAP

### Future Research Directions

1. **Multi-Dataset Validation:** Test on UNSW-NB15, NSL-KDD, CIC-DDoS2019
2. **Real-Time Deployment:** Integrate with Zeek/Suricata for live network monitoring
3. **Adversarial Robustness:** Test against evasion attacks, improve defenses
4. **Alternative XAI Methods:** Compare SHAP vs LIME, Integrated Gradients, Attention mechanisms
5. **SOC Pilot Study:** Partner with organization to measure real MTTR reduction

---

## 10. Ethical Considerations

**1. Responsible Disclosure**
- Will not publish specific evasion techniques that could aid attackers
- Focus on defense improvements, not offensive capabilities

**2. Privacy**
- CICIDS2017 is anonymized; no real user/organization data
- If deployed on real traffic, must comply with data protection regulations (GDPR, CCPA)

**3. False Positive Harm**
- Over-blocking legitimate traffic can disrupt business operations
- Mitigation: Human-in-the-loop for medium-confidence alerts

**4. Automation Bias Awareness**
- Clearly document that XAI is a tool for analysts, not a replacement
- Emphasize need for human oversight and critical thinking

---

## 11. Citations

1. **Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018).** *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization.* 4th International Conference on Information Systems Security and Privacy (ICISSP). https://www.unb.ca/cic/datasets/ids-2017.html

2. **Lundberg, S. M., & Lee, S. I. (2017).** *A Unified Approach to Interpreting Model Predictions.* Advances in Neural Information Processing Systems (NeurIPS). https://arxiv.org/abs/1705.07874

3. **Chen, T., & Guestrin, C. (2016).** *XGBoost: A Scalable Tree Boosting System.* Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. https://arxiv.org/abs/1603.02754

4. **IBM Security. (2023).** *Cost of a Data Breach Report 2023.* IBM Corporation. https://www.ibm.com/security/data-breach

5. **Ponemon Institute. (2020).** *The Cost of Malware Containment.* Ponemon Institute Research Report.

6. **NIST. (2024).** *Cybersecurity Framework.* National Institute of Standards and Technology. https://www.nist.gov/cyberframework

7. **MITRE. (2024).** *ATT&CK Framework - Network Intrusion Detection.* MITRE Corporation. https://attack.mitre.org/

8. **Molnar, C. (2024).** *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable.* https://christophm.github.io/interpretable-ml-book/

---

## 12. Team Roles and Responsibilities

**Team Member 1 (Data Engineer):**
- Data loading, preprocessing, cleaning
- Feature engineering and selection
- Dataset documentation

**Team Member 2 (ML Engineer):**
- XGBoost model training and tuning
- Model evaluation and performance optimization
- Bias mitigation implementation

**Team Member 3 (XAI Specialist):**
- SHAP implementation and visualization
- Case report generation
- Explainability validation and analysis

**All Team Members:**
- Weekly progress meetings
- Collaborative code review
- Final documentation and presentation

---

## 13. Conclusion

This project bridges the critical gap between high-performing machine learning and operational cybersecurity by making network intrusion detection **explainable, trustworthy, and actionable**. By combining XGBoost's state-of-the-art accuracy with SHAP's transparent explanations, we demonstrate that AI-driven SOC automation is not only technically feasible but operationally viable.

Our X-IDS framework directly addresses alert fatigue—a pervasive problem costing organizations millions in analyst time and missed threats—while maintaining the human oversight necessary for responsible AI deployment in high-stakes security environments.

**The result:** A production-ready blueprint for next-generation intrusion detection systems that security teams can trust, understand, and deploy with confidence.

---

**Project Start Date:** [Insert Date]
**Expected Completion:** [Insert Date + 4 weeks]
**Repository:** [GitHub URL - To Be Created]

---

**Team Group 3 - AI4ALL XAI Project**
*Lopez, Itzalen | Frankyan, Shahane | Shanbhag, Nethra*
