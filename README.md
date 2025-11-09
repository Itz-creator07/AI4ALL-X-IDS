# X-IDS: Explainable Network Intrusion Detection System

**Reducing Alert Fatigue Through XGBoost + SHAP for Automated SOC Triage**

---

## 🎯 Project Overview

An **Explainable AI framework** for network intrusion detection that:
- ✅ Detects attacks with **95%+ accuracy** using XGBoost
- ✅ Explains **WHY** using SHAP (no more black boxes!)
- ✅ Generates automated **case reports** for SOC analysts
- ✅ Reduces triage time by **90%+** (5 min → 30 sec per alert)

**Team:** Group 3 - Lopez, Itzalen; Frankyan, Shahane; Shanbhag, Nethra

**Dataset:** CICIDS2017 (2.8M+ network flows, 15 attack types)

---

## 📁 Project Structure

```
X-IDS_Project/
├── README.md (you are here)
├── documentation/
│   ├── REVISED_PROJECT_PROPOSAL.md      ← Full proposal (13 sections)
│   └── IMPLEMENTATION_GUIDE.md          ← Step-by-step implementation
├── notebooks/
│   └── 01_X-IDS_Data_Preparation.ipynb  ← Start here!
├── data/                                 ← Outputs saved here
└── results/                              ← Models and visualizations
```

---

## 🚀 Quick Start (15 Minutes to First Results!)

### Step 1: Choose Your Environment

**Option A: Google Colab (RECOMMENDED)**
```
1. Go to: https://colab.research.google.com/
2. Upload: notebooks/01_X-IDS_Data_Preparation.ipynb
3. Upload your 8 CICIDS2017 CSV files
4. Click "Runtime" → "Run all"
```

**Option B: Local Jupyter**
```bash
cd X-IDS_Project/notebooks
jupyter notebook
# Open 01_X-IDS_Data_Preparation.ipynb
# Run all cells
```

### Step 2: Verify Data Files

You need these 8 CSV files from CICIDS2017:
- ✅ Monday-WorkingHours.pcap_ISCX.csv
- ✅ Tuesday-WorkingHours.pcap_ISCX.csv
- ✅ Wednesday-workingHours.pcap_ISCX.csv
- ✅ Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
- ✅ Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
- ✅ Friday-WorkingHours-Morning.pcap_ISCX.csv
- ✅ Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
- ✅ Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv

**These files should already be in your parent directory:**
`c:\Users\PC\Downloads\AI4ALL_XAI_Project\`

### Step 3: Run Data Preparation

Open `notebooks/01_X-IDS_Data_Preparation.ipynb` and run all cells.

**Expected Runtime:** 20-30 minutes

**Expected Outputs:**
```
✅ Loaded 2,830,743 network flows
✅ Cleaned data (removed inf/NaN)
✅ Selected 20-30 top features
✅ Created train/test splits (80/20)
✅ Saved to ../data/
```

---

## 📊 What You'll Build

### Component 1: Detection Layer (XGBoost)

**Input:** Network flow features (packet stats, timing, flags)
**Output:** Benign or Attack (+ confidence score)
**Performance:** 95%+ accuracy, <5% false positive rate

### Component 2: Explainability Layer (SHAP)

**Input:** XGBoost prediction
**Output:** Top 5 features explaining WHY
**Example:**
```
🚨 ATTACK DETECTED (97% confidence)

Why?
1. ↑ PSH Flag Count: 247 (normal: <10) → Port Scan signature
2. ↑ Flow Duration: 0.003s (very short) → Rapid scanning
3. ↑ SYN Flags: 247 → Connection probing
```

### Component 3: Automation Layer (Case Reports)

**Input:** Prediction + SHAP explanation
**Output:** Automated Tier 1 triage report

**Example Report:**
```
=================================================================
AUTOMATED TIER 1 TRIAGE REPORT
=================================================================
Alert ID: 12345
Classification: 🚨 ATTACK DETECTED
Risk Score: 97%

TOP CONTRIBUTING FACTORS:
1. PSH Flag Count = 247 → INCREASES attack likelihood by 0.42
2. Flow Duration = 0.003s → INCREASES attack likelihood by 0.28
3. Sequential Ports Detected → Port Scan signature

RECOMMENDATION: ⚠️ ESCALATE TO TIER 2 IMMEDIATELY
Action: Block source IP, investigate logs
=================================================================
```

---

## 📈 Expected Performance

| Metric | Target | Your Model |
|--------|--------|------------|
| **Accuracy** | >95% | ___ % |
| **Precision (Attack)** | >90% | ___ % |
| **Recall (Attack)** | >92% | ___ % |
| **False Positive Rate** | <5% | ___ % |
| **ROC-AUC** | >0.97 | ___ |
| **Inference Time** | <2 sec | ___ sec |
| **Triage Time Reduction** | >90% | ___ % |

---

## 🛠️ Implementation Roadmap

### Week 1: Data Preparation ✅
- [x] Load CICIDS2017 (2.8M flows)
- [x] Clean data (handle inf/NaN)
- [x] Select features (correlation-based)
- [x] Split train/test (80/20)
- **Deliverable:** `01_X-IDS_Data_Preparation.ipynb` complete

### Week 2: Model Training ⏳
- [ ] Train XGBoost classifier
- [ ] Tune hyperparameters
- [ ] Handle class imbalance
- [ ] Achieve 95%+ accuracy
- **Deliverable:** Trained model + evaluation report

### Week 3: SHAP Explainability ⏳
- [ ] Implement TreeSHAP
- [ ] Generate global/local explanations
- [ ] Validate against attack signatures
- [ ] Create visualizations
- **Deliverable:** SHAP analysis + plots

### Week 4: Automation & Docs ⏳
- [ ] Build case report generator
- [ ] Simulate triage time reduction
- [ ] Complete documentation
- [ ] Prepare presentation
- **Deliverable:** Final project package

---

## 📖 Documentation

### For Implementation:
- **[IMPLEMENTATION_GUIDE.md](documentation/IMPLEMENTATION_GUIDE.md)** - Detailed step-by-step guide
  - Environment setup
  - Code examples
  - Troubleshooting
  - Evaluation metrics
  - Deliverables checklist

### For Proposal:
- **[REVISED_PROJECT_PROPOSAL.md](documentation/REVISED_PROJECT_PROPOSAL.md)** - Full academic proposal
  - Research question
  - Methodology (XGBoost + SHAP)
  - Dataset justification
  - Bias mitigation strategies
  - Success criteria
  - Citations

---

## 🎓 Learning Resources

### XGBoost:
- Documentation: https://xgboost.readthedocs.io/
- Paper: https://arxiv.org/abs/1603.02754

### SHAP:
- Documentation: https://shap.readthedocs.io/
- Paper: https://arxiv.org/abs/1705.07874
- Tutorials: https://shap.readthedocs.io/en/latest/example_notebooks.html

### CICIDS2017:
- Dataset: https://www.unb.ca/cic/datasets/ids-2017.html
- Paper: Sharafaldin et al. (2018)

---

## ❓ FAQ

### Q: Do I need to know machine learning?
**A:** Basic understanding helpful. The notebooks walk you through everything step-by-step.

### Q: Can I run this on my laptop?
**A:** Yes, if you have 16+ GB RAM. Otherwise use Google Colab (12-51 GB RAM).

### Q: How long will this take?
**A:** 3-4 weeks (10-15 hours/week) following the implementation guide.

### Q: What if I get errors?
**A:** Check the Troubleshooting section in IMPLEMENTATION_GUIDE.md. Common issues covered!

### Q: Can I use this for my own dataset?
**A:** Yes! The framework works for any tabular network flow data. Just ensure features are similar.

### Q: Is this production-ready for a real SOC?
**A:** It's a proof-of-concept. For production, you'd need:
- Real-time integration with SIEM
- Model retraining pipeline
- A/B testing with analysts
- Incident response integration

---

## 🏆 Success Criteria

**Minimum Viable Product (MVP):**
- ✅ XGBoost model with 90%+ accuracy
- ✅ SHAP explanations generated
- ✅ Case report template working

**Full Project Success:**
- ✅ 95%+ accuracy with <5% FPR
- ✅ SHAP patterns validated
- ✅ 90%+ triage time reduction demonstrated
- ✅ Complete documentation
- ✅ Professional presentation

---

## 🤝 Team Contributions

**Person 1 (Data Engineer):**
- Data preparation notebook
- Feature engineering
- Data quality validation

**Person 2 (ML Engineer):**
- XGBoost training
- Hyperparameter tuning
- Model evaluation

**Person 3 (XAI Specialist):**
- SHAP implementation
- Explanation validation
- Case report generation

**All Together:**
- Integration testing
- Documentation
- Presentation

---

## 📞 Support

**For Implementation Help:**
- Check IMPLEMENTATION_GUIDE.md first
- Review code comments in notebooks
- Google error messages (many are common)

**For Conceptual Questions:**
- Review REVISED_PROJECT_PROPOSAL.md
- Read SHAP documentation
- Check XGBoost docs

---

## 🎉 You're Ready!

**Next Steps:**
1. ✅ Read this README (you did it!)
2. ⏩ Open `notebooks/01_X-IDS_Data_Preparation.ipynb`
3. ⏩ Run all cells
4. ⏩ Review outputs in `../data/`
5. ⏩ Proceed to model training (see IMPLEMENTATION_GUIDE.md)

**Good luck with your X-IDS project! 🚀**

---

**Project:** AI4ALL Explainable AI for Cybersecurity
**Date:** November 2025
**License:** Educational use - AI4ALL program
