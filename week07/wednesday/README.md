## git link : 
https://github.com/yash-gupta-7/weekly-assignment-iitgn/tree/main/week07/wednesday

# Week 07 Wednesday: NLP Logic · Hard Patterns & Aspect Sentiment

## Project Title
**ShopSense: Handling Hard NLP Patterns & Aspect-Based Sentiment Analysis (ABSA)**

## Problem Statement
Traditional sentiment analysis often fails when confronted with linguistic nuances like sarcasm, code-mixing (Hinglish), or negation logic (e.g., "not bad at all"). Furthermore, doc-level sentiment provides low-resolution feedback for multi-featured products. This assignment focuses on identifying these high-difficulty patterns and implementing a baseline for aspect-level sentiment extraction.

## Approach
1. **Data Generation:** Built a synthetic corpus containing targeted examples of negation, sarcasm, code-mixing (Hinglish), implicit negativity, and comparative sentiment.
2. **Hard Pattern Detection:** Implemented `sentiment_patterns.py` to programmatically identify these linguistic constructs and explain why traditional Bag-of-Words (BOW) systems fail them.
3. **Aspect Extraction:** Developed `aspect_extractor.py` to isolate specific product traits (Camera, Battery, Support) and map independent polarities to each.
4. **Jupyter Orchestration:** Unified the results into a notebook demonstrating how a single review can simultaneously hold Positive and Negative labels across different aspects.

## Technologies Used
- **Python 3.9**
- **Pandas / Numpy**
- **Regex** (Pattern detection)
- **Jupyter Notebook** (`nbconvert`)

## Results Summary
- **Pattern Robustness:** Demonstrated that "not bad at all" is correctly identified as a flipped positive sentiment despite containing the token "bad".
- **ABSA Resolution:** Successfully extracted conflicting sentiments (Amazing Camera | Atrocious Battery) from a single sentence, proving that review-level F1 is easier but less informative than aspect-level F1.
- **Strategic Roadmap:** Outlined 4 key engineering strategies (Dependency Parsing, Transfer Learning, etc.) to bridge the gap from 71% to 80%+ F1 in aspect classification.

## How to Run
1. Navigate to directory:
```bash
cd week07/wednesday/
```
2. Setup environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy notebook nbconvert nbformat
```
3. Generate dataset:
```bash
python3 src/data_generator.py
```
4. Build and execute the notebook:
```bash
python3 src/notebook_builder.py
jupyter nbconvert --to notebook --execute --inplace notebooks/week07_wednesday_assignment.ipynb
```

---
## 🔗 Git Resource
- **Project Repository**: [GitHub Link](https://github.com/yash-gupta-7/weekly-assignment-iitgn)

