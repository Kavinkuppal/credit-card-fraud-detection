# Credit Card Fraud Detection

A supervised and unsupervised machine learning pipeline that detects fraudulent credit card transactions across 1.8M+ records, engineering 17 behavioral features to catch fraud that raw transaction fields miss entirely.

## Table of Contents

* [Introduction](#introduction)
* [The Core Problem: Extreme Class Imbalance](#the-core-problem-extreme-class-imbalance)
* [Installation](#installation)
* [Usage](#usage)
* [Dataset](#dataset)
* [Feature Engineering](#feature-engineering)
* [Models](#models)
* [Results](#results)
* [Feature Importance](#feature-importance)
* [Unsupervised Extension](#unsupervised-extension)
* [Technical Details](#technical-details)
* [Repository Structure](#repository-structure)
* [Limitations](#limitations)
* [What I Learned](#what-i-learned)
* [Contributing](#contributing)
* [License](#license)
* [Contact Information](#contact-information)
* [Acknowledgments](#acknowledgments)

## Introduction

Payment card fraud cost $33.83 billion globally in 2023. The detection problem is unusual among classification tasks because of how lopsided it is: in this dataset, fraud represents **0.386%** of transactions. A model that predicts "not fraud" every single time achieves 99.6% accuracy and is completely worthless.

That single fact drives every design decision in this project. Accuracy is discarded as a metric. The optimization target is **recall on the fraud class** — the fraction of actual fraud the model catches — with the false positive rate held as a secondary constraint. The asymmetry is economic: a missed fraud is a direct financial loss and a customer whose account was drained, while a false positive is a declined card and an inconvenienced customer. These are not equally bad, and the model shouldn't treat them as if they were.

The second design decision is that **raw transaction fields are nearly useless on their own.** Knowing a transaction was $340 at a gas station tells you nothing. Knowing it was $340 when this cardholder's median transaction is $23, at 3 AM when they've never transacted after 10 PM, at a merchant they've never used, 400 km from their last transaction 20 minutes ago — that's a fraud signal. Every one of those is a *derived* feature. The bulk of the work in this project is constructing them.

**[Full project report with visualizations →](https://kavinkuppal.github.io/credit-card-fraud-detection/)**

## The Core Problem: Extreme Class Imbalance

With 2,145 fraud cases against 553,574 legitimate ones in the test set, three things follow:

1. **Accuracy is meaningless.** Every model here scores 94–99% accuracy. None of that number is informative.
2. **Training needs rebalancing.** SMOTE (Synthetic Minority Oversampling Technique) is applied to the training set — but *after* the train/test split, never before, so no synthetic minority samples leak into evaluation. The test set keeps its natural 0.386% fraud rate, because that's the distribution the model would face in production.
3. **Precision and recall trade off sharply, and the trade is worth making.** Pushing the decision boundary toward catching more fraud necessarily flags more legitimate transactions. At this level of imbalance the arithmetic is brutal: even a model with an excellent 2% false positive rate generates ~11,000 false alarms against ~2,000 true catches, because there are 258 times more legitimate transactions to be wrong about. This is what precision on an imbalanced class looks like, and it is the expected shape of the result rather than a defect. In deployment, flagged transactions go to a review queue or a step-up authentication prompt, not an automatic decline — which is precisely why recall is the metric worth optimizing.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Kavinkuppal/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

2. Set up a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate    # On Windows use: venv\Scripts\activate
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection) and place `fraudTrain.csv` and `fraudTest.csv` in the project root. They are not committed here — together they exceed GitHub's file size limits.

## Usage

Launch the notebook:

```bash
jupyter notebook final.ipynb
```

Run the cells in order. The preprocessing step is the expensive one — it engineers 17 features across 1.8M rows, including rolling time-window aggregations that take several minutes. It caches its output:

```python
# First run: builds features from scratch and writes trainX/trainY/testX/testY CSVs
trainX, trainY, testX, testY = preprocessData(useExisting=False)

# Subsequent runs: loads the cached CSVs in seconds
trainX, trainY, testX, testY = preprocessData()
```

After preprocessing, each model trains independently — you can run just the XGBoost cells without training the others.

## Dataset

The data comes from the [Sparkov Data Generation](https://github.com/namebrandon/Sparkov_Data_Generation) simulator, which produces two years of realistic synthetic credit card transactions (2019–2020).

| | Records | Fraud cases | Fraud rate |
|---|---|---|---|
| Training set | 1,296,675 | 7,506 | 0.579% |
| Test set | 555,719 | 2,145 | 0.386% |
| **Total** | **1,852,394** | **9,651** | **0.521%** |

Each record contains cardholder information (name, address, date of birth, gender), transaction details (timestamp, card number, amount, transaction ID), merchant details (name, category, latitude, longitude), and the binary `is_fraud` target.

## Feature Engineering

17 features across five families. This is where the project's actual leverage is.

**Temporal** — captures transactions at unusual times.

| Feature | Description |
|---|---|
| `day_of_week` | Day index extracted from the transaction timestamp |
| `hour_of_day` | Hour of transaction |
| `is_weekend` | Binary flag for Saturday/Sunday |
| `is_nighttime` | Binary flag for transactions between 10 PM and 6 AM |

**Amount** — the key insight is that amounts are only meaningful *relative to the individual cardholder*. A $500 charge is unremarkable for one card and a five-sigma outlier for another. Both features are computed per card via `groupby("cc_num").transform()`.

| Feature | Description |
|---|---|
| `amt_to_median_ratio` | Transaction amount ÷ that cardholder's median transaction amount |
| `amt_zscore` | Standard deviations from that cardholder's mean, with division-by-zero guarded |

**Merchant** — captures relationship novelty. Fraud disproportionately occurs at merchants the cardholder has never used.

| Feature | Description |
|---|---|
| `new_merchant_flag` | First time this card has transacted with this merchant |
| `merchant_fraud_rate` | Historical fraud rate at this merchant, computed causally |
| `category_frequency` | How often this cardholder purchases in this merchant category |

`merchant_fraud_rate` required care to avoid target leakage — it uses only transactions *preceding* the current one in time, via a sort-then-`cumcount` pattern, so a transaction never contributes to its own feature value.

**Geospatial** — catches physically impossible card usage.

| Feature | Description |
|---|---|
| `transDistance` | Geodesic km between cardholder's home and merchant (`geopy.distance`) |
| `merchant_travel_distance_24hr` | Total distance traveled between consecutive merchants in a 24-hour window |
| `impossible_travel_flag` | Consecutive transactions implying travel faster than 300 km/h |

**Velocity** — fraud arrives in bursts. Implemented with pandas time-based rolling windows (`rolling('1h', on=ts_col)`) grouped per card, with the current transaction subtracted out so a feature never counts itself.

| Feature | Description |
|---|---|
| `transaction_count_1h` | Transactions on this card in the preceding hour |
| `transaction_count_24h` | Transactions on this card in the preceding 24 hours |
| `total_amount_1h` | Total spend in the preceding hour |
| `total_amount_24h` | Total spend in the preceding 24 hours |
| `time_since_last_transaction` | Seconds elapsed since the previous transaction on this card |

Numeric features are scaled with `StandardScaler`; `day_of_week` and `hour_of_day` are one-hot encoded, since hour 23 and hour 0 are adjacent in reality but maximally distant as integers.

## Models

Five approaches were trained and compared:

* **Random Forest** — baseline ensemble. 100 trees, max depth 20. Handles structured data well with little tuning.
* **SGD Classifier** — fast linear baseline with log loss and L2 penalty. Included as a reference point to measure how much nonlinearity actually buys.
* **Multi-Layer Perceptron** — single hidden layer of 100 units, to capture feature interactions the linear model can't represent.
* **XGBoost** — gradient-boosted trees with `binary:logistic` objective, `aucpr` eval metric (area under the precision-recall curve, the correct choice for imbalanced data over ROC-AUC), and `hist` tree method for speed. Tuned via `GridSearchCV` over **16 hyperparameter combinations across 3-fold cross-validation — 48 total fits** — searching learning rate, max depth, subsample, and colsample_bytree.
* **Soft-voting ensemble** — averaged predicted probabilities across all four models above.

Best XGBoost parameters found: `learning_rate=0.05`, `max_depth=8`, `subsample=0.8`, `colsample_bytree=0.7`.

## Results

All metrics are for the **fraud class** on the held-out test set (555,719 transactions, 2,145 fraud cases). Ranked by the metric that matters — false negative rate.

| Model | Recall | Precision | F1 | False Negative Rate | False Positive Rate |
|---|---|---|---|---|---|
| SGD Classifier | 0.86 | 0.05 | 0.10 | 13.5% | 6.32% |
| Random Forest | 0.85 | 0.22 | 0.35 | 15.2% | 1.16% |
| MLP | 0.89 | 0.09 | 0.17 | 10.7% | 4.63% |
| Soft-voting ensemble | 0.89 | 0.13 | 0.23 | 11.0% | 2.32% |
| **XGBoost** | **0.91** | 0.13 | 0.23 | **9.1%** | 2.32% |
| XGBoost + KMeans clusters | 0.90 | 0.15 | 0.25 | 10.1% | 2.03% |

**XGBoost is the strongest model**, catching **91% of all fraudulent transactions** while flagging only 2.3% of legitimate ones.

Its confusion matrix on the test set:

| | Predicted Legitimate | Predicted Fraud |
|---|---|---|
| **Actually Legitimate** | 540,727 | 12,847 |
| **Actually Fraud** | 195 | 1,950 |

Reading that concretely: of 2,145 real fraud cases, the model caught 1,950 and missed 195. Of 553,574 legitimate transactions, it correctly cleared 540,727 and flagged 12,847 for review.

Two things worth noting about the shape of these numbers. First, **Random Forest's higher precision (0.22) is not a better result** — it buys that precision by missing 40% more fraud than XGBoost, which is the expensive kind of error. Second, precision in the 0.13–0.22 range across every model is a function of the base rate, not model quality: when only 1 in 259 transactions is fraudulent, even a 2.3% false positive rate produces roughly 6.6 false alarms per genuine catch. That is the arithmetic of imbalanced detection, and it's why systems like this feed a review queue rather than an auto-decline. The operationally meaningful claim is the pair **91% of fraud caught, 97.7% of legitimate traffic untouched.**

## Feature Importance

Extracted from the trained XGBoost model:

| Feature | Importance |
|---|---|
| `amt_zscore` | 0.295 |
| `total_amount_24h` | 0.237 |
| `amt_to_median_ratio` | 0.131 |
| `is_nighttime` | 0.085 |
| `hour_of_day` | 0.049 |
| `transaction_count_24h` | 0.042 |
| `total_amount_1h` | 0.039 |
| `is_weekend` | 0.033 |
| `time_since_last_transaction` | 0.021 |
| `day_of_week` | 0.019 |
| `category_frequency` | 0.015 |
| `new_merchant_flag` | 0.014 |
| `transaction_count_1h` | 0.007 |
| `impossible_travel_flag` | 0.006 |
| `merchant_travel_distance_24hr` | 0.004 |
| `transDistance` | 0.004 |

The three amount-relative features account for **66% of total importance**. Deviation from a cardholder's own spending baseline is overwhelmingly the strongest fraud signal — which validates the decision to normalize per card rather than using raw amounts.

The geospatial features, by contrast, contributed almost nothing (< 1.5% combined) despite being the most intellectually satisfying to build. `impossible_travel_flag` in particular felt like it should be a smoking gun and turned out to fire too rarely to matter. Useful lesson about the gap between a compelling hypothesis and a useful feature.

## Unsupervised Extension

K-Means clustering was layered on top of the supervised pipeline to test whether unlabeled behavioral structure adds signal.

Testing k from 1 to 20 and plotting inertia produced an elbow at **k = 4**. Each transaction was assigned a cluster label, the labels were one-hot encoded into `cluster_0` through `cluster_3`, and those four columns were appended to the feature set before retraining XGBoost.

The clusters function as behavioral profiles — roughly, frequent small purchases, infrequent large purchases, geographically dispersed activity, and routine local activity. The intent is that a transaction can be evaluated against its behavioral group rather than the population.

**Result:** recall 0.90, precision 0.15, false positive rate 2.03% — slightly better precision and fewer false positives than plain XGBoost, at the cost of 22 more missed fraud cases. Given that missed fraud is the expensive error, plain XGBoost remains the better model for this objective. The honest read is that clustering added marginal value here, likely because the engineered per-cardholder features already capture most of the behavioral normalization the clusters were meant to provide.

## Technical Details

* **Language:** Python 3
* **ML:** scikit-learn (RandomForest, SGDClassifier, MLPClassifier, KMeans, VotingClassifier, GridSearchCV, Pipeline, ColumnTransformer, StandardScaler, OneHotEncoder), XGBoost
* **Imbalance handling:** `imbalanced-learn` SMOTE, applied to training data only, post-split
* **Geospatial:** `geopy.distance` for geodesic distance calculation
* **Data:** pandas, numpy — including time-based `rolling()` windows on datetime-indexed groups for velocity features
* **Visualization:** matplotlib, seaborn — recall-normalized confusion matrices (row-normalized for color, absolute counts as labels, so the fraud row stays readable despite the imbalance)
* **Site:** Jekyll with the Cayman theme, served via GitHub Pages

## Repository Structure

```
credit-card-fraud-detection/
├── final.ipynb              # Complete pipeline: features, 5 models, evaluation, clustering
├── MidtermCheckpoint.ipynb  # Midterm submission — supervised models only
├── index.html               # Full project report rendered via GitHub Pages
├── _config.yml              # Jekyll configuration
├── images/                  # Confusion matrices, feature importance, elbow curve
│   ├── randomConfusionMatrix.png
│   ├── SGDConfusionMatrix.png
│   ├── MLPConfusionMatrix.png
│   ├── xgboost.png
│   ├── featureimportantce.png
│   ├── elbow.png
│   └── unsupervised.jpg
├── requirements.txt
└── README.md
```

## Limitations

* **The data is synthetic.** Sparkov generates realistic transactions, but real fraud is adversarial — fraudsters actively adapt to detection systems, and a simulator has no such feedback loop. Performance here is an upper bound on what to expect against live fraud.
* **No temporal validation split.** Train and test come from the dataset's provided split rather than a strict chronological cutoff. For a production fraud system, evaluation should always train on the past and test on the future, since fraud patterns drift.
* **The decision threshold was never tuned.** All results use the default 0.5 cutoff. Because the models output probabilities, sweeping the threshold and selecting an operating point on the precision-recall curve would let the precision/recall balance be chosen deliberately rather than inherited from a default — the single highest-value next step for this project.
* **No cost-sensitive evaluation.** The right objective function would weight a missed fraud by average fraud loss and a false positive by review cost, then minimize expected cost. That would give a principled threshold rather than an implicit one.
* **`merchant_fraud_rate` is causally computed but still fragile.** It uses only prior transactions, but merchants with few historical transactions produce unstable estimates.

## What I Learned

* **Accuracy can be actively misleading, and knowing which metric to optimize is the real skill.** Every model here reports 94–99% accuracy while their fraud-catching ability ranges from adequate to poor. Learning to reach past the headline number to per-class recall, false negative rate, and the precision-recall curve — and to justify *why* recall is the right target for this specific problem — was the most transferable thing in the project.
* **Feature engineering beat model selection by a wide margin.** Time spent constructing `amt_zscore` and the velocity windows moved performance far more than any hyperparameter search. The grid search over 48 fits produced marginal gains; the per-cardholder normalization features produced the model's entire top three by importance. If I had limited time on a new problem, I now know where to spend it.
* **Temporal leakage is easy to introduce and hard to see.** Computing `merchant_fraud_rate` naively — as a simple groupby over the whole dataset — would let each transaction's own label contribute to its feature value, producing beautiful validation scores and a model that fails completely in production. Building the sort-then-`cumcount` pattern to enforce causality, and applying SMOTE only after the split, taught me to ask "could this feature have been computed at prediction time?" about every single column.
* **Relative features beat absolute ones.** `amt_to_median_ratio` and `amt_zscore` dominate the importance ranking while raw amount does not appear at all. Normalizing against each entity's own baseline rather than a population baseline is a pattern I now look for by default in any behavioral modeling problem.
* **The most intuitive features are not always the useful ones.** `impossible_travel_flag` was the feature I was most excited about — it encodes a genuinely damning signal, a card used in two places faster than physics allows. It landed at 0.006 importance. Building it, measuring it, and accepting the measurement over my intuition was a small but real lesson in letting evidence override a good story.
* **Interpreting results in domain terms, not just statistical ones.** Reporting "precision 0.13" without explaining that a 258:1 class ratio makes that arithmetically inevitable — and that the system is designed to feed a review queue rather than auto-decline — is a failure of communication, not of the model. Learning to translate a confusion matrix into the operational decision it implies changed how I write up results.

## Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

The most valuable contribution would be threshold optimization with a cost-sensitive objective — see [Limitations](#limitations).

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Contact Information

For questions or suggestions, reach me at [kavinuppal@gatech.edu](mailto:kavinuppal@gatech.edu).

## Acknowledgments

* [Sparkov Data Generation](https://github.com/namebrandon/Sparkov_Data_Generation) — transaction simulator
* [Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) on Kaggle
* [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/), [imbalanced-learn](https://imbalanced-learn.org/), [geopy](https://geopy.readthedocs.io/)
* Georgia Tech CS 7641 course staff, and my project teammates
