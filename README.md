# ✈️ Flight Delay Prediction (Delta Airlines)

## 📌 Project Overview

This project builds an end-to-end, leakage-safe machine learning pipeline to predict whether a Delta Airlines flight will be delayed by more than 15 minutes, using only information available at or before departure.

The goal is not just predictive performance, but operationally realistic decision-making, including:

class imbalance handling

validation-based threshold tuning

recall-prioritized optimization aligned with airline operations

## 🎯 Problem Definition

Given a scheduled Delta flight, predict:

Will this flight arrive more than 15 minutes late? (Yes / No)

Why this matters

Delays impact crew scheduling, gate planning, and passenger experience

Missing a delay (false negative) is often costlier than a false alert

The problem is naturally imbalanced (≈13% delayed flights)

## 🗂️ Dataset

### Source: US DOT On-Time Performance Dataset (via Kaggle)

### Files used:

flights.csv (main dataset, read from zip in chunks)

airlines.csv (lookup table)

airports.csv (lookup table)

### Scope

Airline: Delta Airlines only

Timeframe: Full dataset year

Rows after cleaning: ~870K flights

## 🧼 Data Cleaning & Target Definition

### Key cleaning steps:

Removed cancelled and diverted flights

Dropped rows with missing arrival delay

### Defined target:

delay_flag = 1 if ARRIVAL_DELAY > 15

Industry-standard delay definition

Final class distribution:

Delayed: ~13%

On-time: ~87%

## 🔐 Leakage Control (Critical)

To ensure deployability and honest evaluation:

Explicitly removed post-arrival and causal delay fields, including:

ARRIVAL_DELAY

AIR_SYSTEM_DELAY, WEATHER_DELAY, etc.

Used only pre-departure or scheduled information

This prevents inflated performance and ensures real-world applicability.

## 🛠️ Feature Engineering

All features are available before departure.

Time Features

Scheduled departure/arrival converted from HHMM → minutes

Departure hour

Peak-hour indicator (morning & evening congestion windows)

Network Features

Origin hub flag

Destination hub flag

Route feature (ORIGIN_DESTINATION)

Distance Features

Raw flight distance

Distance buckets (for EDA & interpretation)

## 📊 Exploratory Data Analysis (EDA)

EDA was used to validate feature choices, not just visualize data.

### Key insights:

Strong class imbalance

Higher delay rates during peak hours

Hub-origin flights show higher delay risk

Delay rates vary by flight distance

These insights directly informed feature engineering and metric choice.

## 🔀 Train / Validation / Test Split

### Data split strategy:

60% Train

20% Validation

20% Test

Stratified by target to preserve class imbalance

Fixed random seed for reproducibility

### Purpose:

Validation set used for model comparison & threshold tuning

Test set used once for final evaluation

## ⚙️ Preprocessing Pipeline

Implemented using Pipeline and ColumnTransformer to avoid leakage.

Numeric Features

Median imputation

Standard scaling (for model stability)

Categorical Features

Missing values filled with "UNKNOWN"

One-hot encoding with frequency thresholding (min_frequency=50)

Controls high-cardinality features like routes

## 🤖 Models Trained

Two models were trained and compared:

### 1️⃣ Logistic Regression (Baseline)

Interpretable

Class-weighted to handle imbalance

Serves as a transparent benchmark

### 2️⃣ Random Forest (Final Model)

Captures non-linear interactions

Controlled depth to reduce overfitting

Class-weighted

Reproducible (random_state=42)

## 📏 Evaluation Metrics

Due to class imbalance, accuracy was intentionally avoided.

### Primary metrics:

PR-AUC (primary model comparison metric)

ROC-AUC (secondary ranking metric)

F2 score (used for threshold tuning, recall-prioritized)

## 🎚️ Threshold Tuning (Key Design Choice)

Instead of using a default 0.5 threshold:

Swept thresholds from 0.05 → 0.95 on the validation set

Computed Precision, Recall, F1, and F2

Selected the threshold that maximized F2

### Why F2?

Recall is prioritized because missing a delay is more costly

Reflects operational airline decision-making

The final threshold was fixed before test evaluation.

## ✅ Final Test Performance (Random Forest)

At the selected operating point:

Recall (Delayed flights): ~82%

F2 score: ~0.44

PR-AUC: ~0.22

## Interpretation:

The model successfully identifies most delayed flights

Accepts more false positives to avoid missed delays

Performance is realistic for pre-departure prediction

## 📌 Key Takeaways

This is a production-oriented ML project, not a Kaggle exercise

Emphasis on:

leakage prevention

validation discipline

cost-aware decision thresholds

Demonstrates end-to-end data science thinking:

data engineering → modeling → decision policy

## 🚀 Future Improvements

Potential next steps:

Time-based splits to capture seasonality

Cost curves with ops-defined misclassification costs

Model monitoring & threshold recalibration over time

## 🧠 Skills Demonstrated

Large-scale data handling

Feature engineering with domain context

Imbalanced classification

Threshold tuning & decision analysis

Reproducible ML pipelines

Clear operational interpretation
