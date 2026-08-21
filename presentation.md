---
title: Predicting Car Testing Time
marp: true
---

# Predicting Car Testing Time

Mercedes-Benz Greener Manufacturing

---

## The Problem

- Every car configuration goes through physical testing before production.
- Testing time varies a lot by configuration — some take much longer than others.
- Long, unpredictable testing slows the manufacturing line and adds cost.
- Per Kaggle: more time on the test bench also means more CO2 emitted — faster testing, without lowering safety standards, is the "greener" in Greener Manufacturing.
- Goal: predict how long a car configuration will spend on the test bench, before it gets there.

---

## What the Product Does

- Input: a car's configuration (options and features selected).
- Output: a predicted testing time, in seconds.
- Use case: plan and prioritize the test schedule ahead of time instead of finding out testing time only after the car is on the bench.
- Example: one configuration in the data — anonymized codes like `X0=k`, `X1=v`, `X2=at`, plus around 360 option flags — took 130.81 seconds on the bench.

---

## Where the Value Is

- Better scheduling: route long-testing configurations to open slots instead of creating bottlenecks.
- Early warning: flag unusual configurations likely to need extra testing time.
- Faster iteration: manufacturing and engineering teams get a time estimate without waiting for the physical test.

---

## Our Approach

- Each car comes with 364 data points: 8 descriptive tags (like trim or option codes) and 356 simple on/off switches (is this feature installed or not).
- Compared several prediction methods (simple to complex).
- Best model: XGBoost, a tree-based model well suited to this kind of tabular, feature-heavy data.
- Tuned and validated using cross-validation to avoid overfitting to one lucky split of the data.

---

## What We Found

- The model explains about 57% of the variation in testing time — a solid signal, not a perfect one.
- Typical prediction is within about 4-5 seconds of the actual time.
- Accuracy holds up well across most configurations.

---

## Where It Struggles

- A small group of rare, unusual configurations (testing time above 130 seconds, under 1% of cars) sees much larger errors.
- These are cases with too few similar examples in the training data for the model to learn from.
- The model consistently *underestimates* these — it doesn't see them coming.

---

## Recommendations

- Use the model to prioritize and schedule the majority of standard configurations with confidence.
- Flag predictions in the high-time range for manual review rather than fully trusting them.
- Treat this as a scheduling aid, not a replacement for physical testing.

---

## Future Work

- Collect more examples of the rare, long-testing configurations to close the accuracy gap.
- Retrain periodically as new configurations and test results come in.
- Package the model as a simple internal tool the scheduling team can run on new configurations directly.
