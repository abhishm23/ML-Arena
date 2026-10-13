# ML Arena — bundled sample data

Seven ready-to-use datasets live in this folder. They are how a new user gets to
see ML Arena find something real in the first thirty seconds, so every one of
them contains a **genuine, findable pattern** — not random noise.

You never have to generate them: the CSVs are already committed. If you want to
rebuild them (they are byte-for-byte reproducible from a fixed seed):

```
python samples/generate_samples.py
```

Everything below is written for a **non-technical reader**. No jargon.

---

## How to use a sample

1. On the **Start** screen, click one of the bundled sample cards (or drag the CSV
   in from this folder).
2. ML Arena reads the file and shows you what is inside — how many rows, which
   columns look like money, which look like categories, and whether there are any
   data-quality problems.
3. Pick the **use case** listed below for that file.
4. Point the pickers at the columns listed under *"Map these columns"*.
5. Press run. The findings described under *"What you should find"* are really in
   the data — that is the point of the sample.

The filename always starts with the use case it was built for
(`orders_discount_leakage.csv` → use case **Discount leakage**), so if you are
unsure, trust the filename.

> Column names in the *"Map these columns"* tables are the actual CSV column
> names. The picker labels on screen are written in plain business English —
> pick the option whose description matches.

---

## 1. `orders_discount_leakage.csv` — the flagship demo

**Rows / columns:** ~60,000 orders × 20 columns · 11.3 MB · dated Jan 2024 – Dec 2025

**What it is.** Two years of order-level sales for a B2B technology reseller:
who bought, from which region, through which channel, from which sales rep, how
many units, at what list price and what they actually paid, plus cost and margin.

| Column | What it means |
|---|---|
| `order_id`, `order_date`, `customer_id` | the order, when it happened, who placed it |
| `customer_segment` | Enterprise / Mid-Market / SMB / Government |
| `region`, `country`, `sales_rep` | where it was sold and by whom |
| `product_category`, `product_name` | what was sold |
| `channel` | Direct / Partner / Distributor / Online |
| `quantity`, `list_price`, `unit_price`, `net_price` | how many, the list price, the negotiated price, the price after volume rebate |
| `discount_pct` | how far below list price the order actually landed |
| `revenue`, `cost`, `gross_margin_pct` | money in, money out, and the margin left |
| `payment_terms`, `order_status` | commercial terms and where the order got to |

**Use case to pick:** **Discount leakage**

**Map these columns**

| Picker | Column |
|---|---|
| The discount you want explained | `discount_pct` |
| Money at stake | `revenue` (or `gross_margin_pct` if you want margin risk) |
| Customer segment | `customer_segment` |
| Product / category | `product_category`, `product_name` |
| Sales channel | `channel` |
| Region | `region` |
| Sales rep | `sales_rep` |
| Quantity | `quantity` |
| Date | `order_date` |
| Customer | `customer_id` |

**What you should find.**
Most discounts in this file make sense: Enterprise customers get deeper
discounts than SMB, bulk orders get more than small ones, and Partner deals cost
more than direct ones. On top of that, roughly **one order in ten** is discounted
far more than its context justifies. Those orders are **not spread evenly** —
they cluster around a handful of sales reps and two regions. A good run will
surface those reps by name and put a rupee figure on the margin being given
away. Look for a headline like *"₹X crore of margin is at risk, concentrated in
7 reps."*

**Data-quality wrinkles (deliberately included).** `payment_terms` is missing on
~1.4% of rows, `country` on ~0.9%, `sales_rep` on ~0.6%, and there are 42 exact
duplicate rows. About 380 orders have a **negative** gross margin. The data
quality report should notice all of this — that is a feature, not a bug.

---

## 2. `orders_margin_overflow.csv` — margin erosion

**Rows / columns:** ~40,000 orders × 21 columns · 7.6 MB · dated Jan 2024 – Dec 2025

**What it is.** The same shape of order data, but the story is **margin**, not
discount. Every row carries `policy_margin_floor` — the minimum margin that
pricing policy says this product and segment should achieve.

**Use case to pick:** **Margin overflow**

**Map these columns**

| Picker | Column |
|---|---|
| The margin to check | `gross_margin_pct` |
| The policy floor | `policy_margin_floor` |
| Money at stake | `revenue` |
| Segment / product / channel / region / rep | `customer_segment`, `product_category`, `channel`, `region`, `sales_rep` |
| Quantity | `quantity` |
| Date | `order_date` |

**What you should find.** About **one order in five** lands below the policy
floor. Roughly two thirds of those got there through a discount that was too
deep; the rest got there because the cost of delivering the order came in too
high. The problem concentrates in a couple of regions and a few product
categories, and in the same over-discounting reps. You should see a headline
like *"~20% of orders miss the margin floor, worth ₹X crore of revenue."*

`below_margin_floor` is a ready-made 0/1 flag column in the file — handy if you
want to run this as a straight "predict the flag" problem instead of letting the
app derive it.

---

## 3. `sales_demand_series.csv` — demand forecasting

**Rows / columns:** ~30,700 daily rows × 8 columns · 1.6 MB · every day from 2024-01-01 to 2025-12-31

**What it is.** Daily sales for **6 regions × 7 product categories** (42 separate
series), with price, promotions and stock-outs.

| Column | What it means |
|---|---|
| `date`, `region`, `product_category` | the day and which series it belongs to |
| `units_sold` | how many units went out of the door |
| `avg_price` | average selling price that day |
| `promo_flag` | 1 if a promotion was running |
| `stockouts` | how many products were out of stock |
| `revenue` | units × price |

**Use case to pick:** **Demand forecast**

**Map these columns**

| Picker | Column |
|---|---|
| What to forecast | `units_sold` (or `revenue` to forecast money) |
| Date / time | `date` |
| Series to forecast separately | `region`, `product_category` |
| Extra signals | `avg_price`, `promo_flag`, `stockouts` |

**What you should find.** Four clear patterns, all deliberately planted and all
large enough to see with the naked eye:

* a **steady upward trend** of roughly +25% over the two years;
* a **weekly rhythm** — weekdays run ~2.5× weekend volume (it is a business
  product, so nobody buys on Saturday);
* an **annual peak** in March (financial year end) and a second one in
  October–November (festive season), plus a December lull;
* a **promotion lift** of roughly +30% on the days a promo runs.

A seasonal-naive baseline should already look decent; a good model should beat it.

---

## 4. `customers_churn.csv` — who is about to leave

**Rows / columns:** 25,000 customers × 11 columns · 1.7 MB · snapshot dated 2026-06-30

**What it is.** One row per customer: when they signed up, when they last bought,
how much they have spent, how often they have complained, how satisfied they say
they are, and what kind of contract they are on.

| Column | What it means |
|---|---|
| `customer_id`, `signup_date`, `last_purchase_date` | who, since when, until when |
| `total_orders`, `total_revenue`, `avg_order_value` | how much business they have done |
| `days_since_last_order` | how long since their last purchase |
| `support_tickets`, `satisfaction_score` | how much trouble they have had, and how they rated us (1–10) |
| `contract_type` | Monthly / Annual / Two-Year |
| `is_churned` | 1 = we consider this customer lost |

**Use case to pick:** **Churn risk**

**Map these columns**

| Picker | Column |
|---|---|
| Customer ID | `customer_id` |
| Last purchase date | `last_purchase_date` (or `days_since_last_order`) |
| Money measure | `total_revenue` |
| Extra signals | `total_orders`, `avg_order_value`, `support_tickets`, `satisfaction_score`, `contract_type`, `signup_date` |
| Outcome (optional) | `is_churned` |

**What you should find.** Churn is **learnable, not random**. Customers who have
gone quiet for a long time, who have raised more support tickets, who score
themselves low on satisfaction, and who are on Monthly contracts are far more
likely to have left. About **30%** of customers are churned. A good run should
land around 0.93–0.96 separation and should rank `days_since_last_order`,
`satisfaction_score` and `support_tickets` as the top drivers — with a rupee
figure attached to the revenue at risk.

`is_churned` is in the file so you can either pick it as the outcome or let ML
Arena work the label out from the purchase dates itself.

---

## 5. `loan_credit_default.csv` — credit risk

**Rows / columns:** 30,000 loan applications × 16 columns · 2.9 MB · dated Jan 2024 – Dec 2025

**What it is.** Loan applications with the classic credit-bureau signals:
income, existing debt, credit score, how many payments were late, and what rate
was offered.

| Column | What it means |
|---|---|
| `loan_id`, `customer_id`, `application_date` | the application |
| `loan_amount`, `term_months`, `interest_rate` | what was asked for and on what terms |
| `annual_income`, `employment_years` | the borrower's capacity to pay |
| `credit_score` | bureau score, roughly 300–850 |
| `existing_debt`, `debt_to_income` | what they already owe, and how big that is next to income |
| `num_prior_loans`, `num_late_payments` | borrowing history |
| `loan_purpose`, `home_ownership` | what the money is for, and where they live |
| `default_flag` | 1 = the loan went bad |

**Use case to pick:** **Credit default**

**Map these columns**

| Picker | Column |
|---|---|
| Outcome to predict | `default_flag` |
| Money at stake | `loan_amount` |
| Credit score | `credit_score` |
| Debt-to-income | `debt_to_income` (or `existing_debt` + `annual_income`) |
| Late payments | `num_late_payments` |
| Rate / term | `interest_rate`, `term_months` |
| Everything else | `annual_income`, `employment_years`, `num_prior_loans`, `loan_purpose`, `home_ownership`, `application_date` |

**What you should find.** About **12%** of loans default. Default is a genuine
function of **low credit score, high debt-to-income, more late payments and a
higher interest rate** — those four should dominate the driver table. A good run
should reach roughly 0.90–0.94 separation and translate the result into an
expected-loss figure in rupees.

---

## 6. `transactions_anomaly.csv` — finding the weird ones

**Rows / columns:** 50,000 card and ledger transactions × 12 columns · 4.7 MB · twelve months of 2025

**What it is.** A year of retail and card transactions: what was bought, through
which channel, on what device, in which currency, and how many times that account
transacted in the last 24 hours.

| Column | What it means |
|---|---|
| `transaction_id`, `timestamp` | the transaction and when it happened |
| `account_id` | which account |
| `merchant_category`, `channel`, `device_type` | where and how it happened |
| `amount`, `currency`, `country` | how much, in what money, where |
| `is_international` | 1 if the transaction crossed a border |
| `txn_count_24h` | how many transactions that account made in the day |

**Use case to pick:** **Transaction anomaly** (unsupervised — no outcome column
needed)

**Map these columns**

| Picker | Column |
|---|---|
| Money measure | `amount` |
| Category / dimension roles | `merchant_category`, `channel`, `currency`, `country`, `device_type` |
| Numeric signals | `txn_count_24h`, `is_international` |
| Identifier | `transaction_id`, `account_id` |
| Date / time | `timestamp` |

**What you should find.** About **1.5%** of these transactions are genuinely
unusual, and they are unusual in three recognisable ways: amounts that are 8–50×
larger than normal, transactions landing in the small hours of the morning, and
international transactions combined with very high same-day activity. An
isolation-forest-style detector finds them well — expect roughly **0.95+**
agreement with the truth. The app should let you browse the flagged rows and see
the pattern for yourself.

> **About the `_truth_anomaly` column.** This column is a **ground-truth answer
> key** kept only so we can measure how well the detector did. It is for
> evaluation, not analysis. **You can ignore it completely** — do not map it to
> any role. (It is prefixed with an underscore to make it obvious that it is
> metadata.)

---

## 7. `ml_arena_large_benchmark.csv` — the stress test

**Rows / columns:** ~500,000 orders × 20 columns · 94.5 MB · dated Jan 2024 – Dec 2025

**What it is.** The same order structure and the same discount-leakage story as
the flagship sample, just eight times bigger. Its only job is to prove the engine
stays fast on a large file and still scores every row.

**Use case to pick:** **Discount leakage**

**Map these columns:** identical to `orders_discount_leakage.csv` above.

**What you should find.** The same leakage story, at scale. Expect a longer
training step and a very large "margin at risk" number — this file is the one to
use when you want to show that ML Arena copes with a real ERP export rather than
a toy table.

---

## At a glance

| File | Rows | Cols | Size | Use case | Headline you should see |
|---|---:|---:|---:|---|---|
| `orders_discount_leakage.csv` | 60,042 | 20 | 11.3 MB | Discount leakage | ~10% of orders over-discounted, concentrated in a few reps |
| `orders_margin_overflow.csv` | 40,000 | 21 | 7.6 MB | Margin overflow | ~20% of orders below the margin policy floor |
| `sales_demand_series.csv` | 30,702 | 8 | 1.6 MB | Demand forecast | Trend + weekly rhythm + March/Nov peaks + promo lift |
| `customers_churn.csv` | 25,000 | 11 | 1.7 MB | Churn risk | ~30% churned; tickets, satisfaction and recency drive it |
| `loan_credit_default.csv` | 30,000 | 16 | 2.9 MB | Credit default | ~12% default; score, DTI and late payments drive it |
| `transactions_anomaly.csv` | 50,000 | 12 | 4.7 MB | Transaction anomaly | ~1.5% genuine anomalies, findable at 0.95+ agreement |
| `ml_arena_large_benchmark.csv` | 500,350 | 20 | 94.5 MB | Discount leakage | The flagship story, at scale |

**Total:** ~735,000 rows, ~134 MB.

---

## Notes for the curious

* **Everything is in Indian rupees (₹)** and stored as plain numbers — no
  currency symbols, no thousands separators — so any spreadsheet or model can
  read it directly.
* **Dates** are written as `YYYY-MM-DD`; the transaction timestamps are
  `YYYY-MM-DD HH:MM:SS`.
* **Every file is reproducible.** `generate_samples.py` uses one fixed random
  seed, so running it again produces byte-for-byte identical CSVs.
* **The patterns are planted on purpose**, but they are planted as *statistical*
  patterns — real noise, real overlap, real missing data — so the models have to
  do honest work to find them. A model that scores 0.99 on everything would be a
  bad sample; these are tuned so the answer is strong but not trivial.
