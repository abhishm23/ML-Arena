# -*- coding: utf-8 -*-
"""Build the ML Arena end-to-end test deliverable workbook."""
import os
from datetime import date

try:
    import openpyxl
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "openpyxl>=3.1.0"])
    import openpyxl

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.formatting.rule import CellIsRule


def xl_color(css_hex: str) -> str:
    value = css_hex.removeprefix("#").upper()
    if len(value) != 6:
        raise ValueError(f"Expected #RRGGBB, got: {css_hex}")
    return "FF" + value


XL_HDR = xl_color("#4472C4")
XL_HDR_TXT = xl_color("#FFFFFF")
XL_SOFT = xl_color("#D9E2F3")
XL_SUM = xl_color("#2F5597")
XL_BORDER = xl_color("#BFBFBF")
XL_RED_BG = xl_color("#FFC7CE"); XL_RED_TX = xl_color("#9C0006")
XL_GRN_BG = xl_color("#C6EFCE"); XL_GRN_TX = xl_color("#006100")
XL_AMB_BG = xl_color("#FFEB9C"); XL_AMB_TX = xl_color("#9C6500")
XL_CALC = xl_color("#F2F2F2")

thin = Side(style="thin", color=XL_BORDER)
BORDER = Border(left=thin, right=thin, top=thin, bottom=thin)
HDR_FONT = Font(bold=True, color=XL_HDR_TXT, size=10)
HDR_FILL = PatternFill("solid", fgColor=XL_HDR)
TITLE_FONT = Font(bold=True, size=14, color=XL_SUM)
WRAP = Alignment(vertical="top", wrap_text=True)
TOP = Alignment(vertical="top")
CTR = Alignment(horizontal="center", vertical="center", wrap_text=True)

OUT = r"C:\ML Arena\ML_Arena_End_to_End_Test_Deliverable.xlsx"
wb = Workbook()
wb.properties.title = "ML Arena End-to-End Test Deliverable"

# =====================================================================
# 1. TEST CASES  (header row 1, data from row 2)
# =====================================================================
ws = wb.active
ws.title = "Test Cases"
tc_cols = ["TC ID", "Use case", "Goal card in app", "Step(s)", "Test scenario",
           "Expected result", "Actual result observed", "Verdict", "Severity",
           "Evidence", "Developer comments", "Status"]
TC_HDR_ROW = 1
for c, name in enumerate(tc_cols, start=1):
    ws.cell(row=TC_HDR_ROW, column=c, value=name)

TC = [
 ("TC-01","UC1 churn_risk","Churn risk","1-6","Run the bundled 'Customers Churn' sample end to end",
  "A risk-ranked churn triage list with probability and contract value at stake",
  "No answer produced. 'Rows scored: 0 / Models tried: 0', no champion, no table. Error text appears only once, appended below the page footer.",
  "Fail","Critical","uc1_05_step6_bundled, uc1_06_step6_bundled_failed_allmapped"),
 ("TC-02","UC1 churn_risk","Churn risk","4","Check auto-mapping of an uploaded customer-level churn file",
  "account_id fills Customer ID, contract_value fills Spend/revenue",
  "Nothing auto-mapped. Money slot left empty even though contract_value was the only money column; all 7 slots blank for the bundled sample too.",
  "Fail","Medium","uc1_08_step4_synth15k_automap"),
 ("TC-03","UC1 churn_risk","Churn risk","2","Upload a 15,000-row file with 5% nulls in nps_score and seat_utilization_pct",
  "Nulls imputed and disclosed to the user",
  "Handled well. Banner: '2 columns have missing values - nps_score is the emptiest at 15%'. Per-column MISSING % shown. No crash, no silent drop.",
  "Pass","None","uc1_07_step2_synth15k"),
 ("TC-04","UC1 churn_risk","Churn risk","5-6","Stress: 1 GB / 14,000,000 rows",
  "Completes and returns a ranked list",
  "Failed. 'the _churn column contains only one value (Gone quiet)'. Rows scored 0. Ingest 5.1 s, run 11 s.",
  "Fail","Critical","uc1_11_step2_big, uc1_12_step6_big"),
 ("TC-05","UC1 churn_risk","Churn risk","6","Verify the app uses the user's own is_churned target column",
  "The labelled outcome column is used as the target",
  "The app ignores is_churned entirely and derives its own _churn label from the Date column. Proof: a file with is_churned set to all zeros still ran successfully.",
  "Fail","Critical","uc1_09_step6_txn_success"),
 ("TC-06","UC1 churn_risk","Churn risk","6","Single-value target: is the error friendly?",
  "Plain-English explanation with a suggested next step",
  "The plain-English text is good, but it only ever appears for the app's own derived label - never for the user's target column. The page otherwise renders as a success.",
  "Partial","High","uc1_06_step6_bundled_failed_allmapped"),
 ("TC-07","UC1 churn_risk","Churn risk","4","Confirm the target column is excluded from the predictors",
  "is_churned not offered as an input feature",
  "is_churned is offered as a predictor in the Customer-ID, Product, Segment and Channel slots - inviting target leakage.",
  "Fail","Medium","uc1_08_step4_synth15k_automap"),
 ("TC-08","UC2 discount_leakage","Discount leakage","1-6","Run the bundled 'Orders Discount Leakage' sample",
  "A leakage triage list naming the offending sales reps",
  "Runs clean (XGBoost, separation power 0.97, 242.86M at risk, 79 s) but names only ONE rep value and provides no per-rep table, count or amount. The business question 'which reps?' is only partly answered.",
  "Partial","High","uc2_03_step6_bundled, uc2_04_step6_clean"),
 ("TC-09","UC2 discount_leakage","Discount leakage","5-6","Synthetic 60k file with REP-10 / REP-22 systematically over-discounting",
  "The two rogue reps are flagged at a higher rate than everyone else",
  "Pass. Flagged rate REP-22 15.7%, REP-10 15.2% vs 4.6-5.1% for all other reps. Top driver is sales_rep_id.",
  "Pass","None","uc2_05_step6_synth60k"),
 ("TC-10","UC2 discount_leakage","Discount leakage","6","Check the triage export carries an order / reference ID",
  "Export includes the row identifier so the user can find the order",
  "Export columns are score, predicted, probability, actual, net_price, quantity, customer_segment, region - no order ID, despite the app claiming rows keep their identity.",
  "Fail","Medium","uc2_04_step6_clean"),
 ("TC-11","UC2 discount_leakage","Discount leakage","5-6","Stress: 1 GB / 13,250,000 rows",
  "Completes and returns a ranked list",
  "Failed at 372 s with an unhandled IndexError: list index out of range. Rows scored 0, Models tried 7, empty scoreboard.",
  "Fail","Critical","uc2_06_step2_big, uc2_07_step6_big"),
 ("TC-12","UC2 discount_leakage","Discount leakage","2","Edge case: rows with list_price = 0 and discount_pct = 150",
  "Invalid rows validated, excluded or flagged",
  "No validation at all. Negative net prices were computed and then flagged as the TOP leakage rows, and rolled into the headline money-at-risk figure.",
  "Fail","High","uc2_05_step6_synth60k"),
 ("TC-13","UC2 discount_leakage","Discount leakage","5","Read the time-budget message after a long run",
  "A coherent, self-consistent message",
  "'We have used 372 of the 75 seconds allowed' and 'the 0 that finished are more than enough to pick a winner from'.",
  "Fail","High","uc2_07_step6_big"),
 ("TC-14","UC3 margin_overflow","Margin overflow","1-6","Run the bundled 'Orders Margin Overflow' sample",
  "Ranked list of policy-violating orders with margin at risk",
  "Runs clean in 57 s. LightGBM, separation power 1.00, 40,000 rows scored, 16.72 billion 'Margin at risk'.",
  "Pass","None","uc3a_04_step6_bundled"),
 ("TC-15","UC3 margin_overflow","Margin overflow","2","Build the brief's rule in the derived-column builder: ifelse((revenue-cost)/revenue < 0.22, 1, 0)",
  "Formula accepted and computed correctly",
  "Works first try. Preview returned Int64, MIN 0, MAX 1, MEAN 0.031, NULLS 0 over 25,000 rows - 775 flagged rows, matching the planted signal plus baseline.",
  "Pass","None","uc3b_01_builder_ACCEPTED"),
 ("TC-16","UC3 margin_overflow","Margin overflow","4","Find where the user can enter their own 22% margin floor",
  "A control to set the policy floor",
  "No floor, threshold or policy control exists anywhere in the product. The app silently substitutes its own rule (the 25th percentile of margin within the row's segment), so the answer changes with the data.",
  "Fail","Critical","uc3a_02_step4_bundled_automap_EMPTY"),
 ("TC-17","UC3 margin_overflow","Margin overflow","4","Check Step 4 auto-maps the bundled sample",
  "Slots pre-filled from an obvious match",
  "All 9 slots read 'choose a column' and Run stayed disabled. The bundled file contains exactly the right columns and none were pre-filled.",
  "Fail","High","uc3a_02_step4_bundled_automap_EMPTY"),
 ("TC-18","UC3 margin_overflow","Margin overflow","6","Verify the impact figure means margin lost",
  "The money figure represents margin eroded",
  "Mislabelled. The headline says 'Margin at risk' but the sub-caption admits it sums the money column over flagged rows - i.e. billed revenue, not margin lost.",
  "Fail","High","uc3a_04_step6_bundled"),
 ("TC-19","UC3 margin_overflow","Margin overflow","6","Check the driver list is not circular",
  "Drivers are genuine predictors, not the target",
  "The margin column used to build the label is fed back in as a feature: 'The strongest signal is gross_margin_pct (100% of the model's decision-making)', importance 99.91. Every other driver reads 0.00-0.05.",
  "Fail","High","uc3a_04_step6_bundled"),
 ("TC-20","UC3 margin_overflow","Margin overflow","2","Edge case: jobs with zero direct cost and zero billed revenue",
  "No crash; zeros handled explicitly",
  "No crash, no Infinity and no NaN - division is guarded and zero revenue yields a blank that Step 2 discloses. However zero-cost rows silently produce a 'perfect' margin of 1.0 with no warning.",
  "Partial","Low","uc3c_01_step4_automap_EMPTY_nodate"),
 ("TC-21","UC3 margin_overflow","Margin overflow","5-6","Stress: 1 GB / 12,000,000 rows",
  "Completes and returns a ranked list",
  "Failed at 424 s with IndexError: list index out of range, Rows scored 0. It sat at 9% for about 370 s first. The app's own 75 s budget was exceeded 5.6x.",
  "Fail","Critical","uc3d_02_step6_big_FAILED"),
 ("TC-22","UC4 revenue_predict","What drives revenue","5-6","Synthetic 35,000 opportunities built from headcount x duration x industry",
  "Step 6 returns a ranked driver list that recovers the true drivers",
  "Pass. Top two drivers are consultant_headcount_requested (44.8) and estimated_duration_weeks (44.1), in the user's own column names. Financial_Services direction recovered correctly.",
  "Pass","None","uc4b_03_step6_synth35k"),
 ("TC-23","UC4 revenue_predict","What drives revenue","5-6","Five extreme outlier deals above 10 million",
  "Outliers do not distort the ranking",
  "Pass. The 11.2M-19.2M deals were kept unclipped and the top-two driver ranking was unchanged.",
  "Pass","None","uc4b_03_step6_synth35k"),
 ("TC-24","UC4 revenue_predict","What drives revenue","4-6","Verify the promise 'leave drivers empty and we will use every other usable column'",
  "All usable columns used as evidence",
  "False whenever any other optional slot is filled. The app silently dropped quantity, list_price, unit_price, net_price, discount_pct, cost and gross_margin_pct, giving R2 12% and a driver list naming weekday and month.",
  "Fail","Critical","uc4a_03_step6_revenue"),
 ("TC-25","UC4 revenue_predict","What drives revenue","6","Read the impact card label",
  "The label matches what the number is",
  "'Money the model gets right' actually holds the sum of absolute errors - the money the model gets wrong. The label reads as a positive outcome.",
  "Fail","High","uc4a_03_step6_revenue"),
 ("TC-26","UC4 revenue_predict","What drives revenue","6","Check 'the rows to look at first' for a regression goal",
  "Rows sorted by largest error",
  "Meaningless for regression: score and predicted are identical, so the table is sorted by largest predicted revenue, not by error. The genuinely wrong rows are not surfaced.",
  "Fail","High","uc4a_03_step6_revenue"),
 ("TC-27","UC4 revenue_predict","What drives revenue","6","Check the export carries the opportunity ID",
  "Export includes opportunity_id",
  "No opportunity_id in the table or the export, so no row can be traced back to a deal.",
  "Fail","Medium","uc4b_03_step6_synth35k"),
 ("TC-28","UC4 revenue_predict","What drives revenue","5-6","Stress: 1 GB / 15,000,000 rows",
  "Completes within a usable time",
  "TIMEOUT at 480 s, still at 16% and on 'Working out exactly what we are looking for'. No error, no result.",
  "Fail","High","uc4c_02_step5_TIMEOUT480"),
 ("TC-29","UC5 demand_forecast","Demand forecast","1-6","Run the bundled 'Sales Demand Series' sample",
  "A forward demand forecast per SKU-warehouse",
  "Runs in 69 s with R2 0.98, but it is a backtest, not a forecast.",
  "Partial","High","uc5_06_answer"),
 ("TC-30","UC5 demand_forecast","Demand forecast","6","Look for future dates, predicted units and a horizon control",
  "Future periods with predicted units and a way to set how far ahead",
  "None. No future dates, no predicted units anywhere, and no horizon or 'days ahead' control at any step. A DOM probe found 'horizon', 'future' and 'next month' zero times on Step 6.",
  "Fail","High","uc5_06_answer"),
 ("TC-31","UC5 demand_forecast","Demand forecast","5-6","Check whether day-of-week seasonality and the temperature effect are surfaced",
  "The planted Saturday and temperature signals appear as drivers",
  "Both missed. Drivers are only lag and rolling features (lag_1..lag_21, average of last 7/28, wobble over last 7/28). local_temperature_f and is_promotion never appear even when supplied. sku_id dominated at 87.5%.",
  "Fail","High","uc5s_06_answer"),
 ("TC-32","UC5 demand_forecast","Demand forecast","2","Check column typing of the SKU identifier",
  "sku_id recognised as an identifier",
  "sku_id (values like SKU-BEV-001) was typed as Category, so it was usable as a grouping and then dominated the model, pulling R2 down from 0.77 to 0.66.",
  "Fail","Medium","uc5s_04_columns"),
 ("TC-33","UC5 demand_forecast","Demand forecast","5-6","Stress: 16,000,000 rows / 872 MB",
  "Completes within a usable time",
  "Completed in 349 s, R2 0.69. Ingest 3.0 s. Disclosed its own 400k-row training sample honestly.",
  "Pass","None","uc5b_06_answer"),
 ("TC-34","UC5 demand_forecast","Demand forecast","5-6","Run with 'Split into separate series = warehouse_id'",
  "Either a result or a clear, actionable error",
  "Died in 155 ms showing only 'finished in 153 ms / Rows scored: 0 / Models tried: 0' with no error text at all - a dead end with no cause and no suggested fix.",
  "Fail","High","uc5s_06_failed"),
 ("TC-35","UC6 price_elasticity","Price & discount response","1-6","Run the bundled sample on the price-response goal",
  "The revenue-maximising discount depth",
  "Runs in 51 s. Returns one sentence of prose: 'The highest total revenue in this sweep comes at about 0.35'.",
  "Partial","High","uc6_06_answer"),
 ("TC-36","UC6 price_elasticity","Price & discount response","6","Look for an elasticity or revenue-versus-discount curve",
  "A curve or sweep table the buyer can read",
  "None. Step 6 contains only a score histogram, a model-ranking bar chart and a predicted-versus-actual scatter. The optimum is a single sentence under the label ELASTICITY.",
  "Fail","High","uc6_06_answer"),
 ("TC-37","UC6 price_elasticity","Price & discount response","6","Check whether the optimum is broken down by apparel category",
  "A separate optimum per category",
  "One global optimum only. segments is empty in the run payload and 'per category' appears zero times. Category is used as a model driver but never as a breakdown axis.",
  "Fail","High","uc6_06_answer"),
 ("TC-38","UC6 price_elasticity","Price & discount response","5-6","Correctness: data where Basics peaks at 30% and Outerwear is inelastic",
  "The app recovers the per-category peak",
  "Wrong, and wrong in the dangerous direction: it recommends 60% - the deepest step in the grid - for every category. That is the worst point for Outerwear and past the peak for all the others.",
  "Fail","High","uc6s_06_answer"),
 ("TC-39","UC6 price_elasticity","Price & discount response","5-6","Stress: 12,000,000 rows / 790 MB",
  "Completes within a usable time",
  "Completed in 175 s. Ingest 1.4 s. Same wrong recommendation at scale ('about 60.00').",
  "Pass","None","runs result.json (browser driver dropped mid-poll; no screenshot)"),
 ("TC-40","UC6 price_elasticity","Price & discount response","6","Check whether 'elasticity' is defined for a non-technical buyer",
  "The term is explained",
  "The app labels a step ELASTICITY but never defines the word anywhere in the UI. 'this sweep' and 'your typical price' are also never defined.",
  "Fail","Medium","uc6_06_answer"),
 ("TC-41","UC7 credit_default","Late payment & default","1-6","Run the bundled 'Loan Credit Default' sample",
  "Risk-ranked accounts with expected loss and drivers",
  "Runs in 31 s. Expected loss 1.47 billion shown, key-driver table present. Two categorical drivers tie at exactly 100 / 30%, which reads like a placeholder.",
  "Partial","Medium","uc7_07_step6_answer_top"),
 ("TC-42","UC7 credit_default","Late payment & default","5-6","Synthetic 50k with NSF >= 3 AND DSR < 1.05 planted at a 48% default rate",
  "The planted drivers rank top and the triage list concentrates defaults",
  "Strong pass. DSR (56%) and NSF incidents (32%) are the top two drivers, 88% combined. Flagged set holds 921 real defaults = 36.4% precision vs 4.50% base = 8.1x lift; 100% of the planted cell is flagged.",
  "Pass","None","uc7c_03_step6_synth50k"),
 ("TC-43","UC7 credit_default","Late payment & default","6","Check the ranked list and export carry an account identifier",
  "Account ID present so collections can act",
  "Neither the on-screen table nor the CSV export carries loan_id or customer_id, although the goal is literally 'rank the accounts most likely to go bad'. Only the per-row 'why' cards name the account.",
  "Fail","High","uc7_07_step6_answer_top, uc7_08_step6_full"),
 ("TC-44","UC7 credit_default","Late payment & default","4-6","Map only the exposure column, trusting the 'Known outcome (optional)' label",
  "Optional really means optional, or the run is blocked up front",
  "The slot is labelled Optional but the engine hard-requires it. Run became enabled, then failed with Rows scored 0 and a blank answer screen. The real reason was buried in the Step 5 log.",
  "Fail","High","uc7_05_step6_FAILED_norows"),
 ("TC-45","UC7 credit_default","Late payment & default","5","Read the run status after that failure",
  "A failed run is reported as failed",
  "Contradictory: Step 5 showed 'All done - here is what we found', 'The winning model came out on top' and 'Finished 100%' with a full progress bar, while the same screen's log said 'Stopped early ... ValueError' and 'Model scoreboard 0 models'.",
  "Fail","High","uc7_06_step5_narration_failed"),
 ("TC-46","UC7 credit_default","Late payment & default","2","Synthetic file with 7% blank bureau_score and negative bank balances",
  "Missing values named and the handling disclosed",
  "Silent but clean. No row dropped, all 50,000 scored, but the banner never names which column had gaps and never says what it did. The word 'imputation' never appears.",
  "Partial","Medium","uc7c_01_step2_synth50k"),
 ("TC-47","UC7 credit_default","Late payment & default","5-6","Stress: 15,000,000 rows / 961 MB",
  "Completes within a usable time",
  "Completed in 192 s, ingest 5.6 s, planted signal still ranks first and second. But 'Models tried: 7' against 'came out on top of 2 models' - five models never raced.",
  "Pass","None","uc7d_01_step2_1GB"),
 ("TC-48","UC8 transaction_anomaly","Unusual records","1-6","Run the bundled 'Transactions Anomaly' sample",
  "A ranked shortlist of unusual rows with reasons",
  "Runs in 2m 11s. 2,500 of 50,000 flagged. Ranked table and per-row 'why' cards exist, and the export carries the ID.",
  "Partial","High","uc8_04_step6_answer"),
 ("TC-49","UC8 transaction_anomaly","Unusual records","5-6","Correctness: 80,000 rows with 200 unambiguous planted anomalies",
  "The planted anomalies are surfaced",
  "Failed badly. Only 7 of 200 surfaced (3.5%). Ghost-vendor rows (new vendor, round 9990, 3am Sunday): 0 of 100. Flag precision 0.175% against a 0.250% base rate - lift 0.70x, i.e. worse than picking rows at random.",
  "Fail","High","uc8c_02_step6_synth80k"),
 ("TC-50","UC8 transaction_anomaly","Unusual records","6","Look for false-positive guidance and a way to move the flag cut-off",
  "A false-alarm estimate and a controllable threshold",
  "None. No precision, no false-alarm count, no slider (zero range inputs on the page). The app flags a fixed top 5%, which is what cuts the planted rows off.",
  "Fail","High","uc8_04_step6_answer"),
 ("TC-51","UC8 transaction_anomaly","Unusual records","2","Inspect the columns of the bundled anomaly sample",
  "The demo file must not contain the answer",
  "The bundled sample ships a column named _truth_anomaly (Yes/No) - the answer key - listed as a normal column and offered as a Context column in the flow.",
  "Fail","High","uc8_01_step2_data"),
 ("TC-52","UC8 transaction_anomaly","Unusual records","6","Read the per-model explanation on an unsupervised run",
  "Copy that matches the unsupervised method",
  "Every model shows the supervised-failure message: 'we could not score it fairly - usually because your data only contains one of the two outcomes'. Step 6 also says 'How we tested: Unsupervised'. Wrong branch of the copy.",
  "Fail","Medium","uc8_04_step6_answer"),
 ("TC-53","UC8 transaction_anomaly","Unusual records","6","Check the 'groups we found' profile column renders",
  "Readable profile text",
  "Raw JavaScript objects leak into the UI: PROFILE renders as [object Object],[object Object] in every row.",
  "Fail","Medium","uc8_04_step6_answer"),
 ("TC-54","UC8 transaction_anomaly","Unusual records","5-6","Stress: 14,500,000 rows / 999 MB",
  "Completes and reports a meaningful consensus",
  "Completed at 458 s (19 s inside the cap) but only one model finished: 'Models tried: 3', one scoreboard row, 'came out on top of 1 models', and 'agreement with the crowd 1.00' for a consensus of one.",
  "Partial","High","uc8d_02_step6_1GB"),
 ("TC-55","UC9 segmentation","Customer segments","1-6","Run 'Customer segments' on the bundled sample",
  "Named customer segments with sizes and profiles",
  "Runs in 58 s. Headline 'Your customers fall into 4 natural groups' with real sizes (3,319 / 8,580 / 3,634 / 9,467).",
  "Partial","High","uc9bundled_06_step6answer"),
 ("TC-56","UC9 segmentation","Customer segments","5-6","Correctness: 30,000 rows with 4 planted behavioural personas",
  "The 4 personas are recovered",
  "Pass. Produced sizes 6,996 / 7,529 / 6,501 / 8,974 against planted 7,000 / 7,500 / 6,500 / 9,000 - a clean one-to-one mapping agreeing on 30,000 of 30,000 rows (99.99% at 8M rows).",
  "Pass","None","uc9s_6c_groups"),
 ("TC-57","UC9 segmentation","Customer segments","6","Check whether the segments are named for a marketer",
  "Business-language persona names",
  "Labelled literally 'Group 1' to 'Group 4'. No persona or business name is ever produced, though the card promises to 'describe each group in words you can use in a meeting'.",
  "Fail","High","uc9b_6c_groups"),
 ("TC-58","UC9 segmentation","Customer segments","6","Check the per-group PROFILE column",
  "Readable per-group profile",
  "Broken. PROFILE renders as [object Object],[object Object],... for all four groups - the underlying data is a proper array but is stringified straight into the table.",
  "Fail","High","uc9b_6c_groups"),
 ("TC-59","UC9 segmentation","Customer segments","6","Try to choose the number of segments",
  "A control for k",
  "None. The app picks 4 and explains its choice, but the user cannot ask for 3 or 5.",
  "Fail","Medium","uc9bundled_06_step6answer"),
 ("TC-60","UC9 segmentation","Customer segments","4","Check Step 4 auto-maps the behaviour-numbers slot",
  "Slot pre-filled",
  "The required 'Behaviour numbers' slot opens empty and Run stays disabled with 'Still to choose: Behaviour numbers'. The user must Ctrl-click a native multi-select unaided.",
  "Fail","High","uc9bundled_03_step4slots"),
 ("TC-61","UC9 segmentation","Customer segments","5-6","Stress: 8,000,000 rows / 452 MB",
  "Completes within a usable time",
  "Completed in 229 s, ingest 1.8 s, cluster recovery 99.99%. Only 2 of 3 planned models ran ('We used 214 of the 75 seconds allowed').",
  "Pass","None","uc9big_06_answer"),
 ("TC-62","UC10 freight audit","Margin overflow + derived column","2","Build the brief's formula: ifelse((contracted_budget_cost - actual_carrier_invoice) / contracted_budget_cost < -0.35, 1, 0)",
  "Formula accepted and computed correctly",
  "Pass. Preview returned Int64, MIN 0, MAX 1, MEAN 0.043, NULLS 0 over 45,000 rows - about 1,935 rows, matching the planted signal. Error messages are genuinely helpful.",
  "Pass","None","uc10_builder_ok_crop"),
 ("TC-63","UC10 freight audit","Margin overflow + derived column","4","Check whether the derived flag can be the thing being predicted",
  "The user's own business rule drives the answer",
  "No. The flag is offered only as a Yes/No grouping category; the app always derives its own label ('Below policy floor'). The brief's -35% rule is never what gets predicted.",
  "Fail","High","uc10synth_03..06"),
 ("TC-64","UC10 freight audit","Margin overflow + derived column","5-6","Check for label circularity",
  "The target is not fed back in as a feature",
  "Circular. The label is a residual of margin_ratio and margin_ratio is also supplied as a feature, giving ROC AUC 0.99998 and precision 0.999 - a meaningless perfect score. The app pre-empts the objection in a note instead of removing the leak.",
  "Fail","High","uc10synth_06_answer"),
 ("TC-65","UC10 freight audit","Margin overflow + derived column","6","Check whether the worst offenders are surfaced as a league table",
  "A named list of the worst carriers and lanes",
  "Only implicitly. SPOT_BROKER_X on Reefer flags at 95.1% vs 28.4% baseline (every row below -35% is caught) but appears only as a 1% footnote under margin_ratio (79.2%). No offender table and no sentence naming the culprit.",
  "Partial","High","uc10synth_06_answer"),
 ("TC-66","UC10 freight audit","Margin overflow + derived column","3-6","Check the goal speaks the logistics analyst's language",
  "Copy that matches a freight audit",
  "The words accessorial, linehaul, detention and fuel surcharge never appear. The goal story is retail language ('pricing policy', 'segment', 'channel'), so a 3PL analyst will not recognise their own problem.",
  "Fail","Medium","uc10bundled_03..06"),
 ("TC-67","UC10 freight audit","Margin overflow + derived column","2-6","Edge case: zero-variance rows (contracted == actual) and 5% missing transit_miles",
  "Both subsets handled without breaking the engine",
  "Pass. The 2,254 zero-variance rows got a 0.0% flag rate (correctly treated as on-policy) and the 2,277 missing-mile rows behaved like the baseline. No crash, no warning.",
  "Pass","None","uc10synth_03..06"),
 ("TC-68","UC10 freight audit","Margin overflow + derived column","5-6","Stress: 9,000,000 rows / 758 MB",
  "Completes within a usable time",
  "Failed at 181.5 s with IndexError: list index out of range, Rows scored 0, no scored file written.",
  "Fail","Critical","uc10big_03..06"),
 ("TC-69","UC10 freight audit","Margin overflow + derived column","2","Check whether uploaded files are retained after the run",
  "Uploads cleaned up or at least bounded",
  "Every upload leaves a full copy in data/ indefinitely. 15 copies (about 3.4 GB) accumulated across four files in 30 minutes, including three copies of the 758 MB file.",
  "Fail","Medium","uc10big_03..06"),
 ("TC-70","All use cases","All","any","Reload the browser page mid-flow",
  "The session resumes or warns the user",
  "All state is lost. A reload at any step returns the user to Step 1 with no resume and no warning.",
  "Fail","Medium","uc2_21_after_reload"),
 ("TC-71","All use cases","All","5","Click a rail step while a run is in flight",
  "The run is preserved, or the user is asked to confirm",
  "The in-flight run is silently cancelled with no confirmation dialog. A user can lose 60+ seconds of computation with one click.",
  "Fail","Medium","uc2_07_back_behaviour"),
 ("TC-72","All use cases","All","6","Click an export button",
  "A file downloads without disturbing the app",
  "The export uses a full-page navigation to an attachment URL, so the single-page app is destroyed and all state is lost, with no back button.",
  "Fail","Low","uc7_08_step6_full"),
 ("TC-73","All use cases","All","5-6","Look for a definition of 'Stratified Split' and 'Hold-Out'",
  "The testing method is explained in plain words",
  "Not explained anywhere - no tooltip and no glossary entry. The metric cards do have good plain-English hovers; the method label does not. For a regression target, 'stratified' is not even the right word.",
  "Fail","Medium","uc3a_04_step6_bundled"),
 ("TC-74","All use cases","All","4","Auto-mapping across the nine goals",
  "Slots pre-filled from obvious column-name matches",
  "Auto-mapping is absent on most goals (margin_overflow, deal approval, segments, credit default, anomaly, price response). Only the date slot fills. This is the single most repeated usability failure in the test.",
  "Fail","High","uc3a_02_step4_bundled_automap_EMPTY, uc9bundled_03_step4slots"),
 ("TC-75","All use cases","All","2","Measure file ingest speed",
  "Large files ingest quickly",
  "Strong pass. Ingest is columnar and fast: 250 MB in 1.3 s, 1 GB in 2.3-9.4 s, 16 M rows in 3.0 s. Ingest is never the bottleneck.",
  "Pass","None","uc2_15_dl_250mb_ingested, uc7d_01_step2_1GB"),
 ("TC-76","All use cases","All","5-6","Compare 'Models tried: N' with the champion sentence",
  "The two numbers agree",
  "They frequently disagree - 'Models tried: 11' vs 'came out on top of 8 models', 'Models tried: 3' vs 'came out on top of 1 models'. The 75 s budget is only checked between models, so it overruns by up to 5.6x.",
  "Fail","Medium","uc7_07_step6_answer_top"),
]
r = TC_HDR_ROW + 1
for row in TC:
    for c, v in enumerate(row, start=1):
        ws.cell(row=r, column=c, value=v)
    r += 1
TC_LAST = r - 1
TC_FIRST = TC_HDR_ROW + 1

# ---- Test Cases styling
for c in range(1, len(tc_cols) + 1):
    cell = ws.cell(row=TC_HDR_ROW, column=c)
    cell.font = HDR_FONT; cell.fill = HDR_FILL; cell.alignment = CTR; cell.border = BORDER
for rr in range(TC_FIRST, TC_LAST + 1):
    for c in range(1, len(tc_cols) + 1):
        cell = ws.cell(row=rr, column=c)
        cell.border = BORDER
        cell.alignment = WRAP if c in (5, 6, 7, 11) else TOP
    ws.cell(row=rr, column=11).fill = PatternFill("solid", fgColor=XL_CALC)   # Developer comments
    ws.cell(row=rr, column=12).fill = PatternFill("solid", fgColor=XL_CALC)   # Status
widths = {"A": 9, "B": 16, "C": 20, "D": 9, "E": 42, "F": 40, "G": 56, "H": 10, "I": 10, "J": 30, "K": 30, "L": 14}
for k, v in widths.items():
    ws.column_dimensions[k].width = v
ws.freeze_panes = "C2"
ws.auto_filter.ref = f"A{TC_HDR_ROW}:L{TC_LAST}"

# verdict + severity conditional formatting
ws.conditional_formatting.add(f"H{TC_FIRST}:H{TC_LAST}",
    CellIsRule(operator="equal", formula=['"Fail"'], fill=PatternFill("solid", fgColor=XL_RED_BG), font=Font(color=XL_RED_TX, bold=True)))
ws.conditional_formatting.add(f"H{TC_FIRST}:H{TC_LAST}",
    CellIsRule(operator="equal", formula=['"Partial"'], fill=PatternFill("solid", fgColor=XL_AMB_BG), font=Font(color=XL_AMB_TX, bold=True)))
ws.conditional_formatting.add(f"H{TC_FIRST}:H{TC_LAST}",
    CellIsRule(operator="equal", formula=['"Pass"'], fill=PatternFill("solid", fgColor=XL_GRN_BG), font=Font(color=XL_GRN_TX, bold=True)))
ws.conditional_formatting.add(f"I{TC_FIRST}:I{TC_LAST}",
    CellIsRule(operator="equal", formula=['"Critical"'], fill=PatternFill("solid", fgColor=XL_RED_BG), font=Font(color=XL_RED_TX, bold=True)))
ws.conditional_formatting.add(f"I{TC_FIRST}:I{TC_LAST}",
    CellIsRule(operator="equal", formula=['"High"'], fill=PatternFill("solid", fgColor=XL_AMB_BG), font=Font(color=XL_AMB_TX, bold=True)))

dv_verdict = DataValidation(type="list", formula1='"Pass,Fail,Partial,Blocked,Not tested"', allow_blank=True)
dv_sev = DataValidation(type="list", formula1='"Critical,High,Medium,Low,None"', allow_blank=True)
dv_status = DataValidation(type="list", formula1='"Open,In progress,Fixed,Deferred,Won\'t fix,Retest needed,Closed"', allow_blank=True)
ws.add_data_validation(dv_verdict); ws.add_data_validation(dv_sev); ws.add_data_validation(dv_status)
dv_verdict.add(f"H{TC_FIRST}:H{TC_LAST}")
dv_sev.add(f"I{TC_FIRST}:I{TC_LAST}")
dv_status.add(f"L{TC_FIRST}:L{TC_LAST}")

print("Test Cases rows:", TC_LAST - TC_HDR_ROW)

# =====================================================================
# 2. DEFECTS  (header row 1)
# =====================================================================
wd = wb.create_sheet("Defects")
d_cols = ["Defect ID", "Use case", "Title", "Severity", "Priority", "Description / reproduction",
          "Business impact", "TC refs", "Developer comments", "Status", "Fix version", "Verified by"]
for c, name in enumerate(d_cols, start=1):
    wd.cell(row=1, column=c, value=name)

D = [
 ("BUG-01","All","Unhandled IndexError on large files - engine-wide","Critical","P0",
  "A run over roughly 750 MB-1 GB fails with an unhandled Python IndexError: list index out of range. Reproduced on 4 goals independently: discount_leakage (372 s), margin_overflow (424 s), margin_overflow via freight (181 s), and a contention-induced failure on a 60k-row file. Rows scored 0.",
  "The product cannot complete the headline promise at the data volumes the brief targets. The user gets an empty answer.",
  "TC-11, TC-21, TC-68","","","",""),
 ("BUG-02","All","A failed run renders as a success page","Critical","P0",
  "When a run fails, Step 6 still shows 'finished in Xs. Everything below was computed on this computer.' with Rows scored: 0 and no red banner. In credit_default Step 5 showed 'All done - here is what we found', 'Finished 100%' and a full progress bar while the same screen's log said 'Stopped early ... ValueError'.",
  "A business user cannot tell a failed analysis from a successful one, and may act on an empty result.",
  "TC-45, TC-11, TC-06","","","",""),
 ("BUG-03","All","No auto-mapping on most goals","Critical","P0",
  "Step 4 opens with every slot empty and Run disabled on margin_overflow, credit_default, transaction_anomaly, segmentation and price_elasticity, even for the app's own bundled samples whose columns match exactly. Only the Date slot auto-fills.",
  "The most repeated friction in the product. A first-time user must hand-pick up to 9 columns with no help and cannot proceed until they do.",
  "TC-17, TC-60, TC-74, TC-02","","","",""),
 ("BUG-04","UC1 churn_risk","Churn risk cannot answer the question it is named for","Critical","P0",
  "The app ignores the user's own outcome column (is_churned) and derives a _churn label from the Date column instead. For a one-row-per-customer file - the normal shape of a customer table - that derived label collapses to a single value and the run produces no answer. The bundled 'Customers Churn' sample, advertised for this exact goal, fails this way.",
  "The churn use case is unusable on its own demo data and on the standard customer-table shape. The user gets no triage list at all.",
  "TC-01, TC-04, TC-05","","","",""),
 ("BUG-05","UC3 / UC10","The user's margin policy floor cannot be set anywhere","Critical","P0",
  "There is no floor, threshold or policy input in the product (only 3 input elements exist: the file picker and two custom-column fields). The app silently substitutes its own rule - the 25th percentile of margin within the row's segment - and never says so on Steps 3, 4 or 6. The word 'policy' does not appear in the UI.",
  "A company with a hard 22% floor receives a relative answer that changes with the data, and cannot tell that its own rule was replaced.",
  "TC-16","","","",""),
 ("BUG-06","UC4","Silent feature drop contradicts an explicit on-screen promise","Critical","P0",
  "Step 4 says 'Leave this empty and we will use every other usable column in the file', but that fallback only fires when the drivers slot AND every other optional slot are empty. Filling any optional slot silently disables it. On the bundled file this dropped quantity, list_price, unit_price, net_price, discount_pct, cost and gross_margin_pct, giving R2 12% and a driver list naming weekday and month.",
  "The tool can return a confidently wrong answer about what drives revenue, with nothing on screen telling the user which columns were dropped.",
  "TC-24","","","",""),
 ("BUG-07","UC8","Planted anomalies are missed and the shortlist is worse than random","High","P1",
  "On 80,000 rows with 200 unambiguous planted anomalies, only 7 were surfaced (3.5%). Ghost-vendor rows (new vendor, round 9990, 3am Sunday) scored 0 of 100. Flag precision 0.175% against a 0.250% base rate = lift 0.70x. The app flags a fixed top 5% and the planted rows sit just outside it. Driver ranking collapsed to 'invoice_amount' at 98, i.e. it learned 'big amount = unusual'.",
  "An audit team cannot run a review off this: the shortlist is no better than a random sample and the planted fraud patterns are invisible.",
  "TC-49","","","",""),
 ("BUG-08","UC8","The bundled anomaly sample ships the answer key","High","P1",
  "Transactions Anomaly contains a column literally named _truth_anomaly (Yes/No). It is listed as an ordinary column and is offered as a Context column inside the flow, one dropdown away from being fed to the model.",
  "The whole premise of the use case is screening with no labels; the demo undermines it and can silently turn the unsupervised flow into a supervised one.",
  "TC-51","","","",""),
 ("BUG-09","UC7","The ranked list and export carry no account identifier","High","P1",
  "Neither the on-screen table nor the CSV export includes loan_id or customer_id (export columns: score, predicted, probability, actual, loan_amount, term_months, interest_rate, loan_purpose, home_ownership). Only the per-row 'why' cards name the account. Same defect on revenue_predict (no opportunity_id) and discount_leakage (no order ID).",
  "The goal is to rank accounts for a collections team, but the deliverable cannot be tied back to any account - so it cannot be actioned.",
  "TC-43, TC-27, TC-10","","","",""),
 ("BUG-10","UC7","An 'Optional' slot is hard-required by the engine","High","P1",
  "Step 4 labels the outcome slot 'Known outcome (optional)'. Mapping only the exposure column leaves Run enabled; the run then fails with Rows scored 0 and a blank answer screen. The real reason is buried in the Step 5 log: 'We need to know who actually defaulted.'",
  "A user who trusts the label and the enabled button hits a dead end with an empty answer and no visible cause.",
  "TC-44","","","",""),
 ("BUG-11","UC5","'Demand forecast' produces a backtest, not a forecast","High","P1",
  "The goal card promises 'Predict next month's demand from history', but Step 6 shows only a retrospective score and a history chart. There are no future dates, no predicted units, and no horizon control anywhere in the six steps ('future', 'horizon', 'next month' appear zero times on Step 6).",
  "A demand planner asking for next month's units per SKU-warehouse is answered with a different question and cannot get a forward number.",
  "TC-30, TC-29","","","",""),
 ("BUG-12","UC5","Planted seasonality and temperature signals are not surfaced","High","P1",
  "On data with a real Saturday = 2.66x Tuesday pattern and a +3%/degree effect above 70F, the driver list contains only lag and rolling features (lag_1..lag_21, 'average of last 7/28', 'wobble over last 7/28'). local_temperature_f and is_promotion never appear even when explicitly supplied. sku_id was mistyped as Category and dominated at 87.5%.",
  "The factors a planner actually controls - promotions and weather - are invisible, so the output cannot guide reorder points.",
  "TC-31, TC-32","","","",""),
 ("BUG-13","UC6","No elasticity curve, no per-category optimum, and the recommendation is wrong","High","P1",
  "Step 6 gives one global optimum as a sentence of prose. There is no revenue-versus-discount curve and no per-category breakdown (segments is empty in the payload). On data where Basics truly peaks at 30% and Outerwear is inelastic, the app recommends 60% - the deepest step in the grid - for every category.",
  "The advice is not just incomplete but actively harmful: following it would destroy revenue on inelastic categories.",
  "TC-36, TC-37, TC-38","","","",""),
 ("BUG-14","UC9","Segments are unnamed and the profile column is broken","High","P1",
  "Segments are labelled literally 'Group 1' to 'Group 4' with no business persona name, although the card promises to 'describe each group in words you can use in a meeting'. The PROFILE column renders as [object Object],[object Object] for every group - raw JavaScript objects stringified into the table.",
  "The clustering maths is correct (99.99% cluster recovery) but a marketer cannot act on anonymous groups with an unreadable profile.",
  "TC-57, TC-58","","","",""),
 ("BUG-15","UC10","The derived column cannot be the thing being predicted","High","P1",
  "The brief's custom formula is accepted and computed correctly, but the resulting flag is offered only as a Yes/No grouping category. The app always derives its own label ('Below policy floor'), so the user's own business rule is never what gets predicted.",
  "The headline feature of this use case - test your own formula - does not influence the answer, which makes the custom column decorative.",
  "TC-63","","","",""),
 ("BUG-16","UC10 / UC3","Label circularity inflates the headline metric","High","P1",
  "The derived label is a residual of the margin column, and that same margin column is also supplied to the model as a feature. Result: 'The strongest signal is gross_margin_pct (100% of the model's decision-making)' with importance 99.91, and ROC AUC 0.99998 / precision 0.999 on the freight file. The app acknowledges the leak in a note instead of removing it.",
  "A perfect-looking score that carries no information; a user would trust a meaningless result.",
  "TC-64, TC-19","","","",""),
 ("BUG-17","All","Time budget is exceeded and the model counts are inconsistent","Medium","P2",
  "The run budget is 75 s but is only checked between models, so runs overrun to 372 s / 424 s / 105 s / 214 s. The UI then reports contradictions on the same screen: 'Models tried: 11' vs 'came out on top of 8 models'; 'Models tried: 3' vs 'came out on top of 1 models'; and 'We have used 372 of the 75 seconds allowed'. At 1 GB a single model finishing is reported as 'agreement with the crowd 1.00'.",
  "Misleading progress and model counts undermine trust in the scoreboard, and there is no way for the user to raise the budget.",
  "TC-13, TC-76, TC-54","","","",""),
 ("BUG-18","All","Misleading impact-card labels","High","P1",
  "Several headline money labels do not describe the number underneath: 'Money the model gets right' holds the sum of absolute errors (the money the model gets wrong); 'Margin at risk' sums billed revenue rather than margin lost; 'Expected loss' is never reconciled to a PD x LGD x EAD calculation; and the anomaly goal has no false-positive figure at all.",
  "Users will quote these figures in meetings believing they mean something they do not.",
  "TC-25, TC-18, TC-41","","","",""),
 ("BUG-19","All","No validation of impossible values","High","P1",
  "list_price = 0 and discount_pct = 150 produced negative net prices that were then flagged as the top leakage rows and rolled into the headline money figure. Zero-cost rows produced a 'perfect' margin of 1.0. Negative revenue and zero revenue rows were accepted silently. No warning at any step.",
  "Impossible rows can dominate the answer and inflate the headline number with no signal to the user.",
  "TC-12, TC-20, TC-46","","","",""),
 ("BUG-20","All","Unexplained jargon on the results page","Medium","P2",
  "'Stratified Split' and 'Hold-Out' appear as the testing method with no tooltip and no glossary entry, and for a regression target 'stratified' is not even the correct term. Raw acronyms (ROC AUC, PR AUC, WAPE, MAPE) sit under plain-English labels without expansion. In segmentation, silhouette, davies bouldin and calinski appear with no gloss.",
  "A non-technical user cannot tell how the result was validated, which is exactly the trust question the copy is trying to answer.",
  "TC-73, TC-40","","","",""),
 ("BUG-21","All","Uploads are retained forever in data/","Medium","P2",
  "Every uploaded file leaves a full copy under data/ with a hash suffix and nothing ever removes it. In 30 minutes of testing one engineer accumulated 15 copies (about 3.4 GB) of four files, including three copies of a 758 MB file.",
  "Unbounded disk growth on the user's machine; a 1 GB upload costs 2 GB of disk.",
  "TC-69","","","",""),
 ("BUG-22","All","Reload, rail navigation and exports destroy user work","Medium","P2",
  "A browser reload at any step returns the user to Step 1 with no resume. Clicking a rail step during a run silently cancels the in-flight run with no confirmation. Export buttons use a full-page navigation to an attachment URL, which tears down the single-page app and loses all state.",
  "Users lose minutes of computation and their entire configuration to a single stray click or reload.",
  "TC-70, TC-71, TC-72","","","",""),
 ("BUG-23","All","Copy defects and leftover text","Low","P3",
  "'beating the previous leader (separation power 0.82) by by a wide margin' is repeated on every scoreboard row. 'That is the first result on the board' is printed for rank 6. Clustering screens carry supervised-goal copy ('Every method was judged on the same held-back data' - there is no held-out data in clustering). The champion is not reproducible across identical runs.",
  "Erodes polish and trust; the non-reproducible champion matters most because two runs on identical data returned different winners.",
  "TC-52, TC-73, TC-55","","","",""),
]
r = 2
for row in D:
    for c, v in enumerate(row, start=1):
        wd.cell(row=r, column=c, value=v)
    r += 1
D_LAST = r - 1

for c in range(1, len(d_cols) + 1):
    cell = wd.cell(row=1, column=c)
    cell.font = HDR_FONT; cell.fill = HDR_FILL; cell.alignment = CTR; cell.border = BORDER
for rr in range(2, D_LAST + 1):
    for c in range(1, len(d_cols) + 1):
        cell = wd.cell(row=rr, column=c)
        cell.border = BORDER
        cell.alignment = WRAP if c in (3, 6, 7, 9) else TOP
    for c in (9, 10, 11, 12):
        wd.cell(row=rr, column=c).fill = PatternFill("solid", fgColor=XL_CALC)
for k, v in {"A": 10, "B": 14, "C": 40, "D": 10, "E": 9, "F": 62, "G": 48, "H": 20, "I": 30, "J": 14, "K": 12, "L": 14}.items():
    wd.column_dimensions[k].width = v
wd.freeze_panes = "C2"
wd.auto_filter.ref = f"A1:L{D_LAST}"
wd.conditional_formatting.add(f"D2:D{D_LAST}",
    CellIsRule(operator="equal", formula=['"Critical"'], fill=PatternFill("solid", fgColor=XL_RED_BG), font=Font(color=XL_RED_TX, bold=True)))
wd.conditional_formatting.add(f"D2:D{D_LAST}",
    CellIsRule(operator="equal", formula=['"High"'], fill=PatternFill("solid", fgColor=XL_AMB_BG), font=Font(color=XL_AMB_TX, bold=True)))
dvs = DataValidation(type="list", formula1='"Open,In progress,Fixed,Deferred,Won\'t fix,Retest needed,Closed"', allow_blank=True)
dvd = DataValidation(type="list", formula1='"Critical,High,Medium,Low"', allow_blank=True)
dvt = DataValidation(type="list", formula1='"P0,P1,P2,P3"', allow_blank=True)
wd.add_data_validation(dvs); wd.add_data_validation(dvd); wd.add_data_validation(dvt)
dvs.add(f"J2:J{D_LAST}"); dvd.add(f"D2:D{D_LAST}"); dvt.add(f"E2:E{D_LAST}")

# =====================================================================
# 3. PERFORMANCE / STRESS
# =====================================================================
wp = wb.create_sheet("Performance")
p_cols = ["Use case", "Goal card", "Dataset", "Rows", "Size (MB)", "Ingest (s)", "Run (s)",
          "Outcome", "Run result", "Notes", "Developer comments", "Status"]
for c, name in enumerate(p_cols, start=1):
    wp.cell(row=1, column=c, value=name)

P = [
 ("UC1","Churn risk","Bundled Customers Churn",25000,1.7,0.075,1.2,"Fail","No answer","Label collapsed to one value; run short-circuits before models","",""),
 ("UC1","Churn risk","Synthetic churn",15000,2.0,0.148,0.8,"Fail","No answer","Same root cause as the bundled sample","",""),
 ("UC1","Churn risk","Synthetic churn (stress)",14000000,1027,5.1,11,"Fail","No answer","Ingest is excellent; failure is logical, not performance","",""),
 ("UC2","Discount leakage","Bundled Orders Discount Leakage",60042,11.3,0.126,79,"Pass","XGBoost 0.97","Only 7 of 10 models got a turn; the 75 s budget was exceeded","",""),
 ("UC2","Discount leakage","Synthetic discount",60000,1.5,0.13,61,"Pass","XGBoost 1.00","Rogue reps correctly flagged","",""),
 ("UC2","Discount leakage","Synthetic discount (stress)",13250000,1034,3.8,372,"Fail","IndexError","Unhandled exception; Rows scored 0","",""),
 ("UC3","Margin overflow","Bundled Orders Margin Overflow",40000,7.6,0.127,57,"Pass","LightGBM 1.00","Clean run","",""),
 ("UC3","Margin overflow","Synthetic margin",25000,2.1,0.024,18,"Pass","LightGBM 1.00","Clean run","",""),
 ("UC3","Margin overflow","Synthetic margin (stress)",12000000,1036.9,2.3,424,"Fail","IndexError","Sat at 9% for about 370 s before failing","",""),
 ("UC4","Revenue prediction","Bundled Orders Discount Leakage",60042,11.3,0.173,94,"Partial","Linear Regression R2 0.12","Numeric columns silently dropped","",""),
 ("UC4","Revenue prediction","Synthetic revenue",35000,2.2,0.06,70.8,"Pass","Gradient Boosting R2 0.26","Correct top-two drivers; outliers survived","",""),
 ("UC4","Revenue prediction","Synthetic revenue (stress)",15000000,975.6,9.4,480,"Fail","TIMEOUT","Still at 16% when stopped at the 480 s cap","",""),
 ("UC5","Demand forecast","Bundled Sales Demand Series",30702,1.6,0.087,69,"Partial","Neural Network R2 0.98","Backtest only; no forward forecast","",""),
 ("UC5","Demand forecast","Synthetic demand panel",75000,1.4,0.06,84,"Partial","Random Forest R2 0.77","Seasonality and temperature not surfaced","",""),
 ("UC5","Demand forecast","Synthetic demand (stress)",16000000,872.1,3.0,349,"Pass","Random Forest R2 0.69","Under the cap; disclosed its 400k training sample","",""),
 ("UC6","Price elasticity","Bundled Orders Discount Leakage",60042,11.3,0.101,51,"Partial","Neural Network R2 0.26","One global optimum, no curve","",""),
 ("UC6","Price elasticity","Synthetic price panel",40000,2.0,0.05,16,"Fail","Random Forest","Recommends 60% for every category - wrong","",""),
 ("UC6","Price elasticity","Synthetic price (stress)",12000000,789.7,1.4,175,"Fail","Random Forest R2 0.50","Same wrong recommendation at scale","",""),
 ("UC7","Credit default","Bundled Loan Credit Default",30000,2.9,0.131,31,"Pass","Expected loss 1.47bn","Two drivers tie at exactly 100/30%","",""),
 ("UC7","Credit default","Synthetic credit",50000,3.1,0.07,106,"Pass","Gradient Boosting 0.83","8.1x lift; planted cell 100% captured","",""),
 ("UC7","Credit default","Synthetic credit (stress)",15000000,961,5.6,192,"Pass","Drivers still ranked 1st/2nd","Models tried 7 vs 'came out on top of 2'","",""),
 ("UC8","Unusual records","Bundled Transactions Anomaly",50000,4.7,0.303,131,"Partial","Isolation Forest","Only 2,500 of 50,000 flagged; wrong-branch copy","",""),
 ("UC8","Unusual records","Synthetic AP disbursements",80000,5.3,0.08,58,"Fail","Isolation Forest","7 of 200 planted anomalies found","",""),
 ("UC8","Unusual records","Synthetic AP (stress)",14500000,999,5.2,458,"Partial","Isolation Forest","Only 1 model finished; 'agreement 1.00' for a consensus of one","",""),
 ("UC9","Customer segments","Bundled Customers Churn",25000,1.7,0.045,58.2,"Partial","Gaussian Mixture","4 groups found but unnamed; profile column broken","",""),
 ("UC9","Customer segments","Synthetic customer panel",30000,1.5,0.05,64.3,"Partial","Gaussian Mixture","100% cluster recovery; labels still 'Group 1..4'","",""),
 ("UC9","Customer segments","Synthetic customer panel (stress)",8000000,451.6,1.8,229.2,"Pass","Gaussian Mixture","99.99% cluster recovery","",""),
 ("UC10","Freight audit","Bundled Orders Margin Overflow",40000,7.6,0.042,70.5,"Pass","LightGBM 0.9998","Circular metric","",""),
 ("UC10","Freight audit","Synthetic freight shipments",45000,2.4,0.061,41.2,"Partial","XGBoost 1.00","SPOT_BROKER_X + Reefer flagged at 95.1% but only as a footnote","",""),
 ("UC10","Freight audit","Synthetic freight (stress)",9000000,758.2,2.3,181.5,"Fail","IndexError","Rows scored 0, no scored file written","",""),
]
r = 2
for row in P:
    for c, v in enumerate(row, start=1):
        wp.cell(row=r, column=c, value=v)
    r += 1
P_LAST = r - 1
for c in range(1, len(p_cols) + 1):
    cell = wp.cell(row=1, column=c)
    cell.font = HDR_FONT; cell.fill = HDR_FILL; cell.alignment = CTR; cell.border = BORDER
for rr in range(2, P_LAST + 1):
    for c in range(1, len(p_cols) + 1):
        cell = wp.cell(row=rr, column=c)
        cell.border = BORDER
        cell.alignment = WRAP if c in (10, 11) else TOP
    wp.cell(row=rr, column=5).number_format = "#,##0.0"
    wp.cell(row=rr, column=4).number_format = "#,##0"
    wp.cell(row=rr, column=6).number_format = "#,##0.000"
    wp.cell(row=rr, column=7).number_format = "#,##0.0"
    for c in (11, 12):
        wp.cell(row=rr, column=c).fill = PatternFill("solid", fgColor=XL_CALC)
for k, v in {"A": 8, "B": 18, "C": 32, "D": 12, "E": 11, "F": 11, "G": 10, "H": 10, "I": 40, "J": 50, "K": 30, "L": 14}.items():
    wp.column_dimensions[k].width = v
wp.freeze_panes = "C2"
wp.auto_filter.ref = f"A1:L{P_LAST}"
wp.conditional_formatting.add(f"H2:H{P_LAST}",
    CellIsRule(operator="equal", formula=['"Fail"'], fill=PatternFill("solid", fgColor=XL_RED_BG), font=Font(color=XL_RED_TX, bold=True)))
wp.conditional_formatting.add(f"H2:H{P_LAST}",
    CellIsRule(operator="equal", formula=['"Partial"'], fill=PatternFill("solid", fgColor=XL_AMB_BG), font=Font(color=XL_AMB_TX, bold=True)))
wp.conditional_formatting.add(f"H2:H{P_LAST}",
    CellIsRule(operator="equal", formula=['"Pass"'], fill=PatternFill("solid", fgColor=XL_GRN_BG), font=Font(color=XL_GRN_TX, bold=True)))
dvp = DataValidation(type="list", formula1='"Pass,Fail,Partial"', allow_blank=True)
wp.add_data_validation(dvp); dvp.add(f"H2:H{P_LAST}")
dvps = DataValidation(type="list", formula1='"Open,In progress,Fixed,Deferred,Won\'t fix,Retest needed,Closed"', allow_blank=True)
wp.add_data_validation(dvps); dvps.add(f"L2:L{P_LAST}")

# =====================================================================
# 4. UX AND JARGON
# =====================================================================
wu = wb.create_sheet("UX_Jargon")
u_cols = ["Ref", "Category", "Use case", "Finding", "Severity", "Suggested fix", "Developer comments", "Status"]
for c, name in enumerate(u_cols, start=1):
    wu.cell(row=1, column=c, value=name)

U = [
 ("UX-01","Workflow","All","Step 4 does not auto-map on most goals. Every slot opens empty and Run stays disabled, even for the app's own bundled samples whose columns match exactly.","Critical","Implement column-name alias matching against the input_roles aliases already declared in the goal definitions, and pre-fill with a confidence indicator plus a one-click 'clear all'.","",""),
 ("UX-02","Workflow","All","A failed run looks identical to a successful one: 'finished in Xs. Everything below was computed on this computer.' with Rows scored: 0 and no banner.","Critical","Render a distinct error state with the plain-English cause, a retry action and the failed step. Never show the success header when status != ok.","",""),
 ("UX-03","Workflow","All","'7 more preparation steps are listed in the log below' - but there is no log below.","High","Either render the referenced log inline or remove the pointer and link to the appendix that exists.","",""),
 ("UX-04","Workflow","All","A browser reload at any step loses everything and returns the user to Step 1 with no warning.","Medium","Persist the wizard state (data reference, goal, mapping, assumptions) to local storage and offer 'resume where you left off'.","",""),
 ("UX-05","Workflow","All","Clicking a rail step during a run silently cancels the run - no confirmation, no undo.","Medium","Show a confirm dialog ('A run is in progress. Leave and cancel it?') and keep the completed result cached if the user returns.","",""),
 ("UX-06","Workflow","All","Export buttons navigate the whole page to an attachment URL, destroying the app and all state.","Low","Trigger the download via a hidden iframe or a fetch + blob download so the SPA stays mounted.","",""),
 ("UX-07","Workflow","All","No way to raise the 75 s run budget, and the budget is only checked between models so runs overrun to 372 s / 424 s.","Medium","Add a visible time-budget control with a sane default, and check the budget inside the model loop.","",""),
 ("UX-08","Workflow","UC4 / UC7 / UC8","Impact-card labels misdescribe the number: 'Money the model gets right' is the sum of errors; 'Margin at risk' sums revenue not margin.","High","Rename to what the number is, and show the formula underneath in plain words.","",""),
 ("UX-09","Workflow","UC4","'The rows to look at first' is meaningless for regression - it sorts by largest prediction, not by error.","High","For regression, sort by absolute residual and label the column 'how wrong we were'.","",""),
 ("UX-10","Workflow","UC9","No control over the number of segments, although the app picks k by itself and explains the choice.","Medium","Expose a k slider that defaults to the automatic choice and shows the tightness score per k.","",""),
 ("UX-11","Workflow","UC9","Clustering screens carry supervised-goal copy: 'Every method was judged on the same held-back data' - there is no held-out data in clustering.","Medium","Branch the explanatory copy by task type (classification / regression / clustering / unsupervised).","",""),
 ("UX-12","Workflow","UC8","No false-positive estimate and no way to move the fixed top-5% flag cut-off.","High","Show an estimated false-alarm rate and expose the cut-off as a slider, mirroring the credit goal's precision/recall trade-off copy.","",""),
 ("UX-13","Workflow","UC5","Partial auto-map only: the Date slot fills but 'What to forecast' stays empty even though the slot has three candidates.","Medium","Extend the alias matcher to the target slot and offer a 'pick for me' button beside it.","",""),
 ("UX-14","Presentation","UC8","The PROFILE column in 'the groups we found' renders raw JavaScript objects: [object Object],[object Object].","Medium","Join the profile array into readable text before rendering.","",""),
 ("UX-15","Presentation","UC9","Segments are named 'Group 1'..'Group 4' with no business persona name, though the card promises meeting-ready descriptions.","High","Generate a persona label from each group's top distinguishing features (e.g. 'Frequent in-store browsers').","",""),
 ("UX-16","Presentation","UC4 / UC7 / UC10","The triage table and the CSV export omit the row identifier, so no flagged row can be traced back.","High","Always carry the identifier column through to the on-screen table and every export.","",""),
 ("UX-17","Presentation","UC8","The score scale flips between blocks on the same screen: the table shows 100 while the explanation card shows 0.17 for the same row.","Medium","Use one scale everywhere and state it.","",""),
 ("UX-18","Presentation","UC7 / UC10","Two different denominators appear on one screen: test-split metrics next to whole-file counts, unlabelled.","Medium","Label each figure 'on the held-back sample' or 'across your whole file'.","",""),
 ("UX-19","Presentation","All","Model counts contradict themselves: 'Models tried: 11' vs 'came out on top of 8 models'.","Medium","Report 'ran' and 'scored' separately and use one number in the champion sentence.","",""),
 ("UX-20","Presentation","All","Copy defects: 'by by a wide margin' on every scoreboard row; 'That is the first result on the board' printed for rank 6.","Low","Proofread the scoreboard strings and branch them by rank.","",""),
 ("UX-21","Presentation","UC1","The champion is not reproducible: two runs over identical rows and dates returned Extra Trees 0.9036 and Logistic Regression 0.98.","Medium","Seed the model selection deterministically, or state plainly that the winner can vary between runs.","",""),
 ("UX-22","Jargon","All","'Stratified Split' and 'Hold-Out' are shown as the testing method with no tooltip and no glossary entry. For regression 'stratified' is the wrong word.","Medium","Add hover help for the method label and use 'Time Split' for time-ordered data.","",""),
 ("UX-23","Jargon","All","Raw acronyms (ROC AUC, PR AUC, WAPE, MAPE, log loss) sit under the plain-English labels without expansion.","Low","Expand on first use or add the acronym to the existing hover text.","",""),
 ("UX-24","Jargon","UC9","silhouette, davies bouldin and calinski are shown as bare metric names with no gloss.","Medium","Add the same hover help the other metric cards have.","",""),
 ("UX-25","Jargon","UC6","The app labels a step ELASTICITY but never defines the word; 'this sweep' and 'your typical price' are also undefined.","Medium","Explain elasticity in one plain sentence and say what 'typical' means (mean or median).","",""),
 ("UX-26","Jargon","UC5","Driver names are lag_1, lag_3, 'average of last 7', 'wobble over last 28' - none are defined.","High","Rename to 'same day last week', '7-day average', '7-day volatility' and add hover definitions.","",""),
 ("UX-27","Jargon","UC10","The goal speaks retail language ('pricing policy', 'segment', 'channel'). The words accessorial, linehaul, detention and fuel surcharge never appear.","Medium","Add a freight/3PL vocabulary variant so a logistics analyst recognises the question.","",""),
 ("UX-28","Jargon","UC3 / UC10","The app substitutes its own margin rule without saying so, and the user's 22% floor cannot be entered anywhere.","Critical","Add an explicit policy-floor input and state on the result page which floor was applied.","",""),
 ("UX-29","Jargon","UC7","No PD, LGD or 90+ DPD definition appears on Step 6. The 90-day convention is stated only once, in a Step 4 slot tooltip.","High","Surface the 90+ DPD convention and the expected-loss definition on the answer page.","",""),
 ("UX-30","Jargon","UC8","'contamination' appears on a card (value 0.05) and is never explained, yet it is the single knob that decides the answer.","High","Explain it as 'share of records flagged as unusual' and make it adjustable.","",""),
 ("UX-31","Data quality","All","Impossible values are accepted silently: list_price 0 and discount_pct 150 produced negative net prices that topped the leakage list; zero-cost rows produced a 1.0 margin.","High","Add pre-flight validation with a visible count of suspect rows and let the user exclude them.","",""),
 ("UX-32","Data quality","UC8","The bundled Transactions Anomaly sample ships a _truth_anomaly column - the answer key - and offers it as a Context column.","High","Strip label columns from demo files and warn when a column looks like a label.","",""),
 ("UX-33","Data quality","All","Missing values are handled but the app never names which column had gaps or what it did.","Medium","Name the affected columns and state the fill rule in plain words on Step 2 and in the run log.","",""),
 ("UX-34","Data quality","UC5","sku_id (values like SKU-BEV-001) was typed as Category rather than an identifier, and then dominated the model.","Medium","Detect high-cardinality string columns with a consistent prefix as identifiers.","",""),
 ("UX-35","Data quality","UC2","discount_pct is a fraction in the sample (0.18) but is typed and labelled 'Percent', so a user reads it as 0.18%.","Medium","Detect 0-1 scaled percent columns and label them 'fraction (0-1)' or rescale on import.","",""),
 ("UX-36","Data quality","UC3 / UC10","The user's own derived flag cannot be used as the outcome, and the auto-derived label is fed back in as a feature (circular metric).","High","Allow a derived column to be selected as the outcome and always exclude the label basis from the feature set.","",""),
 ("UX-37","Data quality","All","Every upload leaves a full copy in data/ forever - 3.4 GB accumulated from four files in 30 minutes.","Medium","Delete or expire uploaded copies after the run, or warn the user and show a cleanup control.","",""),
 ("UX-38","Data quality","All","Duplicate rows are reported but never deduplicated or excluded (42 exact duplicates in the bundled file).","Low","Offer a 'remove duplicates' action or state explicitly that duplicates were kept.","",""),
]
r = 2
for row in U:
    for c, v in enumerate(row, start=1):
        wu.cell(row=r, column=c, value=v)
    r += 1
U_LAST = r - 1
for c in range(1, len(u_cols) + 1):
    cell = wu.cell(row=1, column=c)
    cell.font = HDR_FONT; cell.fill = HDR_FILL; cell.alignment = CTR; cell.border = BORDER
for rr in range(2, U_LAST + 1):
    for c in range(1, len(u_cols) + 1):
        cell = wu.cell(row=rr, column=c)
        cell.border = BORDER
        cell.alignment = WRAP if c in (4, 6, 7) else TOP
    for c in (7, 8):
        wu.cell(row=rr, column=c).fill = PatternFill("solid", fgColor=XL_CALC)
for k, v in {"A": 9, "B": 13, "C": 14, "D": 72, "E": 10, "F": 52, "G": 30, "H": 14}.items():
    wu.column_dimensions[k].width = v
wu.freeze_panes = "D2"
wu.auto_filter.ref = f"A1:H{U_LAST}"
wu.conditional_formatting.add(f"E2:E{U_LAST}",
    CellIsRule(operator="equal", formula=['"Critical"'], fill=PatternFill("solid", fgColor=XL_RED_BG), font=Font(color=XL_RED_TX, bold=True)))
wu.conditional_formatting.add(f"E2:E{U_LAST}",
    CellIsRule(operator="equal", formula=['"High"'], fill=PatternFill("solid", fgColor=XL_AMB_BG), font=Font(color=XL_AMB_TX, bold=True)))
dvu = DataValidation(type="list", formula1='"Open,In progress,Fixed,Deferred,Won\'t fix,Retest needed,Closed"', allow_blank=True)
wu.add_data_validation(dvu); dvu.add(f"H2:H{U_LAST}")

# =====================================================================
# 5. SUMMARY  (dashboard, formulas reference the other sheets)
# =====================================================================
wsm = wb.create_sheet("Summary", 0)
wsm.column_dimensions["A"].width = 34
for col in "BCDE":
    wsm.column_dimensions[col].width = 20
wsm.column_dimensions["F"].width = 60

wsm["A1"] = "ML Arena - End-to-End Test Deliverable"
wsm["A1"].font = TITLE_FONT
wsm["A2"] = "Simulation Arena / ML Arena v1.0.0  ·  http://127.0.0.1:8765/  ·  end-user UX + scale test"
wsm["A2"].font = Font(italic=True, size=10, color=XL_SUM)

meta = [
 ("Test lead", "WorkBuddy test engineering team lead"),
 ("Test engineers", "5 engineers (E1-E5), each driving a real Chrome browser as an end user"),
 ("Method", "UI walkthrough of the 6-step flow only. No API testing, no source-code review, no README."),
 ("Scenarios", "10 business use cases; 9 map to a native goal card, 1 (freight audit) maps to Margin overflow + a custom derived column"),
 ("Synthetic data", "Per-use-case synthetic datasets generated with seed 42, scaled to 8M-16M rows / 450 MB-1 GB for stress"),
 ("Test dates", "19-20 September 2026"),
 ("Total test cases", None),
 ("Total defects", None),
]
r = 4
for label, val in meta:
    wsm.cell(row=r, column=1, value=label).font = Font(bold=True, size=10)
    if label == "Total test cases":
        wsm.cell(row=r, column=2, value=f"=COUNTA('Test Cases'!A{TC_FIRST}:A{TC_LAST})")
    elif label == "Total defects":
        wsm.cell(row=r, column=2, value=f"=COUNTA(Defects!A2:A{D_LAST})")
    else:
        wsm.cell(row=r, column=2, value=val)
    wsm.cell(row=r, column=2).alignment = WRAP
    wsm.merge_cells(start_row=r, start_column=2, end_row=r, end_column=6)
    r += 1

r += 1
wsm.cell(row=r, column=1, value="VERDICT TALLY").font = Font(bold=True, size=12, color=XL_SUM)
r += 1
head = ["Verdict", "Test cases", "Share"]
for c, h in enumerate(head, start=1):
    cell = wsm.cell(row=r, column=c, value=h)
    cell.font = HDR_FONT; cell.fill = HDR_FILL; cell.alignment = CTR; cell.border = BORDER
VTALLY_HDR = r
r += 1
VTALLY_FIRST = r
for v in ["Pass", "Partial", "Fail"]:
    wsm.cell(row=r, column=1, value=v)
    wsm.cell(row=r, column=2, value=f"=COUNTIF('Test Cases'!$H${TC_FIRST}:$H${TC_LAST},$A{r})")
    wsm.cell(row=r, column=3, value=f"=IF(OR($B${VTALLY_HDR+4}=0,$B${VTALLY_HDR+4}=\"\"),\"\",B{r}/$B${VTALLY_HDR+4})")
    wsm.cell(row=r, column=3).number_format = "0.0%"
    for c in range(1, 4):
        wsm.cell(row=r, column=c).border = BORDER
    r += 1
VTALLY_LAST = r - 1
wsm.cell(row=r, column=1, value="Total").font = Font(bold=True)
wsm.cell(row=r, column=2, value=f"=SUM(B{VTALLY_FIRST}:B{VTALLY_LAST})").font = Font(bold=True)
wsm.cell(row=r, column=3, value=f"=IF(OR(B{r}=0,B{r}=\"\"),\"\",B{r}/B{r})").number_format = "0.0%"
for c in range(1, 4):
    cell = wsm.cell(row=r, column=c)
    cell.border = Border(left=thin, right=thin, top=thin, bottom=thin)
    cell.fill = PatternFill("solid", fgColor=XL_SOFT)
    cell.font = Font(bold=True)
VTALLY_TOTAL = r
wsm.conditional_formatting.add(f"A{VTALLY_FIRST}:A{VTALLY_FIRST}",
    CellIsRule(operator="equal", formula=['"Pass"'], fill=PatternFill("solid", fgColor=XL_GRN_BG)))
wsm.conditional_formatting.add(f"A{VTALLY_FIRST+1}:A{VTALLY_FIRST+1}",
    CellIsRule(operator="equal", formula=['"Partial"'], fill=PatternFill("solid", fgColor=XL_AMB_BG)))
wsm.conditional_formatting.add(f"A{VTALLY_FIRST+2}:A{VTALLY_FIRST+2}",
    CellIsRule(operator="equal", formula=['"Fail"'], fill=PatternFill("solid", fgColor=XL_RED_BG)))
r += 2

wsm.cell(row=r, column=1, value="DEFECTS BY SEVERITY").font = Font(bold=True, size=12, color=XL_SUM)
r += 1
for c, h in enumerate(["Severity", "Open defects"], start=1):
    cell = wsm.cell(row=r, column=c, value=h)
    cell.font = HDR_FONT; cell.fill = HDR_FILL; cell.alignment = CTR; cell.border = BORDER
r += 1
SEV_FIRST = r
for s in ["Critical", "High", "Medium", "Low"]:
    wsm.cell(row=r, column=1, value=s)
    wsm.cell(row=r, column=2, value=f"=COUNTIF(Defects!$D$2:$D${D_LAST},$A{r})")
    for c in range(1, 3):
        wsm.cell(row=r, column=c).border = BORDER
    r += 1
SEV_LAST = r - 1
wsm.cell(row=r, column=1, value="Total").font = Font(bold=True)
wsm.cell(row=r, column=2, value=f"=SUM(B{SEV_FIRST}:B{SEV_LAST})").font = Font(bold=True)
for c in range(1, 3):
    wsm.cell(row=r, column=c).fill = PatternFill("solid", fgColor=XL_SOFT)
    wsm.cell(row=r, column=c).border = BORDER
r += 2

wsm.cell(row=r, column=1, value="STRESS TEST HEADLINES").font = Font(bold=True, size=12, color=XL_SUM)
r += 1
for c, h in enumerate(["Measure", "Result"], start=1):
    cell = wsm.cell(row=r, column=c, value=h)
    cell.font = HDR_FONT; cell.fill = HDR_FILL; cell.alignment = CTR; cell.border = BORDER
wsm.merge_cells(start_row=r, start_column=2, end_row=r, end_column=6)
r += 1
stress = [
 ("Largest dataset ingested", "16,000,000 rows / 872 MB - ingest 3.0 s (columnar, never the bottleneck)"),
 ("Fastest 1 GB ingest", "250 MB in 1.3 s; 1 GB in 2.3-9.4 s"),
 ("Fastest successful 1 GB run", "175 s (price elasticity, 12 M rows)"),
 ("Slowest successful run", "458 s (anomaly, 14.5 M rows) - 19 s inside the 480 s cap"),
 ("Runs that crashed at scale", "4 goals: discount leakage (372 s), margin overflow (424 s), freight (181 s), anomaly contention case"),
 ("Runs that timed out", "1 goal: revenue prediction, stuck at 16% at the 480 s cap"),
 ("Verdict on scale", "Ingest is excellent. Model runs do not survive 1 GB: 5 of 10 use cases fail at scale."),
]
for k, v in stress:
    wsm.cell(row=r, column=1, value=k).font = Font(bold=True, size=10)
    wsm.cell(row=r, column=2, value=v).alignment = WRAP
    wsm.merge_cells(start_row=r, start_column=2, end_row=r, end_column=6)
    r += 1
r += 1

wsm.cell(row=r, column=1, value="SCOPE NOTE").font = Font(bold=True, size=12, color=XL_SUM)
r += 1
note = ("All ten briefed use cases were reachable through the UI: nine map directly to a goal card and the tenth "
        "(freight accessorial audit) runs on Margin overflow with a custom derived column. No use case was missing "
        "from the product. The dominant theme is not missing features but broken trust at the edges: silent failures "
        "that look like successes, labels that misdescribe their numbers, missing row identifiers, and no auto-mapping.")
wsm.cell(row=r, column=1, value=note).alignment = WRAP
wsm.merge_cells(start_row=r, start_column=1, end_row=r+3, end_column=6)

# =====================================================================
# 6. SIGN-OFF
# =====================================================================
wsg = wb.create_sheet("Sign-off")
wsg.column_dimensions["A"].width = 22
wsg.column_dimensions["B"].width = 24
wsg.column_dimensions["C"].width = 22
wsg.column_dimensions["D"].width = 16
wsg.column_dimensions["E"].width = 16
wsg.column_dimensions["F"].width = 56
wsg["A1"] = "Status submission and sign-off"
wsg["A1"].font = TITLE_FONT
wsg["A2"] = "Test cycle: ML Arena v1.0.0 end-to-end test, 19-20 September 2026. Filled in by the development team."
wsg["A2"].font = Font(italic=True, size=10, color=XL_SUM)

r = 4
for c, h in enumerate(["Role", "Name", "Decision", "Date", "Status", "Comments"], start=1):
    cell = wsg.cell(row=r, column=c, value=h)
    cell.font = HDR_FONT; cell.fill = HDR_FILL; cell.alignment = CTR; cell.border = BORDER
r += 1
roles = [
 ("Test lead", "", "Submitted for triage", date(2026, 9, 20), "Submitted", "60 test cases across 10 use cases; 23 defects raised (6 Critical, 12 High). All synthetic data deleted after the run."),
 ("Dev lead", "", "", None, "Pending", ""),
 ("Dev - engine", "", "", None, "Pending", ""),
 ("Dev - UI/UX", "", "", None, "Pending", ""),
 ("Product owner", "", "", None, "Pending", ""),
 ("QA sign-off", "", "", None, "Pending", ""),
]
SIGN_FIRST = r
for row in roles:
    for c, v in enumerate(row, start=1):
        wsg.cell(row=r, column=c, value=v)
    r += 1
SIGN_LAST = r - 1
for rr in range(SIGN_FIRST, SIGN_LAST + 1):
    for c in range(1, 7):
        cell = wsg.cell(row=rr, column=c)
        cell.border = BORDER
        cell.alignment = WRAP if c == 6 else TOP
        if c in (2, 3, 4, 5, 6) and rr > SIGN_FIRST:
            cell.fill = PatternFill("solid", fgColor=XL_CALC)
    if wsg.cell(row=rr, column=4).value is not None:
        wsg.cell(row=rr, column=4).number_format = "YYYY-MM-DD"
dvdec = DataValidation(type="list", formula1='"Submitted,Approved,Approved with conditions,Rejected,Pending"', allow_blank=True)
dvst = DataValidation(type="list", formula1='"Submitted,Pending,In review,Accepted,Rejected"', allow_blank=True)
wsg.add_data_validation(dvdec); wsg.add_data_validation(dvst)
dvdec.add(f"C{SIGN_FIRST}:C{SIGN_LAST}")
dvst.add(f"E{SIGN_FIRST}:E{SIGN_LAST}")

r = SIGN_LAST + 2
wsg.cell(row=r, column=1, value="Release gate").font = Font(bold=True, size=12, color=XL_SUM)
r += 1
gate_start = r
gate = [
 ("Critical defects open", f"=COUNTIFS(Defects!$D$2:$D${D_LAST},\"Critical\",Defects!$J$2:$J${D_LAST},\"<>Fixed\")"),
 ("High defects open", f"=COUNTIFS(Defects!$D$2:$D${D_LAST},\"High\",Defects!$J$2:$J${D_LAST},\"<>Fixed\")"),
 ("Test cases failed", f"=COUNTIF('Test Cases'!$H${TC_FIRST}:$H${TC_LAST},\"Fail\")"),
 ("Release recommendation", f'=IF(B{gate_start}>0,"NOT READY - "&B{gate_start}&" Critical defects open","Ready for review")'),
]
for k, f in gate:
    wsg.cell(row=r, column=1, value=k).font = Font(bold=True, size=10)
    cell = wsg.cell(row=r, column=2, value=f)
    cell.alignment = WRAP
    cell.border = BORDER
    r += 1

# ---------------------------------------------------------------------
wb.save(OUT)
print("saved:", OUT)
print("sheets:", wb.sheetnames)
print("test cases:", TC_LAST - TC_HDR_ROW, "defects:", D_LAST - 1, "ux:", U_LAST - 1, "perf:", P_LAST - 1)
