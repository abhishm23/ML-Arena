"""ML Arena — the business-goal catalogue.

Nine business questions a non-technical user can actually ask of their own data.
Everything in here is written for a shop owner, not a data scientist: the `story`
field is what the user reads on the card, the `tagline` is what they scan, and the
`roles` are the pickers we put in front of them.

There is **no language model anywhere in ML Arena**. Every sentence in this file was
written by a human and is served verbatim. That is a feature: the product must give
the same, reviewable answer on every machine, offline, forever.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Role / UseCase
# --------------------------------------------------------------------------- #


@dataclass
class Role:
    """One column-picker the UI shows on the mapping step."""

    key: str
    label: str
    help: str
    accepts: list[str]          # column kinds the picker is allowed to offer
    required: bool
    multi: bool
    min_count: int = 1


@dataclass
class UseCase:
    key: str
    name: str
    emoji: str
    tagline: str                       # <= 8 words
    story: str                         # 2-3 layman sentences
    business_question: str
    task: str                          # classification|regression|anomaly|clustering
    roles: list[Role]
    target: str                        # strategy key in definitions.py
    primary_metric: str                # roc_auc|r2|silhouette|consensus
    primary_metric_label: str          # plain-English name of that metric
    impact_unit: str                   # currency|records|customers|none
    impact_label: str
    sector: str
    examples: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Shared role factories (keeps the nine definitions readable)
# --------------------------------------------------------------------------- #

_MEASURE = ["measure"]
_DIM = ["dimension"]
_ID_OR_DIM = ["identifier", "dimension"]
_DIM_OR_ID = ["dimension", "identifier"]


def _r_customer(multi: bool = True, required: bool = False) -> Role:
    return Role(
        key="customer",
        label="Customer / account",
        help="Who the row belongs to. Give us this and we can tell you which "
             "customers behave differently from the rest.",
        accepts=_ID_OR_DIM,
        required=required,
        multi=multi,
    )


def _r_segment() -> Role:
    return Role(
        key="segment",
        label="Customer group / segment",
        help="Any grouping you already use — tier, region, industry. We compare "
             "like with like instead of judging every row on its own.",
        accepts=_DIM,
        required=False,
        multi=True,
    )


def _r_product() -> Role:
    return Role(
        key="product",
        label="Product / service",
        help="What was sold. Useful because some products naturally carry bigger "
             "discounts or thinner margins than others.",
        accepts=_DIM_OR_ID,
        required=False,
        multi=True,
    )


def _r_channel() -> Role:
    return Role(
        key="channel",
        label="Channel / store / region",
        help="Where the sale happened — online, a branch, a salesperson. Different "
             "channels often follow different rules.",
        accepts=_DIM,
        required=False,
        multi=True,
    )


def _r_quantity() -> Role:
    return Role(
        key="quantity",
        label="Quantity / units",
        help="How many units were bought. Bigger orders usually earn bigger "
             "discounts, so this helps us judge what was normal for that order.",
        accepts=_MEASURE,
        required=False,
        multi=True,
    )


def _r_date(required: bool = False) -> Role:
    return Role(
        key="date",
        label="Date",
        help="When it happened. This lets us follow the trend over time instead of "
             "treating every day as identical.",
        accepts=["datetime"],
        required=required,
        multi=False,
    )


def _r_money(required: bool = True, key: str = "money") -> Role:
    return Role(
        key=key,
        label="Money column",
        help="The amount of money on the row — order value, revenue, invoice total. "
             "This is how we turn findings into a rupee/dollar figure you can act on.",
        accepts=_MEASURE,
        required=required,
        multi=False,
    )


# --------------------------------------------------------------------------- #
# The nine use cases
# --------------------------------------------------------------------------- #

_USECASES: list[UseCase] = [
    UseCase(
        key="discount_leakage",
        name="Discount leakage",
        emoji="🏷️",
        tagline="Find discounts far bigger than they should be",
        story=(
            "Every order gets a discount, and most of them make sense — the big "
            "customer, the end-of-season line, the bundle deal. The problem is the "
            "few orders where the discount is far larger than that situation ever "
            "justifies. We learn what a normal discount looks like for each kind of "
            "order, then show you the ones that blow past it and what they cost you."
        ),
        business_question="Which orders were discounted far more than they should have been, and how much did that cost?",
        task="classification",
        roles=[
            Role(
                key="discount",
                label="Discount column",
                help="The discount you want to police — either a percentage or a "
                     "rupee/dollar amount. If you have both, give us both.",
                accepts=_MEASURE,
                required=True,
                multi=True,
            ),
            _r_money(key="money"),
            _r_customer(),
            _r_segment(),
            _r_product(),
            _r_channel(),
            _r_quantity(),
            _r_date(),
        ],
        target="discount_leakage",
        primary_metric="roc_auc",
        primary_metric_label="Separation power",
        impact_unit="currency",
        impact_label="Discount value at risk",
        sector="Sales",
        examples=["orders_with_discounts.csv", "sales_lines_2024.xlsx", "quote_register.csv"],
    ),
    UseCase(
        key="margin_overflow",
        name="Margin overflow",
        emoji="📉",
        tagline="Catch orders that sold below your pricing rule",
        story=(
            "Your pricing policy says what margin a deal of this size, in this "
            "segment, through this channel should make. Some orders quietly land far "
            "below that. We work out the margin that was reasonable for each order, "
            "then flag the ones that came in under the floor — so you can see whether "
            "it is one rogue region or a habit across the business."
        ),
        business_question="Which orders made much less margin than their segment and size should have earned?",
        task="classification",
        roles=[
            Role(
                key="margin",
                label="Margin column",
                help="Your margin — a percentage (like 22 or 0.22) or a money amount. "
                     "This is the number we watch.",
                accepts=_MEASURE,
                required=True,
                multi=True,
            ),
            _r_money(key="money"),
            Role(
                key="cost",
                label="Cost column (optional)",
                help="If you give us cost as well as revenue we can check the margin "
                     "yourself calculated, and tell you if it disagrees.",
                accepts=_MEASURE,
                required=False,
                multi=False,
            ),
            _r_segment(),
            _r_product(),
            _r_channel(),
            _r_customer(),
            _r_quantity(),
            _r_date(),
        ],
        target="margin_overflow",
        primary_metric="roc_auc",
        primary_metric_label="Separation power",
        impact_unit="currency",
        impact_label="Margin at risk",
        sector="Finance",
        examples=["order_margins.csv", "product_pnl_2024.xlsx", "deal_register.csv"],
    ),
    UseCase(
        key="revenue_predict",
        name="What drives revenue",
        emoji="💰",
        tagline="Learn what really moves your money",
        story=(
            "You have a column of money you care about — order value, monthly "
            "revenue, spend per customer. We learn what in the rest of the data "
            "predicts it, then rank those factors by how much they matter. You end up "
            "with a short list of levers, not a black box."
        ),
        business_question="What actually drives this money number, and how much can we trust the pattern?",
        task="regression",
        roles=[
            Role(
                key="target",
                label="What to predict (money or number)",
                help="The column you want to explain — order value, revenue, units, "
                     "cost. Everything else you select below becomes evidence.",
                accepts=_MEASURE,
                required=True,
                multi=False,
            ),
            Role(
                key="drivers",
                label="Candidate drivers (optional)",
                help="The columns you suspect matter — price, quantity, region, "
                     "salesperson. Leave this empty and we will use every other "
                     "usable column in the file.",
                accepts=["measure", "dimension", "identifier"],
                required=False,
                multi=True,
            ),
            _r_product(),
            _r_segment(),
            _r_channel(),
            _r_customer(),
            _r_date(),
        ],
        target="revenue_predict",
        primary_metric="r2",
        primary_metric_label="Explained variation",
        impact_unit="currency",
        impact_label="Money the model gets right",
        sector="Sales",
        examples=["orders.csv", "monthly_revenue.xlsx", "customer_spend.csv"],
    ),
    UseCase(
        key="demand_forecast",
        name="Demand forecast",
        emoji="📈",
        tagline="Predict next month's demand from history",
        story=(
            "Sales bounce around: busy weekends, quiet Mondays, a spike every "
            "December. We read the pattern in your history and use it to predict the "
            "next stretch of time. We also compare our forecasts against the simple "
            "'same time last year' rule, so you can see whether the extra work buys "
            "you anything."
        ),
        business_question="What will demand be over the next period, and is that better than the simple seasonal rule?",
        task="regression",
        roles=[
            Role(
                key="target",
                label="What to forecast",
                help="The number you plan around — units sold, orders, revenue per day.",
                accepts=_MEASURE,
                required=True,
                multi=False,
            ),
            _r_date(required=True),
            Role(
                key="series",
                label="Split into separate series (optional)",
                help="If you pick a column like region or product, we forecast each "
                     "one separately, because they follow their own rhythms.",
                accepts=_DIM,
                required=False,
                multi=False,
            ),
            Role(
                key="drivers",
                label="Extra context (optional)",
                help="Any other columns that might shift demand — price, promotion, "
                     "weather. They get added to the forecast as extra evidence.",
                accepts=["measure", "dimension"],
                required=False,
                multi=True,
            ),
        ],
        target="demand_forecast",
        primary_metric="r2",
        primary_metric_label="Explained variation",
        impact_unit="records",
        impact_label="Units typically off the mark",
        sector="Operations",
        examples=["daily_sales.csv", "weekly_demand.xlsx", "store_traffic.csv"],
    ),
    UseCase(
        key="churn_risk",
        name="Churn risk",
        emoji="🚪",
        tagline="Spot customers about to stop buying",
        story=(
            "A customer who is about to leave usually looks a little different "
            "beforehand: they buy less often, spend less each time, and drift further "
            "from their last order. We score every customer on those habits and put "
            "the likeliest leavers at the top, with the revenue that walks out with "
            "them."
        ),
        business_question="Which customers are about to stop buying, and how much revenue is at stake?",
        task="classification",
        roles=[
            Role(
                key="customer",
                label="Customer ID",
                help="The column that identifies a customer. We group all their "
                     "purchases together and score them as one.",
                accepts=_ID_OR_DIM,
                required=True,
                multi=False,
            ),
            _r_date(required=True),
            Role(
                key="money",
                label="Spend / revenue column",
                help="What each purchase was worth. We use it to size the revenue at "
                     "risk for the customers we flag.",
                accepts=_MEASURE,
                required=True,
                multi=True,
            ),
            _r_quantity(),
            _r_product(),
            _r_segment(),
            _r_channel(),
        ],
        target="churn_risk",
        primary_metric="roc_auc",
        primary_metric_label="Separation power",
        impact_unit="currency",
        impact_label="Revenue at risk",
        sector="Customer",
        examples=["transactions.csv", "purchase_history.xlsx", "subscriptions.csv"],
    ),
    UseCase(
        key="credit_default",
        name="Late payment & default",
        emoji="⚠️",
        tagline="Rank the accounts most likely to go bad",
        story=(
            "Some invoices get paid late and some never get paid at all. We learn "
            "what the late payers had in common — the segment, the size, the history — "
            "and put every open account in order of risk. That turns a collections "
            "team's day into a list instead of a guess."
        ),
        business_question="Which accounts are most likely to default, and how much money is exposed?",
        task="classification",
        roles=[
            Role(
                key="outcome",
                label="Known outcome (optional)",
                help="If you already have a column saying who defaulted or paid late "
                     "(yes/no, or a status like 'Written off'), point us at it and we "
                     "will learn from it directly.",
                accepts=_DIM,
                required=False,
                multi=False,
            ),
            Role(
                key="status",
                label="Status column (optional)",
                help="A status column we can read instead — values like Current, "
                     "30 days, Late, Default. We treat the bad ones as the outcome.",
                accepts=_DIM,
                required=False,
                multi=False,
            ),
            Role(
                key="days_past_due",
                label="Days past due (optional)",
                help="If you track how many days late an account is, we treat more "
                     "than 90 days as a default.",
                accepts=_MEASURE,
                required=False,
                multi=False,
            ),
            Role(
                key="money",
                label="Exposure column",
                help="How much money is owed on each account. This is what we multiply "
                     "the risk by to get the expected loss.",
                accepts=_MEASURE,
                required=True,
                multi=False,
            ),
            _r_customer(multi=False),
            _r_segment(),
            _r_product(),
            Role(
                key="drivers",
                label="Other evidence (optional)",
                help="Anything else that might signal risk — tenor, interest rate, "
                     "utilisation, past behaviour.",
                accepts=["measure", "dimension"],
                required=False,
                multi=True,
            ),
            _r_date(),
        ],
        target="credit_default",
        primary_metric="roc_auc",
        primary_metric_label="Separation power",
        impact_unit="currency",
        impact_label="Expected loss",
        sector="Risk",
        examples=["loan_book.csv", "ar_aging.xlsx", "invoice_status.csv"],
    ),
    UseCase(
        key="transaction_anomaly",
        name="Unusual records",
        emoji="🔍",
        tagline="Surface the rows that do not fit",
        story=(
            "You do not need to know in advance what is wrong. We look at every "
            "record and ask which ones simply do not look like the others — the "
            "strange amount, the odd combination, the entry that broke the routine. "
            "It is a shortlist for a human to glance at, ordered by how strange each "
            "record is."
        ),
        business_question="Which records stand out as genuinely unusual and deserve a human look?",
        task="anomaly",
        roles=[
            Role(
                key="measures",
                label="Numbers to judge on",
                help="Two or more numeric columns that describe a record — amount, "
                     "quantity, days to pay, discount. The more angles you give us, "
                     "the better we can spot the odd one out.",
                accepts=_MEASURE,
                required=True,
                multi=True,
                min_count=2,
            ),
            Role(
                key="dimensions",
                label="Context columns (optional)",
                help="Categories like region or type. A record can be odd because of "
                     "the combination it sits in, not just its size.",
                accepts=_DIM,
                required=False,
                multi=True,
            ),
            Role(
                key="key",
                label="Reference column (optional)",
                help="An ID such as an invoice or order number, so you can find the "
                     "flagged record again in your own system.",
                accepts=["identifier"],
                required=False,
                multi=True,
            ),
            Role(
                key="money",
                label="Money column (optional)",
                help="If you tell us the value of each record, we can add up how much "
                     "money is tied up in the unusual ones.",
                accepts=_MEASURE,
                required=False,
                multi=False,
            ),
            _r_date(),
        ],
        target="transaction_anomaly",
        primary_metric="consensus",
        primary_metric_label="Agreement with the crowd",
        impact_unit="records",
        impact_label="Unusual records to review",
        sector="Risk",
        examples=["journal_entries.csv", "expense_claims.xlsx", "claims.csv"],
    ),
    UseCase(
        key="segmentation",
        name="Customer segments",
        emoji="🧩",
        tagline="Sort your customers into natural groups",
        story=(
            "Not every customer wants the same thing, but they rarely arrive in neat "
            "labelled boxes. We group them by how they actually behave — how often "
            "they buy, how much they spend, what they buy — and then describe each "
            "group in words you can use in a meeting."
        ),
        business_question="What natural groups exist in my customers, and how do I describe each one?",
        task="clustering",
        roles=[
            Role(
                key="customer",
                label="Customer ID (optional)",
                help="Give us this and we group each customer's rows together before "
                     "splitting them into segments.",
                accepts=_ID_OR_DIM,
                required=False,
                multi=False,
            ),
            Role(
                key="measures",
                label="Behaviour numbers",
                help="Two or more numbers that describe behaviour — spend, frequency, "
                     "quantity, recency. These are the dimensions we group on.",
                accepts=_MEASURE,
                required=True,
                multi=True,
                min_count=2,
            ),
            Role(
                key="dimensions",
                label="Extra attributes (optional)",
                help="Region, channel, product mix. They help describe the groups "
                     "afterwards and often explain why a group exists.",
                accepts=_DIM,
                required=False,
                multi=True,
            ),
            _r_date(),
        ],
        target="segmentation",
        primary_metric="silhouette",
        primary_metric_label="Group tightness",
        impact_unit="customers",
        impact_label="Customers sorted into groups",
        sector="Customer",
        examples=["customer_summary.csv", "crm_export.xlsx", "loyalty_data.csv"],
    ),
    UseCase(
        key="price_elasticity",
        name="Price & discount response",
        emoji="⚖️",
        tagline="See how demand reacts to price",
        story=(
            "Cutting the price usually sells more, but not always enough to make more "
            "money. We measure how much extra demand each price change actually "
            "produced in your history, then sketch the discount that earns the most "
            "overall. Sometimes the answer is that your current discount is already "
            "past the sweet spot."
        ),
        business_question="How much does demand respond to price, and which price makes the most money?",
        task="regression",
        roles=[
            Role(
                key="quantity",
                label="Demand column",
                help="What you sold — units, orders, subscriptions. This is the "
                     "number we expect to move when price moves.",
                accepts=_MEASURE,
                required=True,
                multi=False,
            ),
            Role(
                key="price",
                label="Price or discount column",
                help="The price paid, or the discount given. Either works — we just "
                     "need to know which, and the sign of the answer flips accordingly.",
                accepts=_MEASURE,
                required=True,
                multi=False,
            ),
            _r_product(),
            _r_segment(),
            _r_channel(),
            _r_customer(),
            _r_date(),
        ],
        target="price_elasticity",
        primary_metric="r2",
        primary_metric_label="Explained variation",
        impact_unit="currency",
        impact_label="Revenue upside from the best price",
        sector="Sales",
        examples=["price_history.csv", "promo_lift.xlsx", "sku_pricing.csv"],
    ),
]

_BY_KEY: dict[str, UseCase] = {uc.key: uc for uc in _USECASES}

TASK_LABELS = {
    "classification": "Sorting rows into groups",
    "regression": "Predicting a number",
    "anomaly": "Finding the odd ones out",
    "clustering": "Discovering natural groups",
}


# --------------------------------------------------------------------------- #
# Public API (contract §6)
# --------------------------------------------------------------------------- #


def all_usecases() -> list[UseCase]:
    """Every use case, in the fixed presentation order."""
    return list(_USECASES)


def get_usecase(key: str) -> UseCase:
    """Look up one use case. Raises KeyError with a helpful message."""
    try:
        return _BY_KEY[key]
    except KeyError:
        known = ", ".join(_BY_KEY)
        raise KeyError(
            f"Unknown use case {key!r}. Available use cases are: {known}."
        ) from None


def serialize(uc: UseCase) -> dict:
    """JSON-safe dict for the API/frontend (contract §6)."""
    return {
        "key": uc.key,
        "name": uc.name,
        "emoji": uc.emoji,
        "tagline": uc.tagline,
        "story": uc.story,
        "business_question": uc.business_question,
        "task": uc.task,
        "task_label": TASK_LABELS.get(uc.task, uc.task),
        "target": uc.target,
        "primary_metric": uc.primary_metric,
        "primary_metric_label": uc.primary_metric_label,
        "impact_unit": uc.impact_unit,
        "impact_label": uc.impact_label,
        "sector": uc.sector,
        "examples": list(uc.examples),
        "roles": [
            {
                "key": r.key,
                "label": r.label,
                "help": r.help,
                "accepts": list(r.accepts),
                "required": bool(r.required),
                "multi": bool(r.multi),
                "min_count": int(r.min_count),
            }
            for r in uc.roles
        ],
    }


def catalog() -> list[dict]:
    """All use cases serialized, grouped order preserved."""
    return [serialize(uc) for uc in _USECASES]


def by_sector() -> dict[str, list[dict]]:
    """Grouped view used by the use-case step of the UI."""
    groups: dict[str, list[dict]] = {}
    for uc in _USECASES:
        groups.setdefault(uc.sector, []).append(serialize(uc))
    return groups


def required_roles(uc: UseCase) -> list[Role]:
    return [r for r in uc.roles if r.required]


def role(uc: UseCase, key: str) -> Role | None:
    for r in uc.roles:
        if r.key == key:
            return r
    return None


__all__ = [
    "Role",
    "UseCase",
    "TASK_LABELS",
    "all_usecases",
    "get_usecase",
    "serialize",
    "catalog",
    "by_sector",
    "required_roles",
    "role",
]
