"""End-to-end integration test for the ML Arena API.

Exercises the real FastAPI app over HTTP and a real WebSocket: health, the
bundled samples, dataset profiling, the custom-column builder, a full
discount-leakage analysis run streamed live, and all four export formats.

    C:\\ML Arena\\.venv\\Scripts\\python.exe tests/test_api.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from mlarena.api.server import app  # noqa: E402

FAILURES: list[str] = []
CHECKS = 0


def check(label: str, condition: bool, extra: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label} {extra}")
        FAILURES.append(label)


def main() -> int:
    print("=" * 78)
    print("ML Arena — end-to-end API test")
    print("=" * 78)

    with TestClient(app) as client:
        # ---- health -------------------------------------------------------- #
        r = client.get("/api/health")
        check("GET /api/health -> 200", r.status_code == 200, r.text[:200])
        health = r.json()
        check("health reports app name", health.get("app") == "ML Arena")
        check("health reports cpu count", isinstance(health.get("cpu_count"), int))
        print(f"        python={health.get('python')} cores={health.get('cpu_count')} "
              f"mem={health.get('memory_gb')}GB gpu={health.get('gpu')}")

        # ---- use cases ----------------------------------------------------- #
        r = client.get("/api/usecases")
        check("GET /api/usecases -> 200", r.status_code == 200)
        uc_payload = r.json()
        usecases = uc_payload.get("usecases", [])
        check("9 use cases returned", len(usecases) == 9, f"got {len(usecases)}")
        check("function catalog present", len(uc_payload.get("functions", [])) > 5)
        check("operators present", len(uc_payload.get("operators", [])) > 0)
        for uc in usecases:
            if not all(k in uc for k in ("key", "name", "emoji", "tagline", "story", "task", "roles")):
                check(f"use case {uc.get('key')} has full shape", False, str(sorted(uc.keys())))
        print("        " + ", ".join(u["key"] for u in usecases))

        # ---- samples ------------------------------------------------------- #
        r = client.get("/api/samples")
        check("GET /api/samples -> 200", r.status_code == 200)
        samples = r.json().get("samples", [])
        check("bundled samples found", len(samples) >= 6, f"got {len(samples)}")
        for s in samples:
            print(f"        {s['name']:<34} {s['size_bytes']/1e6:7.2f} MB  -> {s['usecase']}")

        # ---- load a sample ------------------------------------------------- #
        target = "orders_discount_leakage.csv"
        check("flagship sample present", any(s["name"] == target for s in samples))
        r = client.post(f"/api/samples/{target}/load")
        check("POST /api/samples/{name}/load -> 200", r.status_code == 200, r.text[:300])
        payload = r.json()
        check("load returns top-level dataset_id", "dataset_id" in payload)
        ds = payload["dataset"]
        prof = payload["profile"]
        dataset_id = payload["dataset_id"]
        print(f"        loaded {ds['rows']:,} rows x {ds['cols']} cols in {ds['load_ms']} ms "
              f"via {ds['engine']}")
        print(f"        quality: {prof['quality']['headline']}")

        # ---- profiling intelligence ---------------------------------------- #
        kinds = {c["name"]: (c["kind"], c["semantic"]) for c in prof["columns"]}
        check("money column detected as measure", kinds.get("net_price", ("",))[0] == "measure",
              str(kinds.get("net_price")))
        check("segment detected as dimension", kinds.get("customer_segment", ("",))[0] == "dimension",
              str(kinds.get("customer_segment")))
        check("order_id detected as identifier", kinds.get("order_id", ("",))[0] == "identifier",
              str(kinds.get("order_id")))
        check("order_date detected as datetime", kinds.get("order_date", ("",))[0] == "datetime",
              str(kinds.get("order_date")))
        check("profile exposes measures list", "discount_pct" in prof["measures"], str(prof["measures"]))
        check("profile is JSON-safe", json.dumps(prof) is not None)

        # ---- column suggestions -------------------------------------------- #
        r = client.get(f"/api/datasets/{dataset_id}/suggestions")
        check("GET suggestions -> 200", r.status_code == 200)
        sugg = r.json().get("suggestions", [])
        check("derived-column suggestions produced", len(sugg) > 0, f"got {len(sugg)}")
        for s in sugg[:5]:
            print(f"        suggests {s['name']} = {s['expression']}")
        # The sample already ships discount_pct / gross_margin_pct, so the engine must
        # NOT re-suggest those names — it should offer the non-duplicate derivations.
        names = [s["name"] for s in sugg]
        check("does not re-suggest an existing column", "gross_margin_pct" not in names, str(names))
        cost_pair = [s for s in sugg if "revenue" in s["expression"] and "cost" in s["expression"]]
        check("revenue & cost derivation suggested", len(cost_pair) > 0, str(names))

        # ---- custom column: preview ---------------------------------------- #
        r = client.post(
            f"/api/datasets/{dataset_id}/columns/preview",
            json={"name": "discount_value", "expression": "discount_pct * revenue"},
        )
        check("POST columns/preview -> 200", r.status_code == 200, r.text[:300])
        prev = r.json()
        check("preview is ok", prev.get("ok") is True, str(prev.get("error"))[:200])
        check("preview returns values", len(prev.get("values") or []) > 0)
        check("preview returns stats", isinstance(prev.get("stats"), dict))

        # ---- custom column: apply ------------------------------------------ #
        r = client.post(
            f"/api/datasets/{dataset_id}/columns/apply",
            json={"columns": [{"name": "discount_value", "expression": "discount_pct * revenue"}]},
        )
        check("POST columns/apply -> 200", r.status_code == 200, r.text[:300])
        applied = r.json()
        new_id = applied["dataset_id"]
        check("apply returns a NEW dataset id", new_id != dataset_id)
        check("derived column present in new schema", "discount_value" in applied["dataset"]["columns"])
        check("original dataset unchanged",
              "discount_value" not in client.get(f"/api/datasets/{dataset_id}").json()["dataset"]["columns"])
        check("derived dataset records its origin",
              applied["dataset"].get("derived_from") == dataset_id)

        # ---- run + live websocket ------------------------------------------ #
        mapping = {
            "discount": ["discount_pct"],
            "money": ["revenue"],
            "customer": ["customer_id"],
            "segment": ["customer_segment"],
            "product": ["product_category"],
            "channel": ["channel"],
            "quantity": ["quantity"],
            "date": ["order_date"],
        }
        r = client.post(
            "/api/runs",
            json={"dataset_id": dataset_id, "usecase": "discount_leakage", "mapping": mapping,
                  "options": {"time_budget_s": 75}},
        )
        check("POST /api/runs -> 200", r.status_code == 200, r.text[:300])
        run_id = r.json()["run_id"]
        print(f"        run {run_id} started")

        events: list[dict] = []
        with client.websocket_connect(f"/ws/runs/{run_id}") as ws:
            while True:
                event = ws.receive_json()
                events.append(event)
                if event.get("phase") == "done":
                    break
        check("websocket streamed events", len(events) > 15, f"got {len(events)}")
        check("websocket ended with done", events[-1].get("phase") == "done")
        phases = {e.get("phase") for e in events}
        for required in ("load", "profile", "prepare", "plan", "train", "select", "done"):
            check(f"stream covers phase {required!r}", required in phases, str(sorted(phases)))
        seqs = [e.get("seq") for e in events if isinstance(e.get("seq"), int)]
        check("event seq strictly increasing", seqs == sorted(seqs) and len(set(seqs)) == len(seqs))
        train_events = [e for e in events if e.get("phase") == "train"]
        print(f"        {len(events)} events, {len(train_events)} training events, "
              f"{sum(1 for e in events if e.get('level') == 'champion')} champion marker(s)")

        # ---- result -------------------------------------------------------- #
        r = client.get(f"/api/runs/{run_id}")
        check("GET /api/runs/{id} -> 200", r.status_code == 200)
        body = r.json()
        run = body["run"]
        check("run status succeeded", body["status"] == "succeeded", str(body.get("error"))[:300])
        check("leaderboard populated", len(run.get("leaderboard", [])) >= 5,
              f"got {len(run.get('leaderboard', []))}")
        check("champion chosen", bool(run.get("champion")))
        check("champion explains itself", bool(run["champion"].get("why_won")))
        check("every leaderboard row has a why", all(e.get("why") for e in run["leaderboard"]))
        check("impact has a value", run.get("impact", {}).get("value") is not None,
              str(run.get("impact")))
        check("drivers populated", len(run.get("drivers", [])) > 0)
        check("insights populated", len(run.get("insights", [])) > 0)
        check("narration beats populated", len(run.get("narration", [])) > 0)
        check("preview rows present", len(run.get("preview", {}).get("rows", [])) > 0)
        check("distributions present", bool(run.get("distributions")))
        check("scored file written", Path(run.get("scored_path", "")).exists(),
              run.get("scored_path", ""))
        check("run payload is JSON-safe", json.dumps(run) is not None)

        # ---- per-row explanations ------------------------------------------ #
        exps = run.get("explanations") or []
        check("per-row explanations produced", len(exps) >= 5, f"got {len(exps)}")
        if exps:
            first = exps[0]
            check("explanation has a human label",
                  bool(first.get("label")) and first["label"] not in ("Other", "None"),
                  str(first.get("label")))
            check("explanation label looks like a record id",
                  "ORD-" in first["label"] or "row" in first["label"], str(first["label"]))
            check("explanation has a score", isinstance(first.get("score"), (int, float)))
            check("explanation has a summary", bool(first.get("summary")))
            check("explanation has drivers", len(first.get("drivers") or []) >= 3)
            check("every explanation driver has a plain sentence",
                  all(d.get("plain") for d in first["drivers"]))
            check("driver kinds are classified",
                  all(d.get("kind") in ("number", "category", "percent")
                      for d in first["drivers"]))
            pct = [d for d in first["drivers"] if d.get("kind") == "percent"]
            check("percentage columns render as percentages",
                  (not pct) or any("%" in d["plain"] for d in pct),
                  str([d["plain"][:60] for d in pct]))
            scores = [e["score"] for e in exps]
            check("explanations ordered worst-first", scores == sorted(scores, reverse=True),
                  str(scores[:5]))
            check("explanations are JSON-safe", json.dumps(exps) is not None)
            print(f"        {len(exps)} row explanations, e.g. {first['label']} "
                  f"(score {first['score']})")
            for d in first["drivers"][:2]:
                print(f"          {d['contribution']:+.3f}  {d['plain'][:110]}")

        print(f"        champion : {run['champion'].get('display')} "
              f"({run['champion'].get('persona')})")
        print(f"        metric   : {run['metrics']}")
        print(f"        impact   : {run['impact'].get('label')} = "
              f"{run['impact'].get('value')} {run['impact'].get('unit')}")
        print(f"        duration : {run.get('duration_ms', 0)/1000:.1f} s")

        # ---- layman-language guard ----------------------------------------- #
        banned = ["hyperparameter", "ROC-AUC", "one-hot", "estimator", "cross-validation"]
        offenders = []
        for e in events:
            blob = f"{e.get('title','')} {e.get('detail','')}".lower()
            for term in banned:
                if term.lower() in blob:
                    offenders.append((e.get("seq"), term))
        check("no unglossed jargon in the live log", not offenders, str(offenders[:4]))

        # ---- exports ------------------------------------------------------- #
        for fmt, needle in (("html", b"<!DOCTYPE html"), ("csv", b"\xef\xbb\xbf"),
                            ("xlsx", b"PK"), ("json", b"{")):
            r = client.get(f"/api/runs/{run_id}/export?fmt={fmt}")
            ok = r.status_code == 200 and len(r.content) > 200 and needle in r.content[:4000]
            check(f"export {fmt} downloads a real file ({len(r.content)/1024:.0f} KB)", ok,
                  f"status={r.status_code}")

        r = client.get(f"/api/runs/{run_id}/export?fmt=html")
        report = r.content.decode("utf-8", "replace")
        check("report includes the row explanations section", "Why these rows?" in report)
        check("report explains the method honestly",
              "swapping one column at a time" in report)
        check("report carries no currency symbol",
              "\u20b9" not in report and "$" not in report.replace("&#36;", ""),
              "found a currency symbol")
        if exps:
            check("report names a specific flagged record",
                  exps[0]["label"].split(" ")[0][:6] in report, exps[0]["label"])
        r = client.get(f"/api/runs/{run_id}/export?fmt=bogus")
        check("unknown export format rejected", r.status_code == 422)

        # ---- error handling ------------------------------------------------- #
        check("unknown dataset -> 404", client.get("/api/datasets/nope").status_code == 404)
        check("unknown use case -> 404", client.get("/api/usecases/nope").status_code == 404)
        check("unknown run -> 404", client.get("/api/runs/nope").status_code == 404)
        r = client.post("/api/runs", json={"dataset_id": dataset_id, "usecase": "discount_leakage",
                                          "mapping": {"discount": ["discount_pct"]}})
        check("missing required role -> 422", r.status_code == 422, r.text[:200])
        r = client.post(f"/api/datasets/{dataset_id}/columns/preview",
                        json={"name": "x", "expression": "__import__('os').system('dir')"})
        check("malicious formula refused", r.json().get("ok") is False, r.text[:200])

        # ---- static frontend ------------------------------------------------ #
        r = client.get("/")
        check("GET / serves the interface", r.status_code == 200 and b"<html" in r.content[:2000].lower())
        for asset in ("/styles.css", "/app.js", "/charts.js", "/favicon.svg"):
            rr = client.get(asset)
            check(f"static asset {asset}", rr.status_code == 200 and len(rr.content) > 200,
                  f"status={rr.status_code}")

    print("=" * 78)
    print(f"{CHECKS - len(FAILURES)}/{CHECKS} checks passed")
    if FAILURES:
        print("FAILURES:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
