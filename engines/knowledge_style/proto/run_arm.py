#!/usr/bin/env python3
"""Run every question under one named configuration. One arm of an experiment.

    python3 run_arm.py --arm agent_v2
    python3 run_arm.py --arm agent_v1 --prompt v1
    python3 run_arm.py --arm single --single

An arm is a frozen configuration plus its results, so two arms are comparable
and the thing that changed between them is written down rather than remembered.
"""
import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import agent  # noqa: E402
import llm  # noqa: E402
import paths  # noqa: E402
import ask  # noqa: E402

HERE = pathlib.Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--subject", default="healthygamer")
    ap.add_argument("--config", default=paths.DEFAULT_CONFIG)
    ap.add_argument("--prompt", default="v2", choices=list(agent.PROMPTS))
    ap.add_argument("--max-turns", type=int, default=5)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--expand", type=int, default=1)
    ap.add_argument("--exemplars", type=int, default=8)
    ap.add_argument("--single", action="store_true",
                    help="r0 cell: one search on the raw question, no rewriting")
    ap.add_argument("--gate", action="store_true",
                    help="LLM relevance filter over retrieved spans")
    ap.add_argument("--map", action="store_true",
                    help="put the archive map in front of the searcher "
                         "(recipe phase 3.4 item 2)")
    ap.add_argument("--rollouts", type=int, default=1,
                    help="repeat every question N times. rollouts bypass the "
                         "call cache, or they are copies rather than samples")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()

    subj = paths.Subject(a.subject, a.config).mkdirs()
    chunks, bm, ex = agent.load(a.subject, a.config, n_ex=a.exemplars)
    spec = json.loads(subj.questions.read_text())
    probes = spec["questions"]
    if a.only:
        probes = [p for p in probes if p["id"] in set(a.only)]

    if a.rollouts > 1:
        llm.FRESH = True   # a cached rollout is not a sample, it is a copy

    results = []
    plan = [(p, r) for p in probes for r in range(a.rollouts)]
    for n, (p, roll) in enumerate(plan, 1):
        t0 = time.time()
        try:
            r = agent.run(p["q"], chunks, bm, ex, a.prompt, a.max_turns,
                          a.k, a.expand, single=a.single, gate=a.gate,
                          subject=a.subject, use_map=a.map)
            err = None
        except Exception as e:
            r = {"answer": "", "references": [], "grounding": "error",
                 "queries": [], "turns": 0, "retrieved": []}
            err = str(e)[:200]

        row = dict(p)
        row.update(r)
        row["rollout"] = roll
        row["elapsed_s"] = round(time.time() - t0, 2)
        row["error"] = err
        results.append(row)
        print(f"[{n:2d}/{len(plan)}] {p['id']}"
              f"{('.'+str(roll)) if a.rollouts > 1 else '   '} {p['band']:<13} "
              f"turns={r['turns']} q={len(r['queries'])} "
              f"spans={len(r['retrieved']):3d} ground={r['grounding']:<9} "
              f"{row['elapsed_s']:5.1f}s", flush=True)

    bad = [r["id"] for r in results if r.get("error")]
    if bad:
        print(f"\n!! {len(bad)} FAILED and are recorded as empty rows: "
              f"{' '.join(bad)}\n   this arm is INVALID until they are re-run.")

    dest = subj.arm(a.arm)
    dest.write_text(json.dumps({
        "arm": a.arm, "subject": a.subject,
        "index_version": subj.index_version(),
        "config": {"prompt": a.prompt, "max_turns": a.max_turns, "k": a.k,
                   "expand": a.expand, "exemplars": a.exemplars, "gate": a.gate,
                   "rollouts": a.rollouts,
                   "mode": "single" if a.single else "agent"},
        "provenance": llm.stamp(prompt_sha=llm.prompt_sha(
            agent.PROMPTS[a.prompt], agent.TOOLS, agent.FINAL,
            agent.GATE, agent.EMPTY)),
        "n": len(results), "n_failed": len(bad), "failed": bad,
        "results": results,
    }, indent=2, ensure_ascii=False))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
