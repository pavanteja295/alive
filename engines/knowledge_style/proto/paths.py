#!/usr/bin/env python3
"""Where every artifact for a subject lives, and whether it is still valid.

One place, because the alternative is what this prototype had until now: the
chunk store was scoped by subject and everything derived from it was not, so
onboarding a second creator would have silently overwritten the first one's
oracle, map, questions and reports.

    work/<subject>/
      store/<config>/      chunks.jsonl  exemplars.json  manifest.json
      map.json             navigation metadata, one entry per take
      questions.json       the probe set
      oracle.json          relevance labels, keyed to an index_version
      arms/                runs_<arm>.json and runs_<arm>_scored.json
      reports/             PIPELINE.html  assessment.html  probe_report.html
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
# Per-creator state (indexed transcripts, questions, arms, chats) is data: it lives in
# <alive>/data/answers, not beside the code. The committed records under ./work stay
# where they are; nothing reads them at serve time.
WORK = HERE.resolve().parents[2] / "data" / "answers"
DEFAULT_CONFIG = "chunk-w300-o60"


class Subject:
    def __init__(self, name, config=DEFAULT_CONFIG):
        self.name = name
        self.config = config
        self.root = WORK / name

    # -------------------------------------------------------------- paths
    @property
    def store(self):
        return self.root / "store" / self.config

    @property
    def chunks(self):
        return self.store / "chunks.jsonl"

    @property
    def manifest(self):
        return self.store / "manifest.json"

    @property
    def exemplars(self):
        return self.store / "exemplars.json"

    @property
    def map(self):
        return self.root / "map.json"

    @property
    def questions(self):
        return self.root / "questions.json"

    @property
    def oracle(self):
        return self.root / "oracle.json"

    @property
    def arms(self):
        return self.root / "arms"

    @property
    def reports(self):
        return self.root / "reports"

    def arm(self, name):
        return self.arms / f"runs_{name}.json"

    def scored(self, name):
        return self.arms / f"runs_{name}_scored.json"

    def mkdirs(self):
        for d in (self.store, self.arms, self.reports):
            d.mkdir(parents=True, exist_ok=True)
        return self

    # -------------------------------------------------------------- state
    def index_version(self):
        if not self.manifest.exists():
            return None
        return json.loads(self.manifest.read_text())["index_version"]

    def oracle_version(self):
        if not self.oracle.exists():
            return None
        return json.loads(self.oracle.read_text()).get("built_against_index")

    def mapped_takes(self):
        if not self.map.exists():
            return set()
        return set(json.loads(self.map.read_text())["takes"])

    def store_takes(self):
        if not self.chunks.exists():
            return set()
        return {json.loads(l)["take_id"]
                for l in self.chunks.read_text().splitlines()}

    def status(self):
        """What exists, and what has drifted out of date.

        Adding takes changes index_version. The oracle's labels are timestamp
        spans so they survive re-chunking, but they cannot cover material the
        labeller never saw, so recall against a grown corpus is measured
        against incomplete ground truth. That is a silent wrong number, which
        is the failure mode this whole project keeps running into, so it is
        reported rather than left to be noticed.
        """
        iv, ov = self.index_version(), self.oracle_version()
        st, mt = self.store_takes(), self.mapped_takes()
        return {
            "subject": self.name,
            "config": self.config,
            "has_store": self.chunks.exists(),
            "has_map": self.map.exists(),
            "has_questions": self.questions.exists(),
            "has_oracle": self.oracle.exists(),
            "n_takes": len(st),
            "index_version": iv,
            "oracle_version": ov,
            "oracle_stale": bool(iv and ov and iv != ov),
            "unmapped_takes": sorted(st - mt),
            "arms": sorted(p.stem.replace("runs_", "") for p in
                           self.arms.glob("runs_*.json")
                           if not p.stem.endswith("_scored"))
            if self.arms.exists() else [],
        }


def known():
    if not WORK.exists():
        return []
    return sorted(p.name for p in WORK.iterdir() if p.is_dir())


def main():
    import sys
    subs = sys.argv[1:] or known()
    if not subs:
        print(f"no subjects under {WORK}")
        return
    for name in subs:
        s = Subject(name).status()
        flags = []
        if not s["has_store"]:
            flags.append("NO STORE")
        if not s["has_map"]:
            flags.append("no map")
        if not s["has_questions"]:
            flags.append("no questions")
        if not s["has_oracle"]:
            flags.append("no oracle")
        if s["oracle_stale"]:
            flags.append(f"ORACLE STALE ({s['oracle_version']} != "
                         f"{s['index_version']})")
        if s["unmapped_takes"]:
            flags.append(f"{len(s['unmapped_takes'])} takes not in map")
        print(f"{s['subject']:<18}{s['n_takes']:>3} takes  "
              f"index {str(s['index_version'])[:12]:<14}"
              f"arms {len(s['arms']):<3} "
              f"{' | '.join(flags) if flags else 'ready'}")


if __name__ == "__main__":
    main()
