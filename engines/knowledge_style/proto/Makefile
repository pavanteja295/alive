# Grounded answering prototype.
#
# Two entry points, and they are the two things you actually do:
#
#   make onboard SUBJECT=x TAKES=/path/to/takes     a creator we have never seen
#   make refresh SUBJECT=x                          new takes for a creator we have
#
#   make status                                     what exists, what has drifted
#
# Everything for a subject lives under work/<subject>/ (= data/answers/<subject>/), so two creators never
# collide. The oracle is the expensive reusable artifact: it depends on the
# CORPUS, not on the retriever or the prompt, and its labels are timestamp spans
# so re-chunking does not invalidate them. Adding takes does invalidate it, and
# `make status` says so rather than letting a wrong number through.
#
# The LLM key: data/answers/.protokey, or ANTHROPIC_API_KEY. With neither, generation
# shells out to the `claude` CLI instead.

SUBJECT ?= healthygamer
CONFIG  ?= chunk-w300-o60
TAKES   ?=
PY      ?= python3
W       := $(shell $(PY) -c "import paths; print(paths.WORK)")/$(SUBJECT)

.PHONY: help status check onboard refresh index map questions oracle \
        arms crossed gate score report qprofile preflight qcal qcorpus qtiers qtag qdiag qdiag-full qreview clean-arms clean-subject

help:
	@sed -n '2,18p' Makefile | sed 's/^# \?//'
	@echo ""
	@$(MAKE) --no-print-directory status

# ---------------------------------------------------------------- status
status:
	@$(PY) paths.py

check:
	@command -v $(PY) >/dev/null || { echo "no python3"; exit 1; }
	@command -v claude >/dev/null || { echo "no claude CLI on PATH"; exit 1; }
	@echo "ok: python3, claude CLI"

# ================================================================ ONBOARD
# A creator we have never seen. Builds the entire stack from a takes directory.
# ~15 min of calls for 17 takes: 17 for the map, 1 for questions, 20 for oracle.
onboard: check
	@test -n "$(TAKES)" || { echo "usage: make onboard SUBJECT=name TAKES=/path"; exit 1; }
	@echo "== 1/5 index"      && $(PY) build.py --takes "$(TAKES)" --subject $(SUBJECT)
	@echo "== 2/5 map"        && $(PY) mapindex.py --subject $(SUBJECT) --config $(CONFIG)
	@echo "== 3/5 questions"  && $(PY) makequestions.py --subject $(SUBJECT) --config $(CONFIG)
	@echo "== 4/5 oracle"     && $(PY) oracle.py --creator $(SUBJECT) && $(PY) oracle.py --creator $(SUBJECT) --tighten
	@echo "== 5/5 baseline arm" && $(PY) run_arm.py --arm baseline --subject $(SUBJECT) --config $(CONFIG)
	@$(PY) verify.py --runs baseline --subject $(SUBJECT) --config $(CONFIG) | tail -8
	@echo "" && $(MAKE) --no-print-directory status

# ================================================================ REFRESH
# New takes for a creator we already have. Cheap, because the map caches per
# take and only new ones cost a call. The oracle is NOT auto-rebuilt: it is the
# expensive artifact and rebuilding it silently would hide that every stored
# score was measured against different ground truth. status says it is stale.
refresh: check
	@test -n "$(TAKES)" || { echo "usage: make refresh SUBJECT=name TAKES=/path"; exit 1; }
	@echo "== index (rebuilt, deterministic)" && $(PY) build.py --takes "$(TAKES)" --subject $(SUBJECT)
	@echo "== map (only new takes cost a call)" && $(PY) mapindex.py --subject $(SUBJECT) --config $(CONFIG)
	@echo ""
	@$(PY) paths.py $(SUBJECT)
	@echo ""
	@echo "The oracle was NOT rebuilt. If status says STALE, every existing score"
	@echo "for this subject was measured against ground truth that never saw the"
	@echo "new takes. Rebuild it, then re-run the arms you care about:"
	@echo "    make oracle SUBJECT=$(SUBJECT)"
	@echo "    make arms   SUBJECT=$(SUBJECT)"

# ---------------------------------------------------------------- stages
index:
	$(PY) build.py --takes "$(TAKES)" --subject $(SUBJECT)

map:
	$(PY) mapindex.py --subject $(SUBJECT) --config $(CONFIG)

questions:
	$(PY) makequestions.py --subject $(SUBJECT) --config $(CONFIG)

oracle:
	$(PY) oracle.py --creator $(SUBJECT) && $(PY) oracle.py --creator $(SUBJECT) --tighten

# ---------------------------------------------------------------- arms
arms:
	$(PY) run_arm.py --arm single --single --subject $(SUBJECT) --config $(CONFIG)
	$(PY) run_arm.py --arm agent  --prompt v2 --subject $(SUBJECT) --config $(CONFIG)

# The 2x2. Retrieval crossed with prompt, so a main effect can be told apart
# from an interaction. Running only the diagonal confounds them.
crossed:
	$(PY) run_arm.py --arm r0u0 --single --prompt v1 --subject $(SUBJECT) --config $(CONFIG)
	$(PY) run_arm.py --arm r0u1 --single --prompt v2 --subject $(SUBJECT) --config $(CONFIG)
	$(PY) run_arm.py --arm r1u0 --prompt v1 --subject $(SUBJECT) --config $(CONFIG)
	$(PY) run_arm.py --arm r1u1 --prompt v2 --subject $(SUBJECT) --config $(CONFIG)

gate:
	$(PY) run_arm.py --arm gate --prompt v2 --gate --subject $(SUBJECT) --config $(CONFIG)

# ---------------------------------------------------------------- score
# --no-judge is mechanical span overlap: free, deterministic, no model opinion.
score:
	@for f in $(W)/arms/runs_*.json; do \
	  case "$$f" in *_scored.json) continue;; esac; \
	  echo "== $$(basename $$f)"; \
	  $(PY) verify.py --runs "$$f" --no-judge --subject $(SUBJECT) --config $(CONFIG) | tail -7; \
	done

score-judged:
	@for f in $(W)/arms/runs_*.json; do \
	  case "$$f" in *_scored.json) continue;; esac; \
	  $(PY) verify.py --runs "$$f" --subject $(SUBJECT) --config $(CONFIG) | tail -7; \
	done

report:
	$(PY) assess.py --subject $(SUBJECT)
	$(PY) pipeline_doc.py --subject $(SUBJECT)

# ------------------------------------------------- phase 1: question corpus
# The questions people are MOST LIKELY to ask that a passage already answers.
# The objective and why each piece is shaped this way: recipes/question-corpus/RECIPE.md.
# One call per PASSAGE, not per take. The content model does not run here, and
# retrieval is recorded rather than used as a gate.
#
#   make qcal     SUBJECT=x            10 passages spread across all videos.
#                                      ~15 min. READ IT FIRST.
#   --sample, not --limit: --limit takes the FIRST 10, which are all one video
#   and start with an intro. That returned 6 of 25 and read like a bad creator.
#   make qcorpus  SUBJECT=x N=20       all 292 passages, ~4 h at 8-way
#   make qreview  SUBJECT=x            render for a human to read
#
# Resumable: a killed run picks up at the first passage not already in the file,
# and every call is cached, so a rerun of finished work is free.
N ?= 20

# Drafts profiles/<subject>.py from the corpus. A DRAFT: read it before running.
# This closes the one gap that used to be a copy-and-edit instruction, and the
# dangerous failure it prevents is a profile whose WHO was edited while another
# creator's example questions were left in place.
qprofile:
	$(PY) recipes/question-corpus/tools/qprofile.py --subject $(SUBJECT) --config $(CONFIG)

# Every precondition in RECIPE.md section 0, before any work starts. Raises.
preflight:
	$(PY) recipes/question-corpus/tools/preflight.py --subject $(SUBJECT) --config $(CONFIG)

# fact_bearing per answer, by model call. Required before any style cut: the
# regex this replaced measured 31% wrong in both directions and its errors land
# in training data.
qtag:
	$(PY) recipes/question-corpus/tools/qtag.py --subject $(SUBJECT) --config $(CONFIG)

qcal:
	$(PY) recipes/question-corpus/tools/genq.py --subject $(SUBJECT) --config $(CONFIG) --all --sample 10 --n $(N)
	$(PY) recipes/question-corpus/tools/qreview.py --in $(W)/qcorpus.jsonl --out $(W)/reports/qcorpus.html
	@echo "" && echo "Read $(W)/reports/qcorpus.html before running: make qcorpus"

qcorpus:
	$(PY) recipes/question-corpus/tools/genq.py --subject $(SUBJECT) --config $(CONFIG) --all --n $(N)

qreview:
	$(PY) recipes/question-corpus/tools/qreview.py --in $(W)/qcorpus.jsonl --out $(W)/reports/qcorpus.html

# Tiers 2-4. Run AFTER qcorpus: tier 4 rephrases whatever exists, so it goes
# last and multiplies the rest. Tier 3 has low yield by nature -- cross-video
# questions are rarer and harder to state -- do not force the count.
qtiers:
	$(PY) recipes/question-corpus/tools/genq_tiers.py --subject $(SUBJECT) --config $(CONFIG) --tier 2 --groups 300
	$(PY) recipes/question-corpus/tools/genq_tiers.py --subject $(SUBJECT) --config $(CONFIG) --tier 3 --pairs 120
	$(PY) recipes/question-corpus/tools/genq_tiers.py --subject $(SUBJECT) --config $(CONFIG) --tier 4 --of 800 --per 3

# The free checks. Add the two that call a model once the corpus is built:
#   make qdiag-full SUBJECT=x
qdiag:
	$(PY) recipes/question-corpus/tools/qdiagnose.py --subject $(SUBJECT) --config $(CONFIG)

qdiag-full:
	$(PY) recipes/question-corpus/tools/qdiagnose.py --subject $(SUBJECT) --config $(CONFIG) \
	  --check stats yield position dupes facts retrieval negctl standalone

# ---------------------------------------------------------------- clean
clean-arms:
	rm -f $(W)/arms/*.json

clean-subject:
	@echo "removing $(W) entirely, including the oracle. ctrl-c to abort."
	@sleep 3 && rm -rf $(W)
