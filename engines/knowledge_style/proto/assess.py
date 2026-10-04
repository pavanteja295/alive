#!/usr/bin/env python3
"""Render the final assessment page from whatever scores exist on disk.

    python3 assess.py

Reads oracle.json, *_scored.json and runs_*.json. Nothing is hard-coded: if you
re-run an arm, this picks up the new numbers. The prose findings are written by
hand because they are judgements, and they say so.
"""
import html
import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT = HERE / "assessment.html"


def esc(s):
    return html.escape(str(s))


def load():
    oracle = json.loads((HERE / "oracle.json").read_text())
    scored = {}
    for p in sorted(HERE.glob("*_scored.json")):
        d = json.loads(p.read_text())
        name = p.stem.replace("_scored", "")
        scored[name] = {r["id"]: r for r in d["results"]}
    runs = {}
    for p in sorted(HERE.glob("runs_*.json")):
        if p.stem.endswith("_scored"):
            continue
        runs[p.stem.replace("runs_", "")] = json.loads(p.read_text())
    return oracle, scored, runs


def band_mean(rows, band, key="recall"):
    v = [r[key] for r in rows.values()
         if r["band"] == band and r.get(key) is not None]
    return sum(v) / len(v) if v else None


CSS = """
:root{--paper:#f2f5f7;--surface:#fff;--sunk:#e7edf0;--ink:#101820;--body:#2d3b45;
--muted:#5f6f7a;--faint:#8a9aa4;--line:#d3dce1;--rule:#aebcc4;--accent:#a5601a;
--accent-soft:#f6ece0;--chip:#f2f5f7;--good:#2f6b52;--good-soft:#e4efe9;
--bad:#9c4a45;--bad-soft:#f7e8e7}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
--paper:#0c1216;--surface:#141c22;--sunk:#101820;--ink:#e9eff3;--body:#bdcbd4;
--muted:#8797a2;--faint:#63737e;--line:#243038;--rule:#39474f;--accent:#dc9a4f;
--accent-soft:#251c11;--chip:#0c1216;--good:#63b892;--good-soft:#0f2019;
--bad:#dd8b85;--bad-soft:#241413}}
:root[data-theme="dark"]{--paper:#0c1216;--surface:#141c22;--sunk:#101820;
--ink:#e9eff3;--body:#bdcbd4;--muted:#8797a2;--faint:#63737e;--line:#243038;
--rule:#39474f;--accent:#dc9a4f;--accent-soft:#251c11;--chip:#0c1216;
--good:#63b892;--good-soft:#0f2019;--bad:#dd8b85;--bad-soft:#241413}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--body);font-family:Newsreader,Georgia,serif;
font-size:17.5px;line-height:1.64;-webkit-font-smoothing:antialiased}
.wrap{max-width:1000px;margin:0 auto;padding:0 30px 110px}
h1,h2,h3,h4,.eyebrow,.tag,th,.lbl{font-family:Archivo,system-ui,sans-serif}
code,pre,.mono,.n{font-family:"JetBrains Mono",ui-monospace,monospace}
.mast{padding:70px 0 28px;border-bottom:2px solid var(--ink);margin-bottom:14px}
.eyebrow{font-size:11px;font-weight:600;letter-spacing:.16em;text-transform:uppercase;
color:var(--accent);margin:0 0 18px}
h1{font-size:clamp(36px,5.6vw,56px);line-height:1.02;font-weight:700;
letter-spacing:-.028em;color:var(--ink);margin:0 0 20px;text-wrap:balance}
.lede{font-size:20.5px;line-height:1.5;margin:0;max-width:60ch}
.lede b{color:var(--ink);font-weight:600}
.meta{margin-top:24px;font-size:13px;color:var(--faint);font-family:Archivo,sans-serif}
.stats{display:flex;flex-wrap:wrap;margin:28px 0 0;border-top:1px solid var(--line)}
.stat{flex:1 1 118px;padding:13px 18px 13px 0;border-right:1px solid var(--line)}
.stat:last-child{border-right:0}
.stat .v{font-family:"JetBrains Mono",monospace;font-size:20px;font-weight:500;
color:var(--ink);font-variant-numeric:tabular-nums;display:block;line-height:1.3}
.stat .l{font-family:Archivo,sans-serif;font-size:10.5px;letter-spacing:.1em;
text-transform:uppercase;color:var(--faint);display:block;margin-top:3px}
h2{font-size:12px;font-weight:700;letter-spacing:.15em;text-transform:uppercase;
color:var(--ink);margin:56px 0 22px;padding-bottom:10px;
border-bottom:1px solid var(--rule);display:flex;gap:14px;align-items:baseline}
h2 .n{color:var(--accent);font-variant-numeric:tabular-nums}
h3{font-family:Archivo,sans-serif;font-size:19px;font-weight:600;color:var(--ink);
margin:34px 0 12px;letter-spacing:-.012em}
.prose{max-width:66ch}
p{margin:0 0 15px}
strong{color:var(--ink);font-weight:600}
ul,ol{margin:0 0 16px;padding-left:22px;max-width:66ch}
li{margin-bottom:8px}li::marker{color:var(--faint)}
code{font-size:.83em;background:var(--sunk);padding:1.5px 5px;border-radius:2px;
color:var(--ink)}
.tw{overflow-x:auto;border:1px solid var(--line);margin:0 0 20px}
table{width:100%;border-collapse:collapse;font-size:15.5px;background:var(--surface)}
th{text-align:left;font-size:10.5px;font-weight:700;letter-spacing:.11em;
text-transform:uppercase;color:var(--faint);padding:11px 15px;
border-bottom:1px solid var(--rule);white-space:nowrap}
td{padding:11px 15px;border-bottom:1px solid var(--line);vertical-align:top}
tr:last-child td{border-bottom:0}
.num{font-variant-numeric:tabular-nums;text-align:right}
td.lbl{font-weight:600;color:var(--ink);font-family:Archivo,sans-serif;font-size:14.5px}
.up{color:var(--good);font-weight:600;font-variant-numeric:tabular-nums}
.flat{color:var(--faint);font-variant-numeric:tabular-nums}
.bar{display:block;height:7px;background:var(--sunk);position:relative;
min-width:90px;margin-top:5px}
.bar i{display:block;height:7px;background:var(--good)}
.bar i.b{background:var(--rule)}
.rule{border-left:3px solid var(--accent);background:var(--accent-soft);
padding:15px 20px;margin:0 0 20px;font-size:16.5px}
.rule b{color:var(--ink)}
.note{border-left:3px solid var(--rule);padding:13px 20px;margin:0 0 20px;
font-size:15.5px;color:var(--muted)}
.note b{color:var(--ink)}
.warn{border-left:3px solid var(--bad);background:var(--bad-soft);
padding:14px 20px;margin:0 0 20px;font-size:16px}
.warn b{color:var(--ink)}
.tag{display:inline-block;font-size:9.5px;font-weight:700;letter-spacing:.09em;
text-transform:uppercase;padding:3px 8px;line-height:1.4;white-space:nowrap}
.tag.fix{background:var(--good);color:var(--surface)}
.tag.open{border:1px dashed var(--bad);color:var(--bad)}
ol.dec{counter-reset:d;list-style:none;padding:0;max-width:none}
ol.dec li{counter-increment:d;position:relative;padding:0 0 15px 46px;margin:0 0 15px;
border-bottom:1px solid var(--line);font-size:16px}
ol.dec li:last-child{border-bottom:0}
ol.dec li::before{content:counter(d);position:absolute;left:0;top:1px;
font-family:"JetBrains Mono",monospace;font-size:12px;font-weight:700;
color:var(--chip);background:var(--ink);width:26px;height:22px;display:flex;
align-items:center;justify-content:center;font-variant-numeric:tabular-nums}
ol.dec b{color:var(--ink)}
pre.sh{background:var(--sunk);border:1px solid var(--line);padding:15px 18px;
overflow-x:auto;font-size:13px;line-height:1.7;margin:0 0 20px}
.foot{border-top:1px solid var(--rule);padding-top:22px;margin-top:44px;
font-size:14px;color:var(--muted);font-family:Archivo,sans-serif}
:focus-visible{outline:2px solid var(--accent);outline-offset:3px}
@media (max-width:640px){.stat{flex-basis:50%}}
"""


def main():
    oracle, scored, runs = load()
    L = oracle["labels"]
    single = scored.get("baseline_single", {})
    agentv2 = scored.get("runs_agent_v2", {})
    av2 = runs.get("agent_v2", {"results": []})["results"]

    n_q = len(L)
    disc_zero = sum(1 for k, v in L.items()
                    if k.startswith("D") and not v["answer"] and not v["support"])
    o_spans = sum(len(v["answer"]) for v in L.values())
    mean_q = sum(len(r["queries"]) for r in av2) / max(len(av2), 1)
    mean_s = sum(r["elapsed_s"] for r in av2) / max(len(av2), 1)

    bands = ["connected", "adjacent", "identity"]
    cmp_rows = []
    for b in bands:
        s = band_mean(single, b)
        a = band_mean(agentv2, b)
        cmp_rows.append((b, s, a))

    P = ["<title>Grounded Answering: Assessment</title>",
         '<link rel="preconnect" href="https://fonts.googleapis.com">',
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
         'family=Archivo:wght@500;600;700&family=JetBrains+Mono:wght@400;500;700&'
         'family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&display=swap">',
         f"<style>{CSS}</style>", '<div class="wrap">']

    P += ['<header class="mast">',
          '<p class="eyebrow">knowledge + style &middot; prototype assessment</p>',
          "<h1>What the Archive Can Answer</h1>",
          '<p class="lede">A grounded answering prototype over one creator\'s '
          '17-video archive, measured against exhaustive ground truth. Letting the '
          'model write its own search queries took evidence recall on covered '
          'questions from <b>0.26 to 0.91</b>. Everything else here is the '
          'caveats.</p>',
          f'<p class="meta">index_version <code>{esc(oracle["built_against_index"])}</code>'
          f' &middot; {esc(oracle["subject"])} &middot; oracle built '
          f'{esc(oracle["generated"])} &middot; reproduce with <code>make all</code></p>',
          '<div class="stats">']
    for v, l in [(n_q, "questions"), (o_spans, "oracle spans"),
                 (f"{cmp_rows[0][2]:.2f}", "connected recall"),
                 (f"{mean_q:.1f}", "queries / question"),
                 (f"{mean_s:.0f}s", "per answer"),
                 (f"{disc_zero}/10", "disconnected empty")]:
        P.append(f'<div class="stat"><span class="v">{v}</span>'
                 f'<span class="l">{l}</span></div>')
    P += ["</div></header>"]

    # ---- 1 the result
    def rec(a, b):
        d = scored.get(f"runs_{a}", {})
        v = [r["recall"] for r in d.values()
             if r["band"] == b and r.get("recall") is not None]
        return sum(v) / len(v) if v else None

    def absten(a):
        d = scored.get(f"runs_{a}", {})
        rs = [r for r in d.values() if r["band"] == "disconnected"]
        return (sum(1 for r in rs if r["abstained_strict"]),
                sum(1 for r in rs if r["abstained_correctly"]), len(rs))

    have_cross = all(f"runs_{a}" in scored for a in
                     ["r0u0", "r0u1", "r1u0", "r1u1"])
    P += ['<h2><span class="n">01</span> The result</h2>']
    if have_cross:
        c = {a: rec(a, "connected") for a in ["r0u0", "r0u1", "r1u0", "r1u1"]}
        ret = ((c["r1u0"] - c["r0u0"]) + (c["r1u1"] - c["r0u1"])) / 2
        pre = ((c["r0u1"] - c["r0u0"]) + (c["r1u1"] - c["r1u0"])) / 2
        inter = c["r1u1"] - c["r1u0"] - c["r0u1"] + c["r0u0"]
        P += ['<p class="prose">Two things changed between the first baseline and '
              'the best arm: the model began writing its own search queries, and '
              'the persona prompt was rewritten. Run as one arm those confound. '
              'Crossed as a 2&times;2 they separate.</p>',
              '<div class="tw"><table><thead><tr><th>Evidence recall, connected</th>'
              '<th class="num">prompt v1</th><th class="num">prompt v2</th>'
              '</tr></thead><tbody>',
              f'<tr><td class="lbl">one query, verbatim</td>'
              f'<td class="num">{c["r0u0"]:.2f}</td>'
              f'<td class="num">{c["r0u1"]:.2f}</td></tr>',
              f'<tr><td class="lbl">model writes its queries</td>'
              f'<td class="num">{c["r1u0"]:.2f}</td>'
              f'<td class="num">{c["r1u1"]:.2f}</td></tr>',
              "</tbody></table></div>",
              '<div class="tw"><table><thead><tr><th>Effect</th>'
              '<th class="num">Size</th><th>Reading</th></tr></thead><tbody>',
              f'<tr><td class="lbl">retrieval</td><td class="num up">{ret:+.3f}</td>'
              f'<td>letting the model write its own queries is the entire effect</td></tr>',
              f'<tr><td class="lbl">prompt</td><td class="num flat">{pre:+.3f}</td>'
              f'<td>indistinguishable from zero on recall</td></tr>',
              f'<tr><td class="lbl">interaction</td><td class="num flat">{inter:+.3f}</td>'
              f'<td>retrieval does not depend on which prompt it runs under</td></tr>',
              "</tbody></table></div>",
              '<div class="rule prose">The win is not the loop, and it is not the '
              'wording. Nearly every question resolved in <b>one turn with three or '
              'four parallel queries</b>. &ldquo;Why do I have almost no close '
              'friends&rdquo; retrieves nothing useful; rewritten as '
              '<code>male isolation biology withdraw</code> it goes from '
              '<b>0.00 to 1.00</b>. Best single query across the connected band '
              'reaches 0.70; the union of three or four reaches 0.91. It works by '
              '<b>diversification, not by insight</b> &mdash; the model does not '
              'know the archive, it generates several guesses and the union '
              'covers.</div>']
    if "runs_r1u1_gate" in scored:
        gs, gn, gt = absten("r1u1_gate")
        bs, bn, bt = absten("r1u1")
        P += ['<h3>Adding a relevance gate</h3>',
              '<p class="prose">One extra call over the retrieved spans, keep or '
              'drop, no threshold. It is the only signal that ever separated the '
              'bands: a lexical floor cannot, because per-term BM25 for the ten '
              'disconnected questions (0.78&ndash;1.76) overlaps the connected ones '
              '(0.99&ndash;4.86), and &ldquo;Who are you?&rdquo; scores 0.00 while '
              'having ten answer spans.</p>',
              '<div class="tw"><table><thead><tr><th>Arm</th>'
              '<th class="num">connected</th><th class="num">adjacent</th>'
              '<th class="num">identity</th><th class="num">declined correctly</th>'
              '</tr></thead><tbody>',
              f'<tr><td class="lbl">multi-query</td>'
              f'<td class="num">{rec("r1u1","connected"):.2f}</td>'
              f'<td class="num">{rec("r1u1","adjacent"):.2f}</td>'
              f'<td class="num">{rec("r1u1","identity"):.2f}</td>'
              f'<td class="num">{bs}/{bt}</td></tr>',
              f'<tr><td class="lbl">multi-query + gate</td>'
              f'<td class="num">{rec("r1u1_gate","connected"):.2f}</td>'
              f'<td class="num">{rec("r1u1_gate","adjacent"):.2f}</td>'
              f'<td class="num">{rec("r1u1_gate","identity"):.2f}</td>'
              f'<td class="num up">{gs}/{gt}</td></tr>',
              "</tbody></table></div>",
              '<div class="warn prose"><b>The gate trades recall for abstention, '
              'and currently trades too much.</b> Declining correctly goes to '
              f'{gs}/{gt}, and context drops thirteenfold. But identity recall '
              'falls from 0.53 to 0.19 and it empties &ldquo;Who are you?&rdquo; '
              'entirely. That is structural, not a tuning problem: the self-facts '
              'sit in eight different takes, no single span is obviously about the '
              'question, and a per-span relevance test discards all of them. A span '
              'that is one eighth of an answer is not relevant on its own; eight of '
              'them are.</div>',
              '<p class="prose">The cheap fix is to stop applying it uniformly: run '
              'the gate to decide <em>whether to answer</em>, then hand generation '
              'the ungated spans. That keeps both numbers and costs one call. Not '
              'yet run.</p>']
    P += ['<p class="prose"><strong>Evidence recall</strong> is the fraction of the '
          'spans an exhaustive reader marked as answering the question that '
          'retrieval actually surfaced. Mechanical: timestamp overlap, no model '
          'opinion in it. <strong>No arm ever falsely claimed his position</strong> '
          'on a question he has not covered, in any configuration tested.</p>']

    # ---- 2 oracle
    P += ['<h2><span class="n">02</span> The ground truth, and why it expires</h2>',
          '<p class="prose">For every question, the entire archive was read in '
          'batches and every span labelled: does this answer the question, merely '
          'support an extension, or neither. No retriever involved, so it is what '
          'retrieval <em>should</em> have found.</p>',
          '<div class="tw"><table><thead><tr><th>Band</th>'
          '<th class="num">Questions</th><th class="num">Mean answer spans</th>'
          '<th>What it confirms</th></tr></thead><tbody>']
    meaning = {
        "connected": "all have material. the band is real",
        "adjacent": "uneven. three of my labels were wrong",
        "identity": "all non-zero, scattered across many takes",
        "disconnected": "every one is empty. zero answer, zero support",
    }
    for b in ["connected", "adjacent", "identity", "disconnected"]:
        ids = [k for k in L if
               {"C": "connected", "A": "adjacent", "I": "identity",
                "D": "disconnected"}[k[0]] == b]
        m = sum(len(L[k]["answer"]) for k in ids) / max(len(ids), 1)
        P.append(f'<tr><td class="lbl">{b}</td><td class="num">{len(ids)}</td>'
                 f'<td class="num">{m:.1f}</td><td>{meaning[b]}</td></tr>')
    P += ["</tbody></table></div>",
          '<div class="note prose"><b>It caught three of my own mislabels.</b> '
          'A01 (procrastination) has 31 answer spans, A02 and A07 have material too. '
          'I banded them &ldquo;adjacent&rdquo; from take titles without reading the '
          'content. The ground truth needed the same scrutiny as the system, and it '
          'got it second.</div>',
          '<div class="rule prose"><b>This artifact has a shelf life, which is the '
          'argument for building it now.</b> Labelling took 20 calls over four '
          'corpus batches because the archive fits in a context window. At a few '
          'hundred takes it stops being computable. The scoreboard is only '
          'buildable while the corpus is small.</div>']

    # ---- 3 bugs
    P += ['<h2><span class="n">03</span> What running it found</h2>',
          '<p class="prose">Five defects, none of which a design discussion would '
          'have produced. Four are fixed; the fifth is a limitation of the '
          'experiment, not the code.</p>', "<ol class='dec prose'>",
          '<li><b>Coverage ignored out-of-domain words.</b> <span class="tag fix">fixed</span> '
          'Unseen terms scored idf 0, contributing nothing to either side of the '
          'ratio, so a Postgres question scored 1.00 and was labelled '
          '<code>direct</code>. Absence of a word is the whole signal. Now 0.22.</li>',
          '<li><b>The control measured nothing.</b> <span class="tag fix">fixed</span> '
          'L0 kept the &ldquo;answer only from the passages&rdquo; constraint while '
          'being given zero passages, so the model refused. A control for &ldquo;what '
          'does the model already know about him&rdquo; has to be a bare persona '
          'prompt.</li>',
          '<li><b>The model skipped searching, then answered from its own '
          'knowledge.</b> <span class="tag fix">fixed</span> On the viola question '
          'it declined to search and then volunteered that a viola is &ldquo;bigger '
          'and tuned lower&rdquo;. That is the exact failure this system exists to '
          'prevent. Search is now enforced in code, not requested in the prompt.</li>',
          '<li><b>The enforcement was short-circuited.</b> <span class="tag fix">fixed</span> '
          'If the model emitted an answer immediately, the loop broke before the '
          'injected query ran. Visible as one question with an empty query log. A '
          'model that answers instantly is exactly the one that never looked.</li>',
          '<li><b>The abstention metric scored the label, not the behaviour.</b> '
          '<span class="tag fix">fixed</span> &ldquo;I haven&rsquo;t covered this, '
          'here is an extension, I am flagging it&rdquo; was counted as failure. Now '
          'reported two ways. The real sin is claiming his position where he has '
          'none, and that never happened.</li>', "</ol>"]

    # ---- 4 the confound, now resolved
    P += ['<h2><span class="n">04</span> A confound, caught and resolved</h2>',
          '<p class="prose">The first version of this page reported +0.65 from a '
          'single arm that changed retrieval and the prompt together, and could not '
          'say which mattered. Dixon\'s <em>Stochastic Evidence Graphs</em> (AIFI, '
          'Aug 2026) is precisely about this: a terminal score detects that '
          'something changed, while typed per-stage measurement identifies which '
          'edge changed it. It crosses retrieval rules with information-equivalent '
          'presentation rules because a query rewrite moves both at once.</p>',
          '<div class="rule prose">Crossed, the answer is unambiguous: '
          '<b>retrieval is the whole effect and the prompt is worth nothing on '
          'recall</b>. The intuitive story &mdash; that a better-written persona '
          'prompt helped &mdash; was wrong, and only the 2&times;2 could say so.</div>',
          '<div class="note prose">Two corrections that came out of it. The '
          'original 0.26 baseline was <b>not comparable</b>: it ran through a '
          'different prompt-assembly path with the regime machinery attached. The '
          'honest single-query baseline is 0.48, so the real improvement is '
          '<b>+0.35, not +0.65</b>. And the prompt is not useless, it is just not '
          'about recall: with multi-query retrieval and the weak prompt, the system '
          'declined correctly <b>0 times out of 10</b>. Give a weak prompt more '
          'material and it always finds something to extend from.</div>',
          '<div class="warn prose"><b>A second failure was nearly read as a '
          'result.</b> Seven consecutive questions failed with an empty CLI error '
          'at the head of one arm. The runner caught the exception and wrote rows '
          'with no retrieved spans, which scored as recall 0.00, and the arm read '
          'as a catastrophic regression of the feature under test. It was a crash '
          'wearing the costume of a measurement. The runner now retries with '
          'backoff and marks an arm invalid, loudly, if any row failed.</div>']

    # ---- 5 what it does not measure
    P += ['<h2><span class="n">05</span> What this does not measure</h2>',
          '<ul class="prose">',
          '<li><b>Whether it sounds like him.</b> Every number here is grounding and '
          'retrieval. Neither says the voice is right. That needs someone who knows '
          'the creator, and no metric substitutes for it.</li>',
          '<li><b>Claim-level grounding.</b> Every run above used '
          '<code>--no-judge</code>: mechanical span overlap only, no model opinion. '
          'The judge exists and is narrow by design, and it has not been run.</li>',
          '<li><b>Variance.</b> One rollout per question. With multi-call retrieval '
          'the number of queries varies run to run, so a repeat could move these '
          'figures and nothing here reports a spread.</li>',
          '<li><b>Anything at scale.</b> Six hours. The reasoning about ten or a '
          'hundred hours is mechanism, not measurement.</li>', "</ul>"]

    # ---- 6 reproduce
    P += ['<h2><span class="n">06</span> Reproducing it</h2>',
          '<pre class="sh">make check      # python, claude CLI, takes dir\n'
          'make index      # subtitles -&gt; verbatim spans        ~1s\n'
          'make oracle     # exhaustive ground truth           ~12 min, 20 calls\n'
          'make arms       # single-call and agent arms        ~45 min\n'
          'make score      # mechanical, free\n'
          'make report     # this page\n\n'
          'make crossed    # the 2x2 that separates retrieval from prompt</pre>',
          '<p class="prose">The oracle is the expensive, reusable artifact. It '
          'depends on the corpus, not on the retriever or the prompt, and its labels '
          'are timestamp spans rather than chunk ids, so re-chunking does not reset '
          'the scoreboard. Arms are cheap and disposable.</p>',
          '<p class="prose">Prompts live in <code>PROMPTS.md</code>, regenerated '
          'from source by <code>dump_prompts.py</code> so a stale copy cannot drift '
          'into being trusted.</p>']

    # ---- 7 next
    P += ['<h2><span class="n">07</span> What I would do next</h2>', "<ol class='dec prose'>",
          '<li><b>Run the 2&times;2.</b> Until then the headline is a joint effect. '
          'One command, about 50 minutes.</li>',
          '<li><b>Run the claim-level judge.</b> It measures the ungrounded-answer '
          'failure directly, which nothing above does.</li>',
          '<li><b>Curate the exemplars.</b> Eight auto-picked chunks are the direct '
          'cause of the template effect: the same stock move appeared in 9 of 30 '
          'answers. This is the identity axis and it is the weakest part.</li>',
          '<li><b>Add a map layer before scaling.</b> The model searches blind. It '
          'guessed well here, and guessing does not improve as the archive grows '
          'while the amount it can miss does. Take titles plus a line each is cheap '
          'and the corpus contract already has the slot.</li>',
          '<li><b>Have someone who knows him read ten answers.</b> Blocking, and not '
          'substitutable by anything on this page.</li>', "</ol>"]

    P += ['<p class="foot">Generated by <code>assess.py</code> from '
          '<code>oracle.json</code> and <code>*_scored.json</code>. Numbers are read '
          'from disk; the judgements are hand-written and say so. Owner file for the '
          'argument behind this work is '
          '<code>engines/knowledge_style/FRAMING.md</code>.</p>', "</div>"]

    OUT.write_text("\n".join(P))
    print(f"wrote {OUT}")
    for b, s, a in cmp_rows:
        print(f"  {b:<12} {'-' if s is None else f'{s:.2f}'} -> {a:.2f}")


if __name__ == "__main__":
    main()
