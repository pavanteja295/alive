#!/usr/bin/env python3
"""Generate PIPELINE.html: the full description of what was built and measured.

    python3 pipeline_doc.py

Numbers come from doc_stats.json, which comes from the runs. Prose is written by
hand. Regenerate doc_stats.json first if arms have changed.
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
S = json.loads((HERE / "doc_stats.json").read_text())
A = S["arms"]
IX = S["index"]

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
.wrap{max-width:1020px;margin:0 auto;padding:0 30px 110px}
h1,h2,h3,h4,.eyebrow,.tag,th,.lbl,figcaption{font-family:Archivo,system-ui,sans-serif}
code,pre,.mono,.n{font-family:"JetBrains Mono",ui-monospace,monospace}
.mast{padding:70px 0 26px;border-bottom:2px solid var(--ink);margin-bottom:12px}
.eyebrow{font-size:11px;font-weight:600;letter-spacing:.16em;text-transform:uppercase;
color:var(--accent);margin:0 0 18px}
h1{font-size:clamp(36px,5.6vw,56px);line-height:1.02;font-weight:700;
letter-spacing:-.028em;color:var(--ink);margin:0 0 20px;text-wrap:balance}
.lede{font-size:20.5px;line-height:1.5;margin:0;max-width:62ch}
.lede b{color:var(--ink);font-weight:600}
.meta{margin-top:22px;font-size:13px;color:var(--faint);font-family:Archivo,sans-serif}
.stats{display:flex;flex-wrap:wrap;margin:26px 0 0;border-top:1px solid var(--line)}
.stat{flex:1 1 112px;padding:13px 18px 13px 0;border-right:1px solid var(--line)}
.stat:last-child{border-right:0}
.stat .v{font-family:"JetBrains Mono",monospace;font-size:19px;font-weight:500;
color:var(--ink);font-variant-numeric:tabular-nums;display:block;line-height:1.3}
.stat .l{font-family:Archivo,sans-serif;font-size:10.5px;letter-spacing:.1em;
text-transform:uppercase;color:var(--faint);display:block;margin-top:3px}
nav.toc{margin:0 0 60px;padding:18px 0 0;font-family:Archivo,sans-serif;font-size:13.5px}
nav.toc ol{margin:0;padding:0;list-style:none;columns:2;column-gap:44px}
nav.toc li{margin-bottom:7px;break-inside:avoid}
nav.toc a{color:var(--muted);text-decoration:none;display:flex;gap:10px}
nav.toc a:hover{color:var(--accent)}
nav.toc .n{color:var(--faint);font-variant-numeric:tabular-nums;min-width:18px}
section{margin:0 0 58px;scroll-margin-top:20px}
h2{font-size:12px;font-weight:700;letter-spacing:.15em;text-transform:uppercase;
color:var(--ink);margin:0 0 22px;padding-bottom:10px;border-bottom:1px solid var(--rule);
display:flex;gap:14px;align-items:baseline}
h2 .n{color:var(--accent);font-variant-numeric:tabular-nums}
h3{font-family:Archivo,sans-serif;font-size:19px;font-weight:600;color:var(--ink);
margin:32px 0 12px;letter-spacing:-.012em}
h4{font-family:Archivo,sans-serif;font-size:14px;font-weight:600;color:var(--ink);
margin:24px 0 8px}
.prose{max-width:67ch}
p{margin:0 0 15px}
strong{color:var(--ink);font-weight:600}
ul,ol{margin:0 0 16px;padding-left:22px;max-width:67ch}
li{margin-bottom:8px}li::marker{color:var(--faint)}
code{font-size:.83em;background:var(--sunk);padding:1.5px 5px;border-radius:2px;
color:var(--ink)}
figure{margin:26px 0 24px;border:1px solid var(--line);background:var(--surface)}
figure .svgbox{overflow-x:auto;padding:24px 24px 6px}
figure svg{display:block;max-width:100%;height:auto;min-width:540px;color:var(--body)}
figcaption{font-size:12.5px;line-height:1.5;color:var(--muted);padding:0 24px 16px;
max-width:80ch}
figcaption b{color:var(--ink);font-weight:600}
.s-ink{fill:var(--ink)}.s-body{fill:var(--body)}.s-faint{fill:var(--faint)}
.s-acc{fill:var(--accent)}.s-good{fill:var(--good)}.s-bad{fill:var(--bad)}
.st-line{stroke:var(--rule);fill:none}.st-ink{stroke:var(--ink);fill:none}
.st-good{stroke:var(--good);fill:none}.st-bad{stroke:var(--bad);fill:none}
.bx{fill:var(--surface);stroke:var(--rule)}
.bx-sunk{fill:var(--sunk);stroke:var(--rule)}
.bx-good{fill:var(--good-soft);stroke:var(--good)}
.bx-bad{fill:var(--bad-soft);stroke:var(--bad)}
.bx-acc{fill:var(--accent-soft);stroke:var(--accent)}
text{font-family:Archivo,system-ui,sans-serif;font-size:12.5px}
text.m{font-family:"JetBrains Mono",monospace;font-size:10.5px}
text.lb{font-size:10px;letter-spacing:.09em;text-transform:uppercase}
text.big{font-size:13.5px;font-weight:600}
.tw{overflow-x:auto;border:1px solid var(--line);margin:0 0 20px}
table{width:100%;border-collapse:collapse;font-size:15.5px;background:var(--surface)}
th{text-align:left;font-size:10.5px;font-weight:700;letter-spacing:.11em;
text-transform:uppercase;color:var(--faint);padding:11px 15px;
border-bottom:1px solid var(--rule);white-space:nowrap}
td{padding:11px 15px;border-bottom:1px solid var(--line);vertical-align:top}
tr:last-child td{border-bottom:0}
.num{font-variant-numeric:tabular-nums;text-align:right}
td.lbl{font-weight:600;color:var(--ink);font-family:Archivo,sans-serif;font-size:14.5px}
.up{color:var(--good);font-weight:600}.down{color:var(--bad);font-weight:600}
.flat{color:var(--faint)}
.rule{border-left:3px solid var(--accent);background:var(--accent-soft);
padding:15px 20px;margin:0 0 20px;font-size:16.5px}
.rule b{color:var(--ink)}
.note{border-left:3px solid var(--rule);padding:13px 20px;margin:0 0 20px;
font-size:15.5px;color:var(--muted)}
.note b{color:var(--ink)}
.warn{border-left:3px solid var(--bad);background:var(--bad-soft);padding:14px 20px;
margin:0 0 20px;font-size:16px}
.warn b{color:var(--ink)}
pre.sh{background:var(--sunk);border:1px solid var(--line);padding:15px 18px;
overflow-x:auto;font-size:12.5px;line-height:1.65;margin:0 0 20px;white-space:pre}
pre.sh b{color:var(--ink);font-weight:700}
pre.sh i{color:var(--accent);font-style:normal}
.cite{font-family:"JetBrains Mono",monospace;font-size:.8em;color:var(--muted);
white-space:nowrap}
.cc{font-variant-numeric:tabular-nums;font-weight:500;color:var(--good)}
.tag{display:inline-block;font-size:9.5px;font-weight:700;letter-spacing:.09em;
text-transform:uppercase;padding:3px 8px;line-height:1.4;white-space:nowrap}
.tag.keep{background:var(--good);color:var(--surface)}
.tag.kill{border:1px dashed var(--bad);color:var(--bad)}
.tag.open{border:1px solid var(--accent);color:var(--accent)}
ol.dec{counter-reset:d;list-style:none;padding:0;max-width:none}
ol.dec li{counter-increment:d;position:relative;padding:0 0 15px 46px;margin:0 0 15px;
border-bottom:1px solid var(--line);font-size:16px}
ol.dec li:last-child{border-bottom:0}
ol.dec li::before{content:counter(d);position:absolute;left:0;top:1px;
font-family:"JetBrains Mono",monospace;font-size:12px;font-weight:700;
color:var(--chip);background:var(--ink);width:26px;height:22px;display:flex;
align-items:center;justify-content:center;font-variant-numeric:tabular-nums}
ol.dec b{color:var(--ink)}
.foot{border-top:1px solid var(--rule);padding-top:22px;margin-top:44px;font-size:14px;
color:var(--muted);font-family:Archivo,sans-serif}
:focus-visible{outline:2px solid var(--accent);outline-offset:3px}
@media (max-width:640px){nav.toc ol{columns:1}.stat{flex-basis:50%}}
"""

ARROW = ('<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
         'markerHeight="7" orient="auto-start-reverse">'
         '<path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker>'
         '<marker id="ag" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
         'markerHeight="7" orient="auto-start-reverse">'
         '<path d="M0,0 L10,5 L0,10 z" fill="var(--good)"/></marker>'
         '<marker id="ab" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
         'markerHeight="7" orient="auto-start-reverse">'
         '<path d="M0,0 L10,5 L0,10 z" fill="var(--bad)"/></marker></defs>')


def fig(svg, cap, label):
    return (f'<figure><div class="svgbox"><svg viewBox="{svg[0]}" role="img" '
            f'aria-label="{label}">{ARROW}{svg[1]}</svg></div>'
            f'<figcaption>{cap}</figcaption></figure>')


# ---------------------------------------------------------------- diagrams
D_SHAPE = ("0 0 880 250", '''
<text x="10" y="20" class="lb s-faint">Three machines. Only one runs per question.</text>
<rect x="10" y="34" width="250" height="86" class="bx-sunk"/>
<text x="135" y="56" class="big s-ink" text-anchor="middle">BUILD</text>
<text x="135" y="76" class="s-body" text-anchor="middle">offline, once per corpus</text>
<text x="135" y="94" class="m s-faint" text-anchor="middle">~1s + 17 calls for the map</text>
<text x="135" y="110" class="m s-faint" text-anchor="middle">deterministic, re-runnable</text>

<rect x="315" y="34" width="250" height="86" class="bx-acc"/>
<text x="440" y="56" class="big s-ink" text-anchor="middle">SERVE</text>
<text x="440" y="76" class="s-body" text-anchor="middle">per question</text>
<text x="440" y="94" class="m s-faint" text-anchor="middle">2 calls, ~39s, ~12k tokens</text>
<text x="440" y="110" class="m s-faint" text-anchor="middle">flat in corpus size</text>

<rect x="620" y="34" width="250" height="86" class="bx-good"/>
<text x="745" y="56" class="big s-ink" text-anchor="middle">MEASURE</text>
<text x="745" y="76" class="s-body" text-anchor="middle">frozen, reused across arms</text>
<text x="745" y="94" class="m s-faint" text-anchor="middle">20 calls, once</text>
<text x="745" y="110" class="m s-faint" text-anchor="middle">expires as corpus grows</text>

<line x1="262" y1="77" x2="311" y2="77" class="st-line" marker-end="url(#a)"/>
<text x="286" y="68" class="lb s-faint" text-anchor="middle">index</text>
<line x1="618" y1="77" x2="569" y2="77" class="st-good" marker-end="url(#ag)"/>
<text x="594" y="68" class="lb s-good" text-anchor="middle">scores</text>

<line x1="10" y1="150" x2="870" y2="150" class="st-line" stroke-dasharray="2 4"/>
<text x="10" y="176" class="lb s-faint">What each one owns</text>
<text x="10" y="198" class="s-body">BUILD  verbatim chunks, BM25 index, navigation map, exemplar set</text>
<text x="10" y="218" class="s-body">SERVE  query fan-out, retrieval, relevance gate, generation</text>
<text x="10" y="238" class="s-body">MEASURE  exhaustive relevance labels, evidence recall, abstention, claim judge</text>
''')

D_BUILD = ("0 0 880 300", '''
<text x="10" y="20" class="lb s-faint">Build, and the one place an LLM touches the store</text>
<rect x="10" y="34" width="132" height="56" class="bx-sunk"/>
<text x="76" y="56" class="s-ink" text-anchor="middle">.json3</text>
<text x="76" y="74" class="m s-body" text-anchor="middle">word timestamps</text>
<line x1="144" y1="62" x2="188" y2="62" class="st-line" marker-end="url(#a)"/>
<text x="166" y="52" class="lb s-faint" text-anchor="middle">dedup</text>

<rect x="190" y="34" width="150" height="56" class="bx"/>
<text x="265" y="56" class="s-ink" text-anchor="middle">word stream</text>
<text x="265" y="74" class="m s-body" text-anchor="middle">69,157 words</text>
<line x1="342" y1="62" x2="386" y2="62" class="st-line" marker-end="url(#a)"/>
<text x="364" y="52" class="lb s-faint" text-anchor="middle">window</text>

<rect x="388" y="34" width="160" height="56" class="bx-good"/>
<text x="468" y="52" class="big s-ink" text-anchor="middle">292 chunks</text>
<text x="468" y="70" class="m s-body" text-anchor="middle">300 words / 60 overlap</text>
<text x="468" y="84" class="lb s-good" text-anchor="middle">VERBATIM</text>

<line x1="550" y1="50" x2="596" y2="50" class="st-line" marker-end="url(#a)"/>
<line x1="550" y1="74" x2="596" y2="74" class="st-line" marker-end="url(#a)"/>
<line x1="468" y1="92" x2="468" y2="150" class="st-line"/>
<line x1="468" y1="150" x2="596" y2="150" class="st-line" marker-end="url(#a)"/>

<rect x="598" y="32" width="130" height="36" class="bx"/>
<text x="663" y="55" class="s-ink" text-anchor="middle">BM25 index</text>
<rect x="598" y="58" width="130" height="34" class="bx"/>
<text x="663" y="80" class="s-ink" text-anchor="middle">exemplars</text>
<rect x="598" y="132" width="272" height="38" class="bx-acc"/>
<text x="734" y="149" class="s-ink" text-anchor="middle">map.json &mdash; 17 takes, 302 terms</text>
<text x="734" y="164" class="m s-acc" text-anchor="middle">1 LLM call per take. the only LLM in build.</text>

<line x1="10" y1="196" x2="870" y2="196" class="st-line" stroke-dasharray="2 4"/>
<text x="10" y="222" class="lb s-good">Kept verbatim</text>
<text x="10" y="242" class="s-body">chunks carry take_id + start/end ms. answers cite times, so text must be his.</text>
<text x="10" y="268" class="lb s-acc">Derived, and fenced</text>
<text x="10" y="288" class="s-body">the map is navigation only: never quoted, never cited, never a source for an answer.</text>
''')

D_FACTOR = ("0 0 880 268", '''
<text x="10" y="20" class="lb s-faint">Three axes, three different objects, three different homes</text>
<rect x="10" y="36" width="272" height="118" class="bx-good"/>
<text x="146" y="60" class="big s-ink" text-anchor="middle">IDENTITY</text>
<text x="146" y="80" class="s-body" text-anchor="middle">true of him regardless</text>
<text x="146" y="98" class="s-body" text-anchor="middle">of the question</text>
<text x="146" y="122" class="m s-faint" text-anchor="middle">small, saturates fast</text>
<text x="146" y="140" class="lb s-good" text-anchor="middle">ALWAYS PRESENT</text>

<rect x="304" y="36" width="272" height="118" class="bx-acc"/>
<text x="440" y="60" class="big s-ink" text-anchor="middle">TOPIC</text>
<text x="440" y="80" class="s-body" text-anchor="middle">true of him about</text>
<text x="440" y="98" class="s-body" text-anchor="middle">this subject</text>
<text x="440" y="122" class="m s-faint" text-anchor="middle">large, grows forever</text>
<text x="440" y="140" class="lb s-acc" text-anchor="middle">RETRIEVED</text>

<rect x="598" y="36" width="272" height="118" class="bx"/>
<text x="734" y="60" class="big s-ink" text-anchor="middle">SITUATION</text>
<text x="734" y="80" class="s-body" text-anchor="middle">true of the moment,</text>
<text x="734" y="98" class="s-body" text-anchor="middle">not of him</text>
<text x="734" y="122" class="m s-faint" text-anchor="middle">one observed value</text>
<text x="734" y="140" class="lb s-faint" text-anchor="middle">CONFIG, NOT LEARNED</text>

<text x="10" y="190" class="lb s-faint">Why the homes are forced, not chosen</text>
<text x="10" y="212" class="s-body">Put identity in the vector store and his beliefs become conditional on lexical overlap:</text>
<text x="10" y="230" class="s-body">the clone then contradicts itself between two phrasings of the same question.</text>
<text x="10" y="256" class="s-body">Situation has exactly one observed value in a monologue corpus, so it cannot be estimated.</text>
''')

D_SERVE = ("0 0 880 330", '''
<text x="10" y="20" class="lb s-faint">Serve. Two model calls, one fan-out, one gate.</text>
<rect x="10" y="34" width="104" height="44" class="bx-sunk"/>
<text x="62" y="61" class="s-ink" text-anchor="middle">question</text>
<line x1="116" y1="56" x2="164" y2="56" class="st-line" marker-end="url(#a)"/>

<rect x="166" y="30" width="186" height="52" class="bx-acc"/>
<text x="259" y="50" class="big s-ink" text-anchor="middle">call 1: plan</text>
<text x="259" y="68" class="m s-body" text-anchor="middle">writes 3-4 queries in HIS words</text>

<line x1="354" y1="56" x2="398" y2="56" class="st-good" marker-end="url(#ag)"/>
<text x="376" y="46" class="lb s-good" text-anchor="middle">fan-out</text>

<rect x="400" y="16" width="150" height="22" class="bx"/>
<text x="475" y="32" class="m s-body" text-anchor="middle">"male isolation biology"</text>
<rect x="400" y="44" width="150" height="22" class="bx"/>
<text x="475" y="60" class="m s-body" text-anchor="middle">"male loneliness friends"</text>
<rect x="400" y="72" width="150" height="22" class="bx"/>
<text x="475" y="88" class="m s-body" text-anchor="middle">"shoulder to shoulder"</text>

<line x1="552" y1="56" x2="596" y2="56" class="st-line" marker-end="url(#a)"/>
<text x="574" y="46" class="lb s-faint" text-anchor="middle">BM25</text>
<rect x="598" y="30" width="120" height="52" class="bx"/>
<text x="658" y="50" class="s-ink" text-anchor="middle">union</text>
<text x="658" y="68" class="m s-body" text-anchor="middle">~39 spans, +/-1 ctx</text>

<line x1="720" y1="56" x2="762" y2="56" class="st-line" marker-end="url(#a)"/>
<rect x="764" y="30" width="106" height="52" class="bx-good"/>
<text x="817" y="50" class="big s-ink" text-anchor="middle">gate</text>
<text x="817" y="68" class="m s-body" text-anchor="middle">keep / drop</text>

<line x1="817" y1="84" x2="817" y2="122" class="st-bad" marker-end="url(#ab)"/>
<text x="817" y="104" class="lb s-bad" text-anchor="middle">empty</text>
<line x1="764" y1="56" x2="700" y2="56" class="st-line"/>
<line x1="658" y1="84" x2="658" y2="150" class="st-good"/>
<line x1="658" y1="150" x2="404" y2="150" class="st-good" marker-end="url(#ag)"/>
<text x="530" y="142" class="lb s-good" text-anchor="middle">surviving spans</text>

<rect x="240" y="128" width="164" height="46" class="bx-acc"/>
<text x="322" y="147" class="big s-ink" text-anchor="middle">call 2: answer</text>
<text x="322" y="164" class="m s-body" text-anchor="middle">+ refs + grounding label</text>

<rect x="700" y="128" width="170" height="46" class="bx-bad"/>
<text x="785" y="147" class="big s-ink" text-anchor="middle">refuse + redirect</text>
<text x="785" y="164" class="m s-body" text-anchor="middle">names coverage, from map</text>

<line x1="10" y1="200" x2="870" y2="200" class="st-line" stroke-dasharray="2 4"/>
<text x="10" y="226" class="lb s-faint">The one structural commitment</text>
<text x="10" y="248" class="s-body">The question never passes THROUGH retrieval on its way to the model. The index sits</text>
<text x="10" y="266" class="s-body">beside the model as a resource it queries, which is what makes one-shot and multi-turn</text>
<text x="10" y="284" class="s-body">the same shape, and leaves the model deciding how much it needs.</text>
<text x="10" y="312" class="s-body">Cost is flat in corpus size: ~12k tokens per query whether the archive is 17 takes or 3,000.</text>
''')

D_VERIFY = ("0 0 880 300", '''
<text x="10" y="20" class="lb s-faint">Verification. Two failure modes, measured separately.</text>
<rect x="10" y="34" width="200" height="62" class="bx-good"/>
<text x="110" y="55" class="big s-ink" text-anchor="middle">ORACLE</text>
<text x="110" y="73" class="s-body" text-anchor="middle">whole archive read</text>
<text x="110" y="89" class="m s-faint" text-anchor="middle">210 answer + 259 support</text>

<rect x="10" y="118" width="200" height="52" class="bx-acc"/>
<text x="110" y="139" class="big s-ink" text-anchor="middle">ONE ARM</text>
<text x="110" y="157" class="m s-body" text-anchor="middle">what retrieval surfaced</text>

<line x1="212" y1="66" x2="286" y2="96" class="st-line" marker-end="url(#a)"/>
<line x1="212" y1="144" x2="286" y2="118" class="st-line" marker-end="url(#a)"/>

<rect x="288" y="86" width="192" height="46" class="bx"/>
<text x="384" y="105" class="big s-ink" text-anchor="middle">span overlap</text>
<text x="384" y="122" class="m s-body" text-anchor="middle">mechanical, deterministic</text>

<line x1="482" y1="109" x2="530" y2="109" class="st-line" marker-end="url(#a)"/>
<rect x="532" y="84" width="164" height="50" class="bx-good"/>
<text x="614" y="104" class="big s-ink" text-anchor="middle">MISSED</text>
<text x="614" y="122" class="m s-body" text-anchor="middle">evidence recall</text>

<rect x="288" y="164" width="192" height="46" class="bx"/>
<text x="384" y="183" class="big s-ink" text-anchor="middle">claim judge</text>
<text x="384" y="200" class="m s-body" text-anchor="middle">supported: yes / no</text>
<line x1="110" y1="172" x2="110" y2="187" class="st-line"/>
<line x1="110" y1="187" x2="286" y2="187" class="st-line" marker-end="url(#a)"/>
<line x1="482" y1="187" x2="530" y2="187" class="st-line" marker-end="url(#a)"/>
<rect x="532" y="162" width="164" height="50" class="bx-bad"/>
<text x="614" y="182" class="big s-ink" text-anchor="middle">UNGROUNDED</text>
<text x="614" y="200" class="m s-body" text-anchor="middle">grounding rate</text>

<line x1="10" y1="238" x2="870" y2="238" class="st-line" stroke-dasharray="2 4"/>
<text x="10" y="262" class="s-body">The judge is never asked whether an answer is good. Only "is this claim in these</text>
<text x="10" y="280" class="s-body">excerpts, yes or no". A known-biased judge kept on the one task it is reliable at.</text>
''')

D_2X2 = ("0 0 880 250", f'''
<text x="10" y="20" class="lb s-faint">The 2x2 that separated retrieval from wording</text>
<text x="196" y="52" class="lb s-faint" text-anchor="middle">prompt v1</text>
<text x="356" y="52" class="lb s-faint" text-anchor="middle">prompt v2</text>
<text x="120" y="88" class="s-body" text-anchor="end">one query</text>
<text x="120" y="140" class="s-body" text-anchor="end">model writes queries</text>

<rect x="136" y="64" width="120" height="34" class="bx"/>
<text x="196" y="87" class="big s-ink" text-anchor="middle">{A["r0u0"]["connected"]:.2f}</text>
<rect x="296" y="64" width="120" height="34" class="bx"/>
<text x="356" y="87" class="big s-ink" text-anchor="middle">{A["r0u1"]["connected"]:.2f}</text>
<rect x="136" y="116" width="120" height="34" class="bx-good"/>
<text x="196" y="139" class="big s-ink" text-anchor="middle">{A["r1u0"]["connected"]:.2f}</text>
<rect x="296" y="116" width="120" height="34" class="bx-good"/>
<text x="356" y="139" class="big s-ink" text-anchor="middle">{A["r1u1"]["connected"]:.2f}</text>

<line x1="196" y1="104" x2="196" y2="112" class="st-good" marker-end="url(#ag)"/>
<line x1="356" y1="104" x2="356" y2="112" class="st-good" marker-end="url(#ag)"/>
<line x1="262" y1="81" x2="290" y2="81" class="st-line" marker-end="url(#a)"/>
<line x1="262" y1="133" x2="290" y2="133" class="st-line" marker-end="url(#a)"/>

<rect x="470" y="60" width="400" height="94" class="bx-sunk"/>
<text x="490" y="84" class="lb s-faint">main effect</text>
<text x="700" y="84" class="lb s-faint" text-anchor="end">size</text>
<text x="490" y="108" class="s-ink">retrieval</text>
<text x="700" y="108" class="big s-good" text-anchor="end">+0.351</text>
<text x="490" y="130" class="s-ink">prompt</text>
<text x="700" y="130" class="big s-faint" text-anchor="end">+0.008</text>
<text x="490" y="150" class="s-ink">interaction</text>
<text x="700" y="150" class="big s-faint" text-anchor="end">+0.017</text>

<text x="10" y="196" class="s-body">Run as one arm, the two changes confound and the intuitive story wins: "the better</text>
<text x="10" y="214" class="s-body">prompt helped". Crossed, the prompt is worth nothing on recall and retrieval is</text>
<text x="10" y="232" class="s-body">the entire effect. Only the 2x2 could say which edge moved.</text>
''')


def main():
    P = ["<title>Grounded Answering Pipeline</title>",
         '<link rel="preconnect" href="https://fonts.googleapis.com">',
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
         'family=Archivo:wght@500;600;700&family=JetBrains+Mono:wght@400;500;700&'
         'family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&display=swap">',
         f"<style>{CSS}</style>", '<div class="wrap">']

    P += ['<header class="mast">',
          '<p class="eyebrow">knowledge + style engine &middot; implementation record</p>',
          "<h1>Answering From an Archive</h1>",
          '<p class="lede">A working prototype that answers questions as one creator, '
          'grounded only in his transcripts, with every claim traceable to a timestamp. '
          'This is what was built, what was measured, and which of the obvious ideas '
          'the measurements <b>killed</b>.</p>',
          f'<p class="meta">base model <code>claude-opus-5</code> via Claude Code CLI '
          f'2.1.263 &middot; corpus <code>{S["index"]["subject"]}</code>, '
          f'{IX["n_takes"]} takes, {IX["n_words"]:,} words &middot; '
          f'index <code>{IX["index_version"]}</code> &middot; reproduce with '
          f'<code>make all</code></p>',
          '<div class="stats">']
    for v, l in [(IX["n_chunks"], "chunks"), (S["oracle"]["n"], "probe questions"),
                 (S["oracle"]["ans"], "oracle spans"),
                 (f'{A["r1u1"]["connected"]:.2f}', "recall, connected"),
                 (f'{S["agent"]["mean_queries"]:.1f}', "queries / question"),
                 (f'{S["agent"]["mean_s"]:.0f}s', "per answer")]:
        P.append(f'<div class="stat"><span class="v">{v}</span>'
                 f'<span class="l">{l}</span></div>')
    P += ["</div></header>"]

    P += ['<nav class="toc"><ol>']
    toc = ["The shape", "The base model", "Memory: how the store is built",
           "The factorization", "The harness", "Prompts and redirection",
           "The verification pipeline", "Experiments", "What the experiments say",
           "Related work", "Worth exploring", "Open questions"]
    for i, t in enumerate(toc, 1):
        P.append(f'<li><a href="#s{i}"><span class="n">{i}</span> {t}</a></li>')
    P += ["</ol></nav>"]

    # 1 shape
    P += ['<section id="s1"><h2><span class="n">01</span> The shape</h2>',
          '<p class="prose">Three machines with different lifetimes. Conflating them is '
          'where this kind of system usually goes wrong: the expensive thing gets '
          'rebuilt per question, or the cheap thing gets treated as ground truth.</p>',
          fig(D_SHAPE, "<b>Build runs once per corpus, serve runs per question, "
              "measure runs once and is reused across every experiment.</b> Nothing "
              "written at serve time re-enters the store: in this project the model "
              "does not change and the archive does not grow from conversations.",
              "Three machines: build offline per corpus, serve per question, measure "
              "once and reuse across arms."),
          "</section>"]

    # 2 base model
    P += ['<section id="s2"><h2><span class="n">02</span> The base model, and why</h2>',
          '<div class="tw"><table><thead><tr><th>Role</th><th>Model</th>'
          '<th>Why this one</th></tr></thead><tbody>',
          '<tr><td class="lbl">query planning</td><td class="mono">claude-opus-5</td>'
          '<td>writes 3-4 search queries in his vocabulary. This is where the entire '
          'measured gain lives, so it gets the strongest model available</td></tr>',
          '<tr><td class="lbl">generation</td><td class="mono">claude-opus-5</td>'
          '<td>must hold ~12k tokens of transcript and stay in voice without '
          'inventing. Same model, second call</td></tr>',
          '<tr><td class="lbl">relevance gate</td><td class="mono">claude-opus-5</td>'
          '<td>keep/drop over retrieved spans. A cheaper model is the obvious '
          'substitution and is untested</td></tr>',
          '<tr><td class="lbl">oracle + judge</td><td class="mono">claude-opus-5</td>'
          '<td><b>this is a known weakness.</b> The judge shares a family with the '
          'generator, and self-preference in LLM judges is measured and real</td></tr>',
          "</tbody></table></div>",
          '<div class="note prose"><b>No API key, no SDK, no vector database, no '
          'framework.</b> Generation shells out to the <code>claude</code> CLI over '
          'stdin; retrieval is BM25 in ~40 lines of stdlib Python. At 292 chunks a '
          'vector store solves a problem that does not exist, and a framework would '
          'hide exactly the fields the scoreboard needs: the query the model wrote, '
          'the chunk ids returned, and their scores.</div>',
          '<div class="warn prose"><b>The judge sharing a family with the generator is '
          'the single largest threat to every number here.</b> The mitigation is to '
          'never ask it whether an answer is good, only whether a specific claim '
          'appears in a specific excerpt. That is close to mechanical. It is a '
          'mitigation, not a fix, and an independent judge is the obvious next '
          'control.</div>',
          "</section>"]

    # 3 memory
    P += ['<section id="s3"><h2><span class="n">03</span> Memory: how the store is '
          'built</h2>',
          '<p class="prose">"Memory" here is not the agent-memory sense. Nothing '
          'accumulates. It is a static, versioned index over a fixed archive, and the '
          'whole build is deterministic apart from one LLM pass.</p>',
          fig(D_BUILD, "<b>The build is one transformation and three products.</b> "
              "Chunks stay verbatim because answers cite timestamps, so the text has "
              "to be his. The map is the only derived artifact and it is fenced: "
              "navigation only, never quoted, never cited, never a source.",
              "Build pipeline: json3 subtitles to deduplicated word stream to 292 "
              "verbatim chunks, feeding a BM25 index, an exemplar set, and an "
              "LLM-generated navigation map."),
          '<h3>Decisions, and what earned them</h3>',
          '<div class="tw"><table><thead><tr><th>Decision</th><th>Choice</th>'
          '<th>Evidence</th></tr></thead><tbody>',
          f'<tr><td class="lbl">chunk unit</td><td>fixed {IX["words_per_chunk"]} words, '
          f'{IX["overlap_words"]} overlap</td><td><span class="cite">2410.13070</span> '
          f'NAACL 2025 Findings <span class="cc">69</span>: fixed-size beat semantic '
          f'chunking on retrieval, evidence and answer generation, overhead '
          f'unjustified</td></tr>',
          '<tr><td class="lbl">chunk content</td><td>verbatim + take_id + ms</td>'
          '<td><span class="cite">2603.26680</span> AlpsBench, SIGIR: models fail to '
          'reliably extract latent traits, so extraction loses what it cannot '
          'recover</td></tr>',
          '<tr><td class="lbl">transcript cleanup</td><td>none</td>'
          '<td><span class="cite">2311.07564</span> TACL: normalized lowercase '
          'transcripts work as well or better. Do not prettify ASR</td></tr>',
          '<tr><td class="lbl">rolling caption dedup</td><td>required</td>'
          '<td>raw .vtt inflates word counts ~4x: 307,912 apparent vs 69,157 '
          'real</td></tr>',
          '<tr><td class="lbl">retrieval</td><td>BM25, no embeddings</td>'
          '<td>at 292 chunks, brute-force lexical is microseconds. Embeddings are the '
          'obvious next lever, untested</td></tr>',
          "</tbody></table></div>", "</section>"]

    # 4 factorization
    P += ['<section id="s4"><h2><span class="n">04</span> The factorization</h2>',
          '<p class="prose">The usual split is knowledge versus style. That is the '
          'wrong cut for deciding <em>where things live</em>. The cut that works is by '
          'what a thing is conditional on.</p>',
          fig(D_FACTOR, "<b>Identity, topic and situation are different objects with "
              "different growth curves, and that forces where each one lives.</b> "
              "Identity saturates and must always be present. Topic grows forever and "
              "must be retrieved. Situation has one observed value in a monologue "
              "corpus, so it is a setting rather than something learned.",
              "The three-axis factorization: identity always present, topic retrieved, "
              "situation set by config."),
          '<h3>What is actually implemented on each axis</h3>',
          '<div class="tw"><table><thead><tr><th>Axis</th><th>Implementation</th>'
          '<th>State</th></tr></thead><tbody>',
          '<tr><td class="lbl">topic</td><td>BM25 over verbatim chunks, model-written '
          'queries, union, &plusmn;1 chunk expansion</td>'
          '<td><span class="tag keep">measured, works</span></td></tr>',
          '<tr><td class="lbl">identity</td><td>8 fixed exemplar chunks in every '
          'prompt, auto-picked by <code>build.py</code></td>'
          '<td><span class="tag open">weakest part</span></td></tr>',
          '<tr><td class="lbl">situation</td><td>one line of config: length and '
          'register</td><td><span class="tag keep">trivial, correct</span></td></tr>',
          "</tbody></table></div>",
          '<div class="warn prose"><b>The identity axis is the weakest thing in the '
          'system and it has a measured symptom.</b> The exemplars are chunk #1 from '
          'each take, chosen by nothing. The result is a template: the same stock move '
          '("more than 50% of my job is figuring out what is actually going on") '
          'appeared in <b>9 of 30</b> answers, and "the AI does not ask questions" in '
          '<b>6</b>. When retrieval is thin the model reaches for whatever is standing '
          'in the prompt. Curating this set is the cheapest unexplored '
          'improvement.</div>',
          '<div class="note prose"><b>Situation is not underbuilt, it is '
          'unidentifiable.</b> The corpus is 17 broadcast monologues: one value of the '
          'situation variable. You cannot estimate an effect from data in which it '
          'never varies. Fixing this is a data-collection decision, not a modelling '
          'one: ingest interview or stream material and the axis becomes '
          'estimable.</div>',
          "</section>"]

    # 5 harness
    P += ['<section id="s5"><h2><span class="n">05</span> The harness</h2>',
          fig(D_SERVE, "<b>Two model calls per question, with a fan-out between "
              "them.</b> The first call plans, the second answers. The question never "
              "passes through retrieval on its way to the model: the index sits beside "
              "the model as a resource it queries, which is what makes one-shot and "
              "multi-turn the same shape.",
              "Serve path: question to a planning call that writes several queries, "
              "BM25 fan-out and union, a relevance gate, then either an answer call or "
              "a refuse-and-redirect path."),
          '<h3>The tool surface</h3>',
          '<pre class="sh"><b>search_archive(query)</b> -> [ {take_id, timestamp, '
          'score, text} ]\n\n'
          'Emitted as a line:  <i>SEARCH: male isolation biology withdraw</i>\n\n'
          'Not tools, deliberately kept in the harness:\n'
          '  expansion   each hit is widened by +/-1 chunk. no model judgement needed\n'
          '  reranking   same reason\n'
          '  the gate    runs after the fan-out, not on demand</pre>',
          '<p class="prose">Anything that needs no judgement stays out of the model\'s '
          'hands. Every round trip is latency and a chance to vary.</p>',
          '<h3>Two things enforced in code, not asked for in the prompt</h3>',
          "<ol class='dec prose'>",
          '<li><b>At least one search, always.</b> Left free, the model skipped '
          'searching on questions it assumed were out of domain and then answered from '
          'its own knowledge, volunteering that a viola is "bigger and tuned lower". '
          'That is the exact failure the system exists to prevent. It must look before '
          'it declines.</li>',
          '<li><b>Retry with backoff, and loud failure.</b> Seven consecutive CLI '
          'failures at the head of one arm were caught as exceptions and written as '
          'rows with no spans, which scored as recall 0.00. The arm read as a '
          'catastrophic regression of the feature under test. <b>A silent failure that '
          'scores like a result is worse than a crash.</b></li>', "</ol>",
          f'<p class="prose">Cost per question: 2 model calls (3 with the gate), '
          f'~{S["agent"]["mean_queries"]:.0f} queries, ~12k tokens, '
          f'~{S["agent"]["mean_s"]:.0f}s. <strong>Flat in corpus size.</strong> '
          f'Whole-archive context would be ~1.6M tokens per query at 300 takes, '
          f'linear in the archive and landing the relevant span in the degraded middle '
          f'of the window (<span class="cite">2307.03172</span>, TACL, '
          f'<span class="cc">4899</span>).</p>',
          "</section>"]

    # 6 prompts
    P += ['<section id="s6"><h2><span class="n">06</span> Prompts, and how questions '
          'get redirected</h2>',
          '<p class="prose">Four prompts doing four jobs. The split that matters is '
          'which are allowed to be creative.</p>',
          '<div class="tw"><table><thead><tr><th>Prompt</th><th>Job</th>'
          '<th>Creative?</th></tr></thead><tbody>',
          '<tr><td class="lbl">persona</td><td>answer as him, from retrieved '
          'evidence</td><td>yes, within the evidence</td></tr>',
          '<tr><td class="lbl">tools</td><td>decide what to search for</td>'
          '<td>yes. this is where the gain came from</td></tr>',
          '<tr><td class="lbl">gate</td><td>is this span about the question</td>'
          '<td><b>no.</b> membership only</td></tr>',
          '<tr><td class="lbl">oracle + judge</td><td>ground truth and claim '
          'checking</td><td><b>no.</b> membership only</td></tr>',
          "</tbody></table></div>",
          '<h3>The persona prompt, verbatim</h3>',
          '<pre class="sh">You are answering as the speaker in these transcripts. Not '
          'describing\nhim, not summarising him. You are him, replying to one '
          'person.\n\n<b>Rules that do not bend:</b>\n'
          '- Never say \'he\' about yourself. You are the speaker. If you have\n'
          '  nothing, say \'I haven\'t covered that\', never \'he hasn\'t covered '
          'that\'.\n'
          '- Everything you assert as your position must be in what you retrieved.\n'
          '  Where the archive is silent, say so plainly and stop.\n'
          '- You may reason beyond the archive, but say you are doing it, in your\n'
          '  own voice, without hedging language a person would not use out loud.\n'
          '- Do not reach for the same stock example every time.</pre>',
          '<p class="prose"><strong>Every rule exists because a run violated it.</strong> '
          'The first rule is there because three answers opened "He hasn\'t covered '
          'this" and discussed him in the third person. The last is there because of '
          'the template effect. A prompt written before the failures would not have '
          'contained any of them.</p>',
          '<div class="note prose"><b>His name is a flag, not a given.</b> Naming him '
          'invokes whatever the model already believes about him, which is unmeasured '
          'and uncontrolled. Keeping it as a switch is what makes the bare-persona '
          'control meaningful: that arm measures what is already in the weights, and '
          'it turned out to have his voice but not his content.</div>',
          '<h3>Redirection: what happens when he has not covered it</h3>',
          '<p class="prose">The output carries a self-reported grounding label &mdash; '
          '<code>direct</code>, <code>extended</code>, or <code>none</code> &mdash; '
          'and on every dangerous case that label beat every lexical signal. When '
          'retrieval says "direct" and the model has read the spans and says "none", '
          'the model is right.</p>',
          '<pre class="sh"><b>gate returns empty</b>\n'
          '   -> "I haven\'t covered that", in his voice\n'
          '   -> then name one or two things he HAS covered, from the map\n'
          '   -> never quote the map, never cite it, and do not stretch\n\n'
          '<b>Measured example, the marathon question:</b>\n'
          '   searched 3 ways, 0 spans survived the gate\n'
          '   named deconditioning and controlled-motivation as adjacent\n'
          '   closed with "For the actual training structure, ask a running coach."</pre>',
          '<p class="prose">This replaced the earlier behaviour, which always found '
          'something to extend from and reached for the same stock move each time. '
          'Naming specific coverage is what makes the refusal informative rather than '
          'a brush-off.</p>',
          "</section>"]

    # 7 verification
    P += ['<section id="s7"><h2><span class="n">07</span> The verification '
          'pipeline</h2>',
          '<p class="prose">Two failure modes, deliberately never merged into one '
          'score.</p>',
          '<div class="tw"><table><thead><tr><th>Failure</th><th>Meaning</th>'
          '<th>How measured</th></tr></thead><tbody>',
          '<tr><td class="lbl">MISSED</td><td>the archive has the answer, retrieval '
          'did not surface it</td><td>mechanical. timestamp overlap between oracle '
          'spans and retrieved spans. no model opinion</td></tr>',
          '<tr><td class="lbl">UNGROUNDED</td><td>the answer asserts something the '
          'evidence does not support</td><td>a judge answering only "is this claim in '
          'these excerpts, yes or no"</td></tr>',
          "</tbody></table></div>",
          fig(D_VERIFY, "<b>The oracle is computed without any retriever in the "
              "loop</b>, so it is what retrieval should have found rather than a "
              "rerun of the same mistake. Labels are timestamp spans, not chunk ids, "
              "so re-chunking the corpus does not invalidate the scoreboard.",
              "Verification: an oracle built by reading the whole archive, compared "
              "against one arm's retrieved spans by mechanical overlap for missed "
              "evidence, and by a claim judge for ungrounded assertions."),
          '<h3>Why the oracle is the expensive, correct thing to build first</h3>',
          '<ul class="prose">',
          f'<li>For each question the <b>entire archive</b> is read in four batches '
          f'and every span labelled: answers the question, merely supports an '
          f'extension, or neither. {S["oracle"]["ans"]} answer spans and '
          f'{S["oracle"]["sup"]} support spans across {S["oracle"]["n"]} '
          f'questions.</li>',
          '<li><b>It validated the probe design independently.</b> All ten '
          'disconnected questions came back with zero answer and zero support '
          'spans.</li>',
          '<li><b>It corrected the author.</b> Three questions hand-labelled '
          '"adjacent" turned out to have material, one with 31 answer spans. The '
          'ground truth needed the same scrutiny as the system.</li>',
          '<li><b>It expires.</b> 20 calls today because the archive fits in a context '
          'window in batches. At a few hundred takes this stops being computable. '
          '<b>The scoreboard is only buildable while the corpus is small</b>, which is '
          'the argument for building it before scaling rather than after.</li>',
          "</ul>",
          '<div class="note prose"><b>Abstention is reported two ways</b>, because the '
          'reading is a judgement. Strict counts only an explicit "none". Fair counts '
          'anything that did not claim <code>direct</code>, on the grounds that "I '
          'have not covered this, here is a flagged extension" is honest behaviour and '
          'scoring it as failure punishes the label rather than the conduct.</div>',
          "</section>"]

    # 8 experiments
    P += ['<section id="s8"><h2><span class="n">08</span> Experiments</h2>',
          '<h3>The probe set</h3>',
          '<p class="prose">38 questions in four bands, written by hand. The system is '
          'never told the band; it is held out and the oracle is what actually scores '
          'against it.</p>',
          '<div class="tw"><table><thead><tr><th>Band</th><th class="num">n</th>'
          '<th>What it tests</th></tr></thead><tbody>',
          '<tr><td class="lbl">connected</td><td class="num">10</td><td>a take is '
          'about exactly this. can retrieval find what plainly exists</td></tr>',
          '<tr><td class="lbl">adjacent</td><td class="num">10</td><td>his domain, not '
          'his subject. can it extend without overclaiming</td></tr>',
          '<tr><td class="lbl">identity</td><td class="num">8</td><td>about him. '
          'scattered across many takes, never the subject of one. <b>assembly, not '
          'retrieval</b></td></tr>',
          '<tr><td class="lbl">disconnected</td><td class="num">10</td><td>outside his '
          'world. can it decline</td></tr>',
          "</tbody></table></div>",
          '<h3>The crossed design</h3>',
          fig(D_2X2, "<b>Two changes were made at once in the first experiment, so "
              "the result could not be attributed.</b> Crossing retrieval against "
              "presentation separates them: retrieval is the entire effect, the prompt "
              "is worth nothing on recall, and the two do not interact.",
              "A two by two crossing single-query against model-written queries, and "
              "prompt v1 against v2, showing retrieval main effect plus 0.351 and "
              "prompt plus 0.008."),
          '<h3>All arms</h3>',
          '<div class="tw"><table><thead><tr><th>Arm</th><th>Configuration</th>'
          '<th class="num">connected</th><th class="num">adjacent</th>'
          '<th class="num">identity</th><th class="num">declined</th>'
          '<th class="num">spans</th></tr></thead><tbody>']
    rows = [("r0u0", "1 query &middot; prompt v1"), ("r0u1", "1 query &middot; prompt v2"),
            ("r1u0", "model queries &middot; v1"), ("r1u1", "model queries &middot; v2"),
            ("r1u1_gate", "model queries &middot; v2 &middot; + gate")]
    for a, desc in rows:
        x = A[a]
        cls = ' class="up"' if a == "r1u1" else ""
        P.append(f'<tr><td class="lbl">{a}</td><td>{desc}</td>'
                 f'<td class="num"{cls}>{x["connected"]:.2f}</td>'
                 f'<td class="num">{x["adjacent"]:.2f}</td>'
                 f'<td class="num">{x["identity"]:.2f}</td>'
                 f'<td class="num">{x["none"]}/10</td>'
                 f'<td class="num">{x["spans"]:.0f}</td></tr>')
    P += ["</tbody></table></div>",
          '<p class="prose"><strong>Across every arm, in every configuration, the '
          'system never once falsely claimed his position on a question he has not '
          'covered: 10/10 each time.</strong> The ungrounded failure mode was not the '
          'problem. The missed one was, and it was invisible until there was ground '
          'truth to measure it against.</p>',
          "</section>"]

    # 9 what they say
    P += ['<section id="s9"><h2><span class="n">09</span> What the experiments '
          'say</h2>',
          "<ol class='dec prose'>",
          '<li><b>Query diversification is the architecture.</b> Best single query '
          'across the connected band reaches 0.70; the union of three or four reaches '
          '0.91. It works by <b>shotgun, not insight</b> &mdash; the model does not '
          'know the archive. Queries in the asker\'s framing return zero ("wasted my '
          'twenties regret", 0/10); queries in his return most of it ("controlled '
          'motivation autonomous motivation").</li>',
          '<li><b>Prompt wording is worth +0.008 on recall.</b> The intuitive story was '
          'wrong and only the crossed design could say so. What the prompt does buy is '
          'declining: with the weak prompt and rich retrieval, the system declined '
          'correctly <b>0 times out of 10</b>. Give a weak prompt more material and it '
          'always finds something to extend from.</li>',
          '<li><b>Lexical signals cannot detect coverage.</b> Two independent attempts '
          'failed the same way. Query-term coverage scored a Postgres question at 1.00 '
          'because unseen words contributed to neither side of the ratio. A BM25 score '
          'floor overlaps across bands: nihilism scores 0.99 with 17 oracle spans, '
          'marathon scores 1.76 with zero, and <b>"Who are you?" scores 0.00 with '
          'ten</b>. Word overlap is not topical presence.</li>',
          '<li><b>The model reading the evidence beats every metric over that same '
          'evidence.</b> On the viola and electric-car questions, retrieval said '
          '<code>direct</code> and the model said <code>none</code>. The model was '
          'right both times.</li>',
          '<li><b>The gate trades recall for abstention, and currently trades too '
          'much.</b> Declining goes 2/10 to 10/10 and context drops thirteenfold, but '
          'identity recall falls 0.53 to 0.19 and it empties "Who are you?" entirely. '
          '<b>That is structural.</b> Self-facts sit in eight takes; no single span is '
          'obviously about the question; a per-span test discards all of them. A span '
          'that is one eighth of an answer is not relevant alone, but eight of them '
          'are.</li>',
          '<li><b>Identity questions are a different task.</b> They move most under '
          'query diversification (0.17 to 0.53, roughly 3x) and break worst under '
          'per-span gating. Retrieval finds a topic; identity requires assembly.</li>',
          "</ol>", "</section>"]

    # 10 related work
    P += ['<section id="s10"><h2><span class="n">10</span> Related work</h2>',
          '<p class="prose">Full survey and verdicts in '
          '<code>engines/knowledge_style/REFERENCES.md</code> and '
          '<code>blocks/knowledge/REFERENCES.md</code>. Citation counts from Semantic '
          'Scholar. What follows is only what this build actually rests on.</p>',
          '<div class="tw"><table><thead><tr><th>Work</th><th class="num">cites</th>'
          '<th>How it is used here</th></tr></thead><tbody>',
          '<tr><td class="lbl">Lost in the Middle <br><span class="cite">2307.03172</span> '
          'TACL</td><td class="num"><span class="cc">4899</span></td>'
          '<td>why whole-archive context is a measuring stick and not a baseline: '
          'accuracy is not uniform across a long window, and a bigger window enlarges '
          'the middle</td></tr>',
          '<tr><td class="lbl">Generative agents from interviews '
          '<br><span class="cite">2411.10109</span></td>'
          '<td class="num"><span class="cc">311</span></td>'
          '<td>the reason to expect this to work at all: transcripts of a person '
          'talking predict their held-out positions at 83% of their own test-retest '
          'consistency. Also fixes the right ceiling: self-consistency, not '
          'perfection</td></tr>',
          '<tr><td class="lbl">STRAP <br><span class="cite">2010.05700</span> EMNLP</td>'
          '<td class="num"><span class="cc">289</span></td>'
          '<td>the pair-construction recipe for the style layer. Not built here; this '
          'is the content half only</td></tr>',
          '<tr><td class="lbl">SeCom <br><span class="cite">2502.05589</span> ICLR</td>'
          '<td class="num"><span class="cc">101</span></td>'
          '<td>segment-level memory beats turn/session/summary on dialogue. Deferred '
          'on <b>domain</b>, not quality: its segmenter keys on multi-session dialogue '
          'and these takes are continuous monologue</td></tr>',
          '<tr><td class="lbl">Semantic chunking <br><span class="cite">2410.13070</span> '
          'NAACL Findings</td><td class="num"><span class="cc">69</span></td>'
          '<td>decides the chunker. Fixed-size consistently beat semantic; the '
          'overhead was not justified</td></tr>',
          '<tr><td class="lbl">Authorship on speech '
          '<br><span class="cite">2311.07564</span> TACL</td>'
          '<td class="num"><span class="cc">10</span></td>'
          '<td>do not prettify ASR; character n-grams are competitive with neural '
          'models on speech; topic control is mandatory or a discriminator wins by '
          'reading the topic</td></tr>',
          '<tr><td class="lbl">Stochastic Evidence Graphs <br><span class="cite">'
          'Dixon, AIFI 2026</span></td><td class="num flat">preprint</td>'
          '<td><b>caught the confound in this project.</b> A terminal score detects '
          'that something changed; typed per-stage measurement identifies which edge. '
          'Its crossed retrieval-by-presentation design is exactly the 2&times;2 run '
          'here</td></tr>',
          '<tr><td class="lbl">pwc grounding architecture <br><span class="cite">'
          'internal</span></td><td class="num flat">&mdash;</td>'
          '<td>the claim trichotomy (internal evidence / external evidence / '
          'hypothesis) and its own verdict: <b>unify the mental model in prompts, not '
          'the code</b>. Its Protocol layer is deliberately not copied &mdash; one '
          'retrieval backend does not justify the indirection</td></tr>',
          "</tbody></table></div>",
          '<div class="note prose"><b>Deliberately not adopted:</b> LightRAG and '
          'RAG-Anything index by LLM entity extraction, which reverses the '
          'verbatim-storage decision and adds an LLM pass over the whole corpus to the '
          'build. kotaemon is a product shell solving a problem this does not have. '
          'Every low-resource style-transfer method (TinyStyler, SETTP, AuthorMix) '
          'exists to work around not having data; at &gt;100k words this corpus is in '
          'the opposite regime.</div>',
          "</section>"]

    # 11 worth exploring
    P += ['<section id="s11"><h2><span class="n">11</span> Modelling choices worth '
          'exploring</h2>',
          '<p class="prose">Ordered by expected value per unit of effort. Each is one '
          'variable, one arm, scored against the same frozen oracle.</p>',
          '<div class="tw"><table><thead><tr><th>Change</th><th>Rationale</th>'
          '<th>Cost</th></tr></thead><tbody>',
          '<tr><td class="lbl">Gate the decision, not the context</td>'
          '<td>run the gate to decide empty-or-not, then hand generation the '
          '<em>ungated</em> spans. Keeps 10/10 declining and 0.84 recall instead of '
          'trading one for the other</td><td>1 arm</td></tr>',
          '<tr><td class="lbl">Curate the exemplars</td>'
          '<td>the identity axis is 8 chunks chosen by nothing, and it is the measured '
          'cause of the template effect. Hand-picking 15-20 is an afternoon</td>'
          '<td>1 arm</td></tr>',
          '<tr><td class="lbl">Feed the map to the searcher</td>'
          '<td>rephrasing is currently a blind guess that works by volume. With the '
          'map it becomes a lookup. <b>This is the one that decides scaling</b>: '
          'guessing does not improve as the archive grows, while the amount it can '
          'miss does</td><td>1 arm</td></tr>',
          '<tr><td class="lbl">Pseudo-relevance feedback</td>'
          '<td>search once, harvest distinctive terms from the hits, search again. '
          'Learns his vocabulary from the corpus instead of guessing it. No LLM call, '
          'no new dependency</td><td>1 arm</td></tr>',
          '<tr><td class="lbl">Embeddings alongside BM25</td>'
          '<td>lexical retrieval is why "Who are you?" scores 0.00. Semantic '
          'similarity measures topical proximity rather than word overlap</td>'
          '<td>1 arm + a model</td></tr>',
          '<tr><td class="lbl">Claim-level grounding markers</td>'
          '<td>from pwc: mark each claim as his-position / extension / outside, at '
          'generation time rather than one label for the whole answer. The verifier '
          'already computes this after the fact</td><td>prompt + parser</td></tr>',
          '<tr><td class="lbl">An independent judge</td>'
          '<td>the judge currently shares a family with the generator. This is the '
          'largest methodological weakness in the whole scoreboard</td>'
          '<td>a second provider</td></tr>',
          "</tbody></table></div>", "</section>"]

    # 12 open questions
    P += ['<section id="s12"><h2><span class="n">12</span> Open questions</h2>',
          "<ol class='dec prose'>",
          '<li><b>Does it sound like him?</b> Every number here measures grounding and '
          'retrieval. None of them says the voice is right. This is blocking, it needs '
          'someone who knows the creator, and no metric substitutes for it.</li>',
          '<li><b>Monologue or conversational?</b> The corpus is 17 broadcast '
          'monologues, and deployment is a person asking a question. The situation '
          'axis has one observed value, so the person cannot be separated from the '
          'format. This is a data decision, not a modelling one, and everything '
          'downstream is conditional on it.</li>',
          '<li><b>What is the model\'s prior worth?</b> The bare-persona control '
          'produced fluent, plausible, correctly-registered answers with none of his '
          'specific content. If a frontier model already carries the voice, the '
          'archive\'s job is content and the style layer has far less to add than '
          'assumed.</li>',
          '<li><b>Does any of this hold at scale?</b> Six hours. The reasoning about '
          'ten or a hundred is mechanism, not measurement, and the oracle that would '
          'settle it stops being computable at roughly the same point.</li>',
          '<li><b>How much variance is in these numbers?</b> One rollout per question '
          'throughout. With model-written queries the retrieval is stochastic, so a '
          'repeat could move every figure on this page. Nothing here reports a '
          'spread.</li>',
          '<li><b>Is the claim judge measuring anything?</b> Every number reported is '
          '<code>--no-judge</code>: mechanical span overlap only. The ungrounded '
          'failure mode has never actually been measured, only observed not to '
          'happen.</li>',
          "</ol>",
          '<div class="rule prose"><b>The honest summary.</b> The retrieval half is '
          'measured, attributed and reproducible. The grounding half is observed but '
          'not yet measured. The voice half is not measured at all. Two of those three '
          'have a cheap path forward; the third needs a person.</div>',
          "</section>"]

    P += ['<p class="foot">Generated by <code>pipeline_doc.py</code> from '
          '<code>doc_stats.json</code>, which comes from the runs. Prompts in '
          '<code>PROMPTS.md</code> (regenerated from source). Pipeline in '
          '<code>Makefile</code>. Argument behind the work in '
          '<code>engines/knowledge_style/FRAMING.md</code>.</p>', "</div>"]

    out = HERE / "PIPELINE.html"
    out.write_text("\n".join(P))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
