#!/usr/bin/env python3
"""Render a question corpus for a person to read. Nothing here judges anything.

    python3 qreview.py --in work/<subject>/qcorpus.jsonl --out qcorpus.html

This is the step nothing replaces. At 3,000 items nobody reads them all, so the
judge's contract has to be right BEFORE the full run -- read the calibration
batch, then launch. `RECIPE.md`, *Calibrate, and separately check at scale*.

Bucket A/B is shown per item because it is the one thing a reader would
otherwise assume: a question is NOT rejected when retrieval misses it. It is
kept, and the miss is the point.
"""
import argparse
import collections
import html
import json
import pathlib
import re
import statistics as st

CSS = """
:root{--bg:#faf8f5;--ink:#1a1714;--mut:#6b625a;--line:#e0d9d0;--ochre:#a8622a;
      --ok:#3d7a4e;--miss:#a8622a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:15px/1.65 Archivo,-apple-system,Segoe UI,sans-serif}
.wrap{max-width:880px;margin:0 auto;padding:48px 24px 80px}
h1{font:600 30px/1.2 Newsreader,Georgia,serif;margin:0 0 6px}
.sub{color:var(--mut);margin:0 0 28px;font-size:14px;max-width:62ch}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(118px,1fr));
       gap:1px;background:var(--line);border:1px solid var(--line);margin:0 0 12px}
.stat{background:var(--bg);padding:12px 14px}
.stat b{display:block;font:600 22px/1.2 Newsreader,serif}
.stat span{font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--mut)}
.q{border-top:1px solid var(--line);padding:22px 0}
.q h2{font:500 17px/1.45 Newsreader,serif;margin:0 0 10px}
.n{color:var(--mut);font:400 12px JetBrains Mono,monospace;margin-right:8px}
.a{background:#fff;border-left:2px solid var(--ochre);padding:12px 15px;
   font:400 14px/1.75 Newsreader,serif;margin:0 0 10px}
.f{background:#f6edd2;color:#7a6216;border-radius:2px;padding:0 3px;
   font:400 13px JetBrains Mono,monospace}
.meta{font:400 11.5px JetBrains Mono,monospace;color:var(--mut);
      display:flex;flex-wrap:wrap;gap:14px;align-items:center}
.bar{position:sticky;top:0;background:var(--bg);padding:12px 0;margin:0 0 4px;
     border-bottom:1px solid var(--line);z-index:5;display:flex;gap:8px;
     flex-wrap:wrap;align-items:center}
.bar input{flex:1;min-width:200px;padding:8px 10px;border:1px solid var(--line);
     background:#fff;font:400 14px inherit;border-radius:2px}
.bar button{padding:7px 12px;border:1px solid var(--line);background:#fff;
     cursor:pointer;font:500 12px JetBrains Mono,monospace;border-radius:2px;
     color:var(--mut)}
.bar button.on{background:var(--ink);color:var(--bg);border-color:var(--ink)}
.count{font:400 12px JetBrains Mono,monospace;color:var(--mut)}
.q h2{cursor:pointer}
.q h2:hover{color:var(--ochre)}
.q.shut .a{display:none}
.hide{display:none!important}
.pill{border-radius:2px;padding:1px 6px;font-weight:600}
.A{background:#e4f0e6;color:var(--ok)} .B{background:#f6e6da;color:var(--miss)}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default="qcorpus.html")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    rows = [json.loads(l) for l in pathlib.Path(a.inp).read_text().splitlines()
            if l.strip()]
    if not rows:
        raise SystemExit(f"{a.inp} is empty")
    show = rows[:a.limit] if a.limit else rows
    med = lambda k: st.median([r[k] for r in rows if k in r]) if rows else 0
    nA = sum(1 for r in rows if r.get("retrieval_found_source"))
    takes = len({r.get("source_take") for r in rows})
    # fact_bearing is written by qtag.py with a model call, not guessed here.

    out = [f"<title>Question corpus</title><style>{CSS}</style>",
           "<div class=wrap><h1>Question corpus</h1>",
           "<p class=sub>Each answer was assembled by an independent judge out of "
           "the creator's own words. Bracketed spans are the judge's connective "
           "fillers; everything else is verbatim. "
           "<b>A</b> means our retriever finds the answering passage — ready to "
           "deploy. <b>B</b> means it does not: the question is still good and is "
           "kept, and the miss is a defect in the retriever, not the question.</p>",
           "<div class=stats>",
           f"<div class=stat><b>{len(rows)}</b><span>questions</span></div>",
           f"<div class=stat><b>{takes}</b><span>videos</span></div>",
           f"<div class=stat><b>{100*nA//len(rows)}%</b><span>bucket A</span></div>",
           f"<div class=stat><b>{len(rows)-nA}</b><span>bucket B</span></div>",
           f"<div class=stat><b>{med('filler_ratio'):.3f}</b><span>median filler</span></div>",
           f"<div class=stat><b>{med('grounded_trigram'):.2f}</b><span>median verbatim</span></div>",
           f"<div class=stat><b>{med('answer_words'):.0f}</b><span>median words</span></div>",
           "</div>",
           "<div class=bar>"
           "<input id=s placeholder='search questions and answers...'>"
           "<button data-f=all class=on>all</button>"
           "<button data-f=A>bucket A</button>"
           "<button data-f=B>bucket B</button>"
           "<button data-f=t2>tier 2+</button>"
           "<button data-f=fact>fact-bearing</button>"
           "<button id=exp>expand all</button>"
           "<span class=count id=c></span></div>"]

    for i, r in enumerate(show, 1):
        ans = re.sub(r"\[([^\]]*)\]", r"<span class=f>\1</span>",
                     html.escape(r["answer"]))
        b = "A" if r.get("retrieval_found_source") else "B"
        rank = r.get("retrieval_rank_of_source")
        bits = [f"<span class='pill {b}'>{b}</span>"]
        if b == "A" and rank:
            bits.append(f"<span>found at rank {rank}</span>")
        if r.get("answers_on_this_passage"):
            bits.append(f"<span>{r['answers_on_this_passage']} other questions "
                        f"on this passage, max overlap "
                        f"{r.get('max_answer_overlap_here',0):.2f}</span>")
        if r.get("fact_bearing"):
            bits.append("<span>fact-bearing</span>")
        bits += [f"<span>filler {r.get('filler_ratio',0):.3f}</span>",
                 f"<span>{html.escape(str(r.get('source_ts','')))}</span>",
                 f"<span>{html.escape(str(r.get('source_take',''))[:34])}</span>"]
        tags = " ".join(filter(None, [
            b, f"t{r.get('tier',1)}",
            "t2" if r.get("tier", 1) >= 2 else "",
            "fact" if r.get("fact_bearing") else ""]))
        blob = html.escape((r["q"] + " " + r["answer"]).lower())
        out.append(f"<div class='q shut' data-tags='{tags}' data-blob=\"{blob}\">"
                   f"<h2><span class=n>{i:04d}</span>"
                   f"{html.escape(r['q'])}</h2><div class=a>{ans}</div>"
                   f"<div class=meta>{''.join(bits)}</div></div>")

    out.append("""</div><script>
const qs=[...document.querySelectorAll('.q')],
      box=document.getElementById('s'),cnt=document.getElementById('c');
let filt='all';
function apply(){
  const t=box.value.trim().toLowerCase();
  let n=0;
  for(const q of qs){
    const okF = filt==='all' || q.dataset.tags.split(' ').includes(filt);
    const okT = !t || q.dataset.blob.includes(t);
    const show = okF && okT;
    q.classList.toggle('hide', !show);
    if(show) n++;
  }
  cnt.textContent = n + ' shown';
}
box.addEventListener('input', apply);
for(const b of document.querySelectorAll('.bar button[data-f]')){
  b.addEventListener('click', ()=>{
    document.querySelectorAll('.bar button[data-f]').forEach(x=>x.classList.remove('on'));
    b.classList.add('on'); filt=b.dataset.f; apply();
  });
}
// answers start collapsed: at thousands of items you scan questions, not prose
for(const q of qs) q.querySelector('h2').addEventListener('click',
  ()=>q.classList.toggle('shut'));
let open=false;
document.getElementById('exp').addEventListener('click', e=>{
  open=!open; qs.forEach(q=>q.classList.toggle('shut', !open));
  e.target.textContent = open ? 'collapse all' : 'expand all';
});
apply();
</script>""")
    pathlib.Path(a.out).write_text("\n".join(out))
    print(f"{len(show)} of {len(rows)} -> {a.out}")


if __name__ == "__main__":
    main()
