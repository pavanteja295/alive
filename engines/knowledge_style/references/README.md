# references/

The papers for this system, inside the system. `problems.md` N20.

```
REFERENCES.md     the owner. Every entry carries a verdict against one of our
                  decisions and names the seam or issue it would change
references/
  SURVEY.md       the waiting room. A collected batch, read and rated, before
                  any of it has earned a verdict. Entries graduate out of here
  BASELINE.md     what the system should be, and how the field measures it
  BELLS.md        improvements that assume a baseline already exists
  pdf/            the papers themselves
```

**Reading order is enforced by the split.** Nothing in `BELLS.md` is read until
`BASELINE.md` is complete — complete meaning each of the four components has its source
named: input/output processing, the pipeline, the metrics, the reference baseline. Until
then a bells paper cannot be evaluated, because there is nothing for it to improve on.

**A paper moves BELLS → BASELINE when we adopt it**, and that move is a commit. So
`BASELINE.md` becomes the description of what the system is made of, assembled one dated
adoption at a time.

**What bounds the set** changes with the state (`improve-system` skill):

| state | filed only if it touches | looking for |
|---|---|---|
| MAPPING | a seam in `SYSTEM.md` or an axis in `METRICS.md` | how the field decomposes this, and what it measures |
| EXHAUSTING / EXPLORING | a `problems.md` entry | mechanisms that might move a measured thing |

An entry with no `touches:` is not filed. If the verdict can only be written vaguely, it
is background reading.

**First batch read 2026-09-15**, in `SURVEY.md`: Delphi (the company shipping this
product), `2411.10109` with its Stanford HAI policy brief, `2603.29890`, `2601.06490`, and
the open-source landscape. None has graduated to `BASELINE.md` yet.

`Mycode/persona_style_papers` holds 54 PDFs downloaded for an
earlier question, with an index and Semantic Scholar citation counts. Those are candidates
to draw from, not the collection — a paper joins this system when it has been read, with
its related work and its forward citations, and filed with a verdict.
