// One list of pages, in reading order. Each page has <nav class="rail" data-on="<id>">
// and an empty <div class="pager">; this fills both.
(function () {
  const PAGES = [
    ["Start", "index", "Start here", ""],
    ["Start", "setup", "Set up a machine", ""],
    ["Start", "run", "Run the app", "s-live"],
    ["Build a creator", "build", "How building works", ""],
    ["Build a creator", "answers", "1 · Answers", "s-answers"],
    ["Build a creator", "voice", "2 · Voice", "s-voice"],
    ["Build a creator", "clips", "3 · Face clips", "s-clips"],
    ["Build a creator", "motion", "4 · Voice to face motion", "s-motion"],
    ["Build a creator", "render", "5 · Face motion to picture", "s-render"],
    ["Build a creator", "live", "6 · The live app", "s-live"],
    ["Reference", "new-creator", "A new creator, checklist", ""],
    ["Reference", "credits", "Credits", ""],
  ];
  const nav = document.querySelector("nav.rail");
  const on = nav ? nav.dataset.on : "";
  if (nav) {
    let html = '<a class="brand" href="index.html">alive</a>', group = "";
    for (const [g, id, title, cls] of PAGES) {
      if (g !== group) { html += `<div class="group">${g}</div>`; group = g; }
      html += `<a class="p ${cls} ${id === on ? "on" : ""}" href="${id}.html">${title}</a>`;
    }
    html += '<div class="group">Repository</div><a class="p" href="https://github.com/pavanteja295/alive">github.com/pavanteja295/alive</a>';
    nav.innerHTML = html;
  }
  const pager = document.querySelector(".pager");
  const i = PAGES.findIndex(p => p[1] === on);
  if (pager && i >= 0) {
    const prev = PAGES[i - 1], next = PAGES[i + 1];
    pager.innerHTML = (prev ? `<a href="${prev[1]}.html">&larr; ${prev[2]}</a>` : "<span></span>")
                    + (next ? `<a href="${next[1]}.html">${next[2]} &rarr;</a>` : "<span></span>");
  }
})();
