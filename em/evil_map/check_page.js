// Persona map (em/evil_map/PREREG.md): run the built page's script against a minimal DOM stub, so a page that throws
// on load is caught before it is published. It found a real one: json.dump writes bare NaN, which Python's own parser
// accepts but JSON.parse rejects, so the page went blank while every local check passed.
//   node em/evil_map/check_page.js results/evil_map/em_trajectories_full.html
const fs = require('fs');
const html = fs.readFileSync(process.argv[2] || 'results/evil_map/em_trajectories_full.html', 'utf8');
const data = html.match(/<script id="data" type="application\/json">([\s\S]*?)<\/script>/)[1].replace(/<\\\//g, '</');
const code = html.match(/<script>\n(const D = JSON\.parse[\s\S]*?)<\/script>/)[1];
const ids = new Set([...html.matchAll(/id="([^"]+)"/g)].map(m => m[1]));
const mk = (id) => ({ id, innerHTML: '', textContent: '', value: '', hidden: false, disabled: false,
  style: {}, dataset: {}, addEventListener(){}, setAttribute(){}, getAttribute(){return null;},
  querySelectorAll(){return nodes();}, closest(){return null;}, appendChild(){}, });
const nodes = () => [0, 1, 2].map((i) => ({ dataset: { v: String(i), d: String(i === 0 ? -1 : 1) }, disabled: false,
  setAttribute(){}, getAttribute(){return null;}, textContent: '', value: '' }));
const store = {};
global.document = {
  getElementById: (id) => { if (id === 'data') return { textContent: data };
    if (!ids.has(id)) { console.log('MISSING ELEMENT:', id); return null; } return store[id] ||= mk(id); },
  querySelectorAll: () => nodes(), querySelector: () => null,
  documentElement: {}, head: { appendChild(){} },
};
global.getComputedStyle = () => ({ getPropertyValue: () => '#808080' });
global.matchMedia = () => ({ addEventListener(){} });
global.MutationObserver = class { observe(){} };
global.Plotly = { react(){}, newPlot(){} };
global.fetch = () => Promise.resolve({ ok: true, json: () => ({}) });
try { eval(code); console.log('script ran with no throw'); }
catch (e) { console.log('THROWS:', e.constructor.name + ':', e.message); console.log(String(e.stack).split('\n').slice(0, 4).join('\n')); process.exit(1); }
