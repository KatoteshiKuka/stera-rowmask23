// Graduation talk: STERA. Builds presentation/STERA_graduation_talk.pptx (slide text in Italian, the language of the defense).
// Font: Calibri. Three text sizes: 36 (titles and numbers), 20 (body), 14 (captions and charts).
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.APPLY_THEME);

const T = 36, B = 20, S = 14;
const HEX = { navy: "002A61", stera: "1F6F8B", ts33: "C2672B", ts8: "7A8B3A", grigio: "6B7280", chiaro: "B9C7D2",
              testo: "1F2937", tenue: "EEF3F7", bordo: "D1D5DB", bianco: "FFFFFF" };
const THEME = { name: "STERA Roma Tre", headFontFace: "Calibri", bodyFontFace: "Calibri",
  colors: { dk1: HEX.testo, lt1: HEX.bianco, dk2: HEX.navy, lt2: HEX.tenue, accent1: HEX.stera, accent2: HEX.ts33,
            accent3: HEX.ts8, accent4: HEX.grigio, accent5: HEX.chiaro, accent6: HEX.navy, hlink: HEX.stera, folHlink: HEX.grigio } };
const IMG = (f) => path.join(__dirname, "images", f);

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";            // 13,333 x 7,5 pollici
pres.author = "Andrea Nardoni";
pres.title = "STERA: parametrizzazione strutturata dei pesi per modelli linguistici Transformer";
pres.theme = { headFontFace: "Calibri", bodyFontFace: "Calibri" };
const C = pres.SchemeColor;
const W = 13.333, M = 0.6;

// ---------- layout ----------
pres.defineSlideMaster({ title: "COPERTINA", background: { color: HEX.navy },
  objects: [
    { image: { path: IMG("logo_white.png"), x: M + 0.2, y: 0.6, w: 3.4, h: 3.4 * 1004 / 3508 } },
    { placeholder: { options: { name: "title", type: "title", x: M + 0.2, y: 2.1, w: W - 2 * M - 0.4, h: 2.1, fontSize: T, bold: true,
                                color: C.background1, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M + 0.2, y: 4.45, w: W - 2 * M - 0.4, h: 2.4, fontSize: B,
                                color: C.background1, valign: "top", align: "left", margin: 0 }, text: "" } },
  ] });
pres.defineSlideMaster({ title: "CONTENUTO", background: { color: HEX.bianco },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.35, w: W - 2 * M, h: 0.9, fontSize: T, bold: true,
                                color: C.text2, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { image: { path: IMG("logo_blue.png"), x: M, y: 6.95, w: 1.4, h: 1.4 * 1004 / 3508 } },
  ],
  slideNumber: { x: W - M - 0.6, y: 6.9, w: 0.6, h: 0.4, fontFace: "Calibri", fontSize: S, color: HEX.grigio, align: "right" } });
pres.defineSlideMaster({ title: "CHIUSURA", background: { color: HEX.navy },
  objects: [
    { image: { path: IMG("logo_white.png"), x: (W - 3.4) / 2, y: 0.9, w: 3.4, h: 3.4 * 1004 / 3508 } },
    { placeholder: { options: { name: "title", type: "title", x: M, y: 2.7, w: W - 2 * M, h: 1.2, fontSize: T, bold: true,
                                color: C.background1, align: "center", valign: "middle", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 4.1, w: W - 2 * M, h: 1.6, fontSize: B,
                                color: C.background1, align: "center", valign: "top", margin: 0 }, text: "" } },
  ] });

// ---------- helpers ----------
let nOgg = 0;
const nome = (p) => `${p}_${++nOgg}`;
function testo(s, t, o) { s.addText(t, { isTextBox: true, fontSize: B, color: C.text1, margin: 0, valign: "top", objectName: nome("testo"), ...o }); }
function riquadro(s, x, y, w, h, o = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: o.fill || C.background2 },
    line: o.line ? { color: o.line, width: o.lw || 1.5 } : { type: "none" }, objectName: nome("riquadro") });
}
function freccia(s, x1, y1, x2, y2, col = C.accent4) {
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001, h: Math.abs(y2 - y1) || 0.001,
    flipH: x2 < x1, flipV: y2 < y1, line: { color: col, width: 2, endArrowType: "triangle" }, objectName: nome("freccia") });
}
function cella(s, x, y, d, fill, transp = 0) {
  s.addShape(pres.shapes.RECTANGLE, { x, y, w: d, h: d, fill: { color: fill, transparency: transp }, line: { color: HEX.bordo, width: 0.75 }, objectName: nome("cella") });
}
function nuova(sezione, titolo, note) {
  const s = pres.addSlide({ masterName: "CONTENUTO", sectionTitle: sezione });
  s.addText(titolo, { placeholder: "title" });
  if (note) s.addNotes(note);
  return s;
}
const assi = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt",
  catAxisLabelFontSize: S, valAxisLabelFontSize: S, dataLabelFontSize: S, legendFontSize: S,
  catAxisLabelColor: HEX.testo, valAxisLabelColor: HEX.grigio, valGridLine: { color: "E5E7EB", size: 0.75 }, catGridLine: { style: "none" },
  catAxisLineShow: false, valAxisLineShow: false };
const MODELLI = ["TinyStories-8M", "STERA 30M", "TinyStories-33M"];
const COL_MOD = [HEX.ts8, HEX.stera, HEX.ts33];
function barreModelli(s, valori, o) {
  s.addChart(pres.charts.BAR, MODELLI.map((m, i) => ({ name: m, labels: [""], values: [valori[i]] })), {
    barDir: "col", barGrouping: "clustered", barGapWidthPct: 40, chartColors: COL_MOD, showValue: true, dataLabelPosition: "outEnd",
    dataLabelColor: HEX.testo, showLegend: true, legendPos: "b", legendColor: HEX.testo, valAxisHidden: true, valGridLine: { style: "none" },
    ...assi, valGridLine: { style: "none" }, catAxisHidden: true, ...o });
}

// ---------- 1. title slide ----------
pres.addSection({ title: "Apertura" });
let s = pres.addSlide({ masterName: "COPERTINA", sectionTitle: "Apertura" });
s.addText("STERA: progettazione e valutazione di una parametrizzazione strutturata dei pesi per modelli linguistici Transformer", { placeholder: "title" });
s.addText([
  { text: "Laureando: Andrea Nardoni", options: { breakLine: true } },
  { text: "Relatore: Prof. Fabio Gasparetti", options: { breakLine: true } },
  { text: " ", options: { breakLine: true, fontSize: S } },
  { text: "Corso di Laurea Magistrale in Ingegneria Informatica · Dipartimento di Ingegneria Civile, Informatica e delle Tecnologie Aeronautiche · A.A. 2025/2026", options: { fontSize: S } },
], { placeholder: "body" });
s.addNotes("Buongiorno. Presento STERA, una rappresentazione compatta dei pesi dei modelli linguistici, addestrata da zero e valutata sia sulla qualità del testo sia sui costi di esecuzione.");

// ---------- 2. introduction ----------
pres.addSection({ title: "Contesto" });
s = nuova("Contesto", "I modelli crescono, la memoria dei dispositivi no",
  "Un modello da 7 miliardi di parametri in mezza precisione pesa circa 14 GB, quasi tutta la memoria di un portatile da 16 GB. In generazione ogni token rilegge tutti i pesi: con 68 GB/s di banda il limite teorico è circa 5 token al secondo, indipendentemente dalla potenza di calcolo.");
[["14 GB", "pesi di un modello da 7 miliardi di parametri in FP16"], ["16 GB", "memoria del Mac M1, condivisa fra CPU e GPU"],
 ["≈ 5 token/s", "limite teorico se ogni token rilegge 14 GB a 68 GB/s"]].forEach(([n, l], i) => {
  const x = M + i * 4.15; riquadro(s, x, 1.75, 3.75, 3.1);
  testo(s, n, { x: x + 0.35, y: 2.05, w: 3.05, h: 0.8, fontSize: T, bold: true, color: C.accent1 });
  testo(s, l, { x: x + 0.35, y: 3.0, w: 3.05, h: 1.6 });
});
testo(s, "Ridurre i byte dei pesi è la leva più diretta per eseguire modelli più grandi sullo stesso dispositivo",
  { x: M, y: 5.35, w: W - 2 * M, h: 0.9, bold: true, color: C.text2 });

// ---------- 3. problem ----------
s = nuova("Contesto", "Il collo di bottiglia è la memoria",
  "Nella generazione token per token ogni matrice viene moltiplicata per un solo vettore: per ogni peso letto si fanno due operazioni. Conta quindi la banda di memoria, non il calcolo. Se il modello non entra nella memoria principale, i pesi vanno letti dal disco, circa trenta volte più lento.");
const liv = [["Unità di calcolo", 2.6, C.accent1, C.background1], ["Memoria principale · 16 GB", 4.4, C.background2, C.text1], ["Disco SSD", 6.2, C.background1, C.text1]];
liv.forEach(([t, w, f, tc], i) => {
  const y = 1.75 + i * 1.75, x = M + (6.2 - w) / 2;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.85, rectRadius: 0.08, fill: { color: f }, line: { color: HEX.chiaro, width: 1 }, objectName: nome("livello") });
  testo(s, t, { x, y, w, h: 0.85, align: "center", valign: "middle", bold: true, color: tc });
});
freccia(s, M + 3.1, 3.5, M + 3.1, 2.65, C.accent1); testo(s, "68 GB/s", { x: M + 3.3, y: 2.85, w: 1.6, h: 0.45, fontSize: S, color: C.accent4 });
freccia(s, M + 3.1, 5.25, M + 3.1, 4.4, C.accent2); testo(s, "2,4 GB/s", { x: M + 3.3, y: 4.6, w: 1.6, h: 0.45, fontSize: S, color: C.accent2 });
testo(s, [
  { text: "Ogni token generato rilegge tutti i pesi: conta la banda, non il calcolo", options: { bullet: true, breakLine: true, paraSpaceAfter: 18 } },
  { text: "Se il modello non entra in memoria, i pesi vanno letti dal disco, circa 30 volte più lento", options: { bullet: true } },
], { x: 7.6, y: 2.0, w: 5.1, h: 3.4 });

// ---------- 4. existing solutions ----------
pres.addSection({ title: "Stato dell'arte" });
s = nuova("Stato dell'arte", "Le strade già percorse",
  "Quattro famiglie principali. La potatura elimina i pesi meno importanti, la quantizzazione usa meno bit per peso, le fattorizzazioni low-rank scrivono la matrice come prodotto di due matrici sottili, le matrici strutturate impongono una forma con un prodotto veloce, come circolanti, Butterfly e Monarch.");
[["Potatura", "elimina i pesi meno importanti"], ["Quantizzazione", "usa meno bit per ogni peso"],
 ["Low-rank", "scrive la matrice come prodotto U · Vᵀ"], ["Matrici strutturate", "circolanti, Butterfly, Monarch"]].forEach(([h, d], i) => {
  const x = M + (i % 2) * 6.25, y = 1.65 + Math.floor(i / 2) * 2.45;
  riquadro(s, x, y, 5.85, 2.1);
  s.addShape(pres.shapes.OVAL, { x: x + 0.35, y: y + 0.35, w: 0.65, h: 0.65, fill: { color: C.accent1 }, line: { type: "none" }, objectName: nome("numero") });
  testo(s, String(i + 1), { x: x + 0.35, y: y + 0.35, w: 0.65, h: 0.65, align: "center", valign: "middle", bold: true, color: C.background1 });
  testo(s, h, { x: x + 1.25, y: y + 0.4, w: 4.3, h: 0.55, bold: true, color: C.text2 });
  testo(s, d, { x: x + 1.25, y: y + 1.05, w: 4.3, h: 0.8 });
});

// ---------- 5. limitations ----------
s = nuova("Stato dell'arte", "Ognuna ha un prezzo",
  "La potatura lascia una sparsità irregolare, difficile da trasformare in velocità. La quantizzazione arrotonda i valori. Il low-rank limita il rango della matrice. Le matrici strutturate classiche sono rigide: non hanno gradi di libertà locali. Cerchiamo una struttura fissa, a rango pieno, con prodotto veloce e addestrata da zero.");
const intest = (t) => ({ text: t, options: { bold: true, color: HEX.bianco, fill: { color: HEX.navy }, fontSize: B } });
const riga = (a, b, c, i) => [a, b, c].map((t, j) => ({ text: t, options: { fontSize: B, color: HEX.testo, bold: j === 0, fill: { color: i % 2 ? HEX.bianco : HEX.tenue } } }));
s.addTable([
  [intest("Famiglia"), intest("Quando interviene"), intest("Limite principale")],
  riga("Potatura", "dopo il training", "sparsità irregolare, difficile da accelerare", 0),
  riga("Quantizzazione", "dopo o durante", "riduce la precisione di ogni valore", 1),
  riga("Low-rank", "durante", "limita il rango della matrice", 2),
  riga("Matrici strutturate", "durante", "struttura rigida, senza libertà locale", 3),
], { x: M, y: 1.6, w: W - 2 * M, colW: [3.0, 3.1, 6.033], rowH: 0.72, fontFace: "Calibri", valign: "middle", margin: [0, 0.15, 0, 0.15],
     border: { type: "solid", color: HEX.bordo, pt: 0.75 }, objectName: nome("tabella") });
testo(s, "Serve una struttura fissa, a rango pieno, con un prodotto veloce, da usare già in training",
  { x: M, y: 5.6, w: W - 2 * M, h: 0.8, bold: true, color: C.accent2 });

// ---------- 6. proposed solution ----------
pres.addSection({ title: "La proposta" });
s = nuova("La proposta", "La proposta: STERA",
  "STERA usa una famiglia di matrici strutturate, RowMask23, per tutte le proiezioni del Transformer. Ogni blocco 128 per 128 è descritto da 3.176 coordinate invece di 16.384 valori. Le coordinate sono memorizzate, addestrate e usate direttamente in inferenza.");
riquadro(s, M, 1.6, 4.4, 3.6, { fill: C.background2 });
testo(s, "3.176", { x: M + 0.4, y: 1.95, w: 3.6, h: 0.8, fontSize: T, bold: true, color: C.accent1 });
testo(s, "coordinate invece di 16.384 valori per blocco 128 × 128", { x: M + 0.4, y: 2.85, w: 3.6, h: 1.0 });
testo(s, "19,4% dei valori di un blocco denso", { x: M + 0.4, y: 4.3, w: 3.6, h: 0.6, fontSize: S, color: C.accent4 });
const passi = ["Memorizzare le coordinate", "Addestrare le coordinate", "Calcolare senza ricostruire la matrice"];
passi.forEach((p, i) => {
  const y = 1.6 + i * 1.3;
  riquadro(s, 5.65, y, 7.1, 0.95, { fill: C.background1, line: HEX.chiaro });
  s.addShape(pres.shapes.OVAL, { x: 5.85, y: y + 0.16, w: 0.63, h: 0.63, fill: { color: C.accent1 }, line: { type: "none" }, objectName: nome("numero") });
  testo(s, String(i + 1), { x: 5.85, y: y + 0.16, w: 0.63, h: 0.63, align: "center", valign: "middle", bold: true, color: C.background1 });
  testo(s, p, { x: 6.75, y, w: 5.8, h: 0.95, valign: "middle", bold: true, color: C.text2 });
});
testo(s, "Una sola rappresentazione dal training all'inferenza", { x: 5.65, y: 5.6, w: 7.1, h: 0.6, color: C.accent4 });

// ---------- 7. RowMask23: components ----------
s = nuova("La proposta", "RowMask23: tre componenti per blocco",
  "Ogni blocco è la somma di tre parti. Una diagonale libera, 128 valori. Una base condivisa lungo le diagonali generalizzate: tutte le posizioni con lo stesso i XOR j hanno lo stesso valore, 127 valori. E 23 correzioni libere per riga in posizioni fisse, 2.921 valori. In tutto 3.176 coordinate. La mappa da coordinate a matrice è lineare, iniettiva ed esatta.");
const d = 0.31, n8 = 8, xs = [0.95, 5.05, 9.15], y0 = 1.65;
const tinte = ["", "1F6F8B", "3B8AA3", "5A9FB4", "7A8B3A", "98A45C", "B9C7D2", "C2672B"];
for (let i = 0; i < n8; i++) for (let j = 0; j < n8; j++) {
  cella(s, xs[0] + j * d, y0 + i * d, d, i === j ? HEX.stera : HEX.bianco, i === j ? 25 : 0);
  cella(s, xs[1] + j * d, y0 + i * d, d, i === j ? HEX.bianco : tinte[i ^ j], i === j ? 0 : 35);
  const corr = (i * 3 + 2) % n8 === i ? (i * 3 + 3) % n8 : (i * 3 + 2) % n8;
  cella(s, xs[2] + j * d, y0 + i * d, d, j === corr ? HEX.ts33 : HEX.bianco, j === corr ? 25 : 0);
}
testo(s, "+", { x: xs[0] + 8 * d + 0.25, y: y0 + 0.9, w: 0.8, h: 0.8, fontSize: T, bold: true, color: C.accent4, align: "center" });
testo(s, "+", { x: xs[1] + 8 * d + 0.25, y: y0 + 0.9, w: 0.8, h: 0.8, fontSize: T, bold: true, color: C.accent4, align: "center" });
[["Diagonale", "128 valori"], ["Base XOR", "127 valori, uno per ogni i ⊕ j"], ["Correzioni", "23 per riga in posizioni fisse: 2.921"]].forEach(([h, l], k) => {
  testo(s, h, { x: xs[k], y: 4.3, w: 3.4, h: 0.5, bold: true, color: C.text2 });
  testo(s, l, { x: xs[k], y: 4.8, w: 3.4, h: 0.8 });
});
testo(s, "128 + 127 + 2.921 = 3.176 coordinate", { x: M, y: 5.75, w: 8, h: 0.6, bold: true, color: C.accent1 });
testo(s, "Schema ridotto 8 × 8; nel modello i blocchi sono 128 × 128", { x: M, y: 6.35, w: 8, h: 0.4, fontSize: S, color: C.accent4 });

// ---------- 8. RowMask23: block mosaic ----------
s = nuova("La proposta", "Ogni matrice del Transformer è un mosaico di blocchi",
  "STERA 30M ha 6 layer, dimensione 768 e 16 head. Ogni proiezione, compreso l'embedding condiviso con la testa di uscita, è un mosaico di blocchi RowMask23, con guadagni per riga e colonna e un termine low-rank di rango 96. Totale: 30.346.182 parametri addestrabili, contro 82.721.280 valori della forma densa equivalente.");
const b6 = 0.5, xm = M + 0.2, ym = 1.65;
for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++)
  s.addShape(pres.shapes.RECTANGLE, { x: xm + j * b6, y: ym + i * b6, w: b6 - 0.05, h: b6 - 0.05, fill: { color: HEX.stera, transparency: (i + j) % 2 ? 55 : 30 }, line: { type: "none" }, objectName: nome("blocco") });
testo(s, "Proiezione 768 × 768 = 36 blocchi RowMask23", { x: M, y: 4.8, w: 3.6, h: 0.9, fontSize: S, color: C.accent4 });
testo(s, "30.346.182", { x: 4.6, y: 1.6, w: 4.2, h: 0.7, fontSize: T, bold: true, color: C.accent1 });
testo(s, "parametri addestrabili", { x: 4.6, y: 2.3, w: 4.2, h: 0.5 });
testo(s, "82.721.280", { x: 8.75, y: 1.6, w: 4.0, h: 0.7, fontSize: T, bold: true, color: C.accent4 });
testo(s, "valori nella forma densa", { x: 8.75, y: 2.3, w: 4.0, h: 0.5 });
testo(s, [
  { text: "6 layer, dimensione 768, 16 head", options: { bullet: true, breakLine: true, paraSpaceAfter: 10 } },
  { text: "Anche l'embedding è un mosaico", options: { bullet: true, breakLine: true, paraSpaceAfter: 10 } },
  { text: "Più guadagni per riga e colonna e un termine low-rank", options: { bullet: true } },
], { x: 4.6, y: 3.3, w: 8.1, h: 2.4 });

// ---------- 9. training ----------
s = nuova("La proposta", "Training: il gradiente arriva alle coordinate",
  "Il forward ricostruisce le matrici dalle coordinate; il passo all'indietro calcola il gradiente rispetto a ogni matrice e lo porta alle coordinate in forma chiusa ed esatta: la diagonale prende la diagonale del gradiente, la base somma lungo le diagonali XOR, ogni correzione prende la sua posizione. L'optimizer aggiorna solo le coordinate. STERA 30M: 122.000 aggiornamenti, 1,53 miliardi di token, due GPU P100.");
const fl = [["Loss sul token successivo", 0], ["Gradiente rispetto a ogni matrice", 1]];
fl.forEach(([t, i]) => { const y = 1.6 + i * 1.35; riquadro(s, M, y, 6.6, 0.85, { fill: C.background2 });
  testo(s, t, { x: M, y, w: 6.6, h: 0.85, align: "center", valign: "middle", bold: true, color: C.text2 }); });
freccia(s, M + 3.3, 2.45, M + 3.3, 2.95); freccia(s, M + 3.3, 3.8, M + 3.3, 4.3);
["Diagonale", "Base XOR", "Correzioni"].forEach((t, k) => { const x = M + k * 2.25;
  riquadro(s, x, 4.3, 2.1, 0.85, { fill: C.background1, line: HEX.stera });
  testo(s, t, { x, y: 4.3, w: 2.1, h: 0.85, align: "center", valign: "middle", bold: true, color: C.accent1 }); });
testo(s, "Gradiente in forma chiusa, esatto", { x: M, y: 5.3, w: 6.6, h: 0.45, fontSize: S, color: C.accent4, align: "center" });
[["122.000", "aggiornamenti"], ["1,53 mld", "token di training"], ["2 × P100", "GPU NVIDIA"]].forEach(([n, l], k) => {
  const y = 1.6 + k * 1.4; riquadro(s, 7.95, y, 4.8, 1.15);
  testo(s, n, { x: 8.25, y, w: 2.6, h: 1.15, valign: "middle", fontSize: T, bold: true, color: C.accent1 });
  testo(s, l, { x: 10.85, y, w: 1.85, h: 1.15, valign: "middle" });
});

// ---------- 10. inference ----------
s = nuova("La proposta", "Inferenza: la matrice densa non serve",
  "Tre modi di eseguire le stesse proiezioni. Si può tenere la matrice densa in memoria, la si può espandere al bisogno, oppure si applica direttamente alle coordinate: la parte XOR diventa un prodotto elemento per elemento nel dominio della trasformata di Walsh-Hadamard, che costa n log n. In memoria restano solo le coordinate. I token generati coincidono con quelli del motore denso in tutte le prove.");
[["Espansione residente", "in memoria la matrice densa"], ["Espansione al bisogno", "coordinate più una matrice temporanea"],
 ["Prodotto diretto", "solo coordinate, trasformata di Walsh–Hadamard"]].forEach(([h, l], k) => {
  const x = M + k * 4.15, scelto = k === 2;
  riquadro(s, x, 1.6, 3.85, 4.0, { fill: scelto ? C.background2 : C.background1, line: scelto ? HEX.stera : HEX.bordo, lw: scelto ? 2.25 : 1 });
  testo(s, h, { x: x + 0.3, y: 1.8, w: 3.25, h: 0.55, bold: true, color: scelto ? C.accent1 : C.text2 });
  if (k === 0) { s.addShape(pres.shapes.RECTANGLE, { x: x + 1.05, y: 2.6, w: 1.75, h: 1.75, fill: { color: "E5E7EB" }, line: { color: HEX.grigio, width: 1 }, objectName: nome("densa") });
    testo(s, "W", { x: x + 1.05, y: 2.6, w: 1.75, h: 1.75, align: "center", valign: "middle", fontSize: T, color: C.accent4 }); }
  if (k === 1) { s.addShape(pres.shapes.RECTANGLE, { x: x + 0.35, y: 3.0, w: 0.85, h: 0.85, fill: { color: HEX.stera, transparency: 70 }, line: { color: HEX.stera, width: 1 }, objectName: nome("coord") });
    testo(s, "c", { x: x + 0.35, y: 3.0, w: 0.85, h: 0.85, align: "center", valign: "middle", color: C.accent1, bold: true });
    freccia(s, x + 1.3, 3.42, x + 2.05, 3.42);
    s.addShape(pres.shapes.RECTANGLE, { x: x + 2.15, y: 2.75, w: 1.35, h: 1.35, fill: { color: "F3F4F6" }, line: { color: HEX.grigio, width: 1, dashType: "dash" }, objectName: nome("temp") });
    testo(s, "W", { x: x + 2.15, y: 2.75, w: 1.35, h: 1.35, align: "center", valign: "middle", color: C.accent4, bold: true }); }
  if (k === 2) { s.addShape(pres.shapes.RECTANGLE, { x: x + 1.5, y: 3.0, w: 0.85, h: 0.85, fill: { color: HEX.stera, transparency: 70 }, line: { color: HEX.stera, width: 1 }, objectName: nome("coord") });
    testo(s, "c", { x: x + 1.5, y: 3.0, w: 0.85, h: 0.85, align: "center", valign: "middle", color: C.accent1, bold: true }); }
  testo(s, l, { x: x + 0.3, y: 4.55, w: 3.25, h: 0.9 });
});
freccia(s, M, 5.95, W - M, 5.95, C.accent4);
testo(s, "memoria occupata dai pesi: sempre meno", { x: M, y: 6.05, w: 6, h: 0.45, fontSize: S, color: C.accent4 });

// ---------- 11. the model learns ----------
pres.addSection({ title: "Verifica sperimentale" });
s = nuova("Verifica sperimentale", "Il modello impara e generalizza",
  "Prima domanda: il modello impara? La NLL sui racconti di validazione parte da 10,93, il valore di un modello che tira a indovinare, e scende fino a 1,152 al passo 122.000, senza segni di sovradattamento. Sui 300 racconti di test mai visti, STERA 30M ha la NLL più bassa dei tre modelli: 1,174 contro 1,324 di TinyStories-33M e 1,515 di TinyStories-8M.");
const VAL = require(path.join(__dirname, "nll_dev.json"));
const punti = VAL.filter(([p]) => p >= 2000);
s.addChart(pres.charts.LINE, [{ name: "NLL su DEV", labels: punti.map(([p]) => p % 20000 === 0 ? (p / 1000) + "k" : ""), values: punti.map(([, v]) => v) }], {
  x: M, y: 1.5, w: 7.3, h: 4.7, chartColors: [HEX.stera], lineSize: 2.5, lineDataSymbol: "none", showLegend: false,
  showTitle: true, title: "NLL su DEV durante il training", titleFontFace: "+mn-lt", titleFontSize: S, titleColor: HEX.testo,
  valAxisMinVal: 1.1, valAxisMaxVal: 1.9, valAxisLabelFormatCode: "0.0", catAxisLabelFrequency: 1, catAxisOrientation: "minMax", ...assi });
testo(s, "Parte da 10,93 (un modello che tira a indovinare) e arriva a 1,152", { x: M, y: 6.25, w: 7.3, h: 0.45, fontSize: S, color: C.accent4 });
barreModelli(s, [1.515, 1.174, 1.324], { x: 8.3, y: 1.5, w: 4.45, h: 4.7, showTitle: true, title: "NLL su 300 racconti di test", titleFontFace: "+mn-lt",
  titleFontSize: S, titleColor: HEX.testo, dataLabelFormatCode: "0.000", valAxisMinVal: 0, valAxisMaxVal: 1.8 });
testo(s, "Più bassa è meglio", { x: 8.3, y: 6.25, w: 4.45, h: 0.45, fontSize: S, color: C.accent4 });

// ---------- 12. stories ----------
s = nuova("Verifica sperimentale", "Sa anche raccontare? Giudizio cieco su 900 testi",
  "Non basta che la NLL scenda: il modello deve scrivere racconti coerenti. Ogni modello ha continuato gli stessi 300 prompt, con la stessa procedura di generazione. I 900 testi, mescolati e anonimi, sono stati valutati da un giudice automatico, Claude Opus 5.5, con una rubrica fissata in anticipo e senza sapere quale modello li avesse scritti. STERA 30M scrive un racconto corretto nel 44% dei casi: 22,3 punti più di TinyStories-8M, differenza significativa, e 4,3 punti meno di TinyStories-33M, che ha più del doppio dei parametri, differenza non significativa.");
riquadro(s, M, 1.55, 6.3, 3.75);
testo(s, "Un racconto di STERA 30M (in grigio il prompt dato al modello)", { x: M + 0.3, y: 1.7, w: 5.7, h: 0.4, fontSize: S, bold: true, color: C.accent1 });
testo(s, [
  { text: "Once upon a time, there was an adorable whale. The whale lived in the big, blue sea. ", options: { color: HEX.grigio, italic: true } },
  { text: "The whale had many friends, like fish and crabs. They played all day in the water. One day, the whale saw a big rock. The whale wanted to see what was behind the rock. The whale swam around the rock and got closer. The whale was very curious. As the whale swam to the rock, the rock moved! But it was not a rock. It was a big turtle! The turtle was nice and played with the whale. They became best friends and had lots of fun in the water. And the whale was very happy." },
], { x: M + 0.3, y: 2.2, w: 5.7, h: 4.0, fontSize: S, lineSpacingMultiple: 1.1 });
barreModelli(s, [21.7, 44.0, 48.3], { x: 7.2, y: 1.5, w: 5.55, h: 4.0, showTitle: true, title: "Racconti giudicati corretti (%)", titleFontFace: "+mn-lt",
  titleFontSize: S, titleColor: HEX.testo, dataLabelFormatCode: "0.0", valAxisMinVal: 0, valAxisMaxVal: 60 });
testo(s, [
  { text: "+22,3 punti su TinyStories-8M (significativo)", options: { breakLine: true } },
  { text: "−4,3 punti su TinyStories-33M (non significativo)" },
], { x: 7.2, y: 5.6, w: 5.55, h: 0.8, fontSize: S, color: C.text1 });

// ---------- 13. memory ----------
pres.addSection({ title: "Costi di esecuzione" });
s = nuova("Costi di esecuzione", "Memoria: più capace in meno byte",
  "I blocchi RowMask23 garantiscono l'80,6% di valori in meno sulla parte strutturata; con scale, guadagni e low-rank STERA 30M occupa 121 MB in FP32, il 63% in meno della propria forma densa e il 56% in meno di TinyStories-33M. Il termine low-rank si quantizza a 8 bit senza perdita significativa (NLL +0,007%, giudizio non significativamente diverso): i pesi scendono a 86 MB, il 74% in meno della forma densa.");
s.addChart(pres.charts.BAR, [{ name: "MB", labels: ["TinyStories-8M", "STERA 30M, rango 8 bit", "STERA 30M", "TinyStories-33M", "STERA 30M, forma densa"], values: [78.8, 86.1, 121.4, 274.1, 330.9] }], {
  x: M, y: 1.5, w: 7.4, h: 4.8, barDir: "bar", chartColors: [HEX.stera], barGapWidthPct: 45, showValue: true, dataLabelPosition: "outEnd",
  dataLabelFormatCode: "0.0", dataLabelColor: HEX.testo, showLegend: false, valAxisHidden: true, showTitle: true, title: "Pesi FP32 (MB)",
  titleFontFace: "+mn-lt", titleFontSize: S, titleColor: HEX.testo, catAxisOrientation: "maxMin", valAxisMaxVal: 400, ...assi, valGridLine: { style: "none" } });
[["−63%", "byte rispetto alla stessa rete in forma densa; −80,6% garantito sui blocchi"], ["86 MB", "con il low-rank a 8 bit, senza perdita significativa"]].forEach(([n, l], k) => {
  const y = 1.6 + k * 2.4; riquadro(s, 8.4, y, 4.35, 2.1);
  testo(s, n, { x: 8.7, y: y + 0.25, w: 3.8, h: 0.7, fontSize: T, bold: true, color: C.accent1 });
  testo(s, l, { x: 8.7, y: y + 1.0, w: 3.8, h: 1.0 });
});

// ---------- 14. timing, small models ----------
s = nuova("Costi di esecuzione", "Tempi: più veloce della propria forma densa",
  "Stesso modello, stessi pesi, due motori sul Mac M1: il motore che lavora sulle coordinate riduce del 34,8% il tempo per token rispetto a quello con le matrici dense, con token identici in tutte le 30 coppie, e il vantaggio resta costante da 32 a 384 token di ingresso. Eseguendo i tre modelli con lo stesso motore, STERA 30M è più veloce di TinyStories-33M e più lento di TinyStories-8M, che fa circa un quarto dei calcoli. Con il low-rank a 8 bit arriva a 242 token al secondo. Il TTFT resta più lento del denso; espandere le matrici solo durante il prefill dimezza il divario.");
riquadro(s, M, 1.6, 4.6, 4.6);
testo(s, "−34,8%", { x: M + 0.35, y: 1.9, w: 3.9, h: 0.8, fontSize: T, bold: true, color: C.accent1 });
testo(s, "tempo per token rispetto allo stesso modello con matrici dense", { x: M + 0.35, y: 2.8, w: 3.9, h: 1.3 });
testo(s, "Token identici in 30 coppie su 30. Mac M1, MLX/Metal, 256 token in ingresso", { x: M + 0.35, y: 4.6, w: 3.9, h: 1.2, fontSize: S, color: C.accent4 });
barreModelli(s, [311.9, 201.0, 158.0], { x: 5.6, y: 1.5, w: 7.15, h: 4.3, showTitle: true, title: "Token generati al secondo, stesso motore MLX", titleFontFace: "+mn-lt",
  titleFontSize: S, titleColor: HEX.testo, dataLabelFormatCode: "0.0", valAxisMinVal: 0, valAxisMaxVal: 380 });
testo(s, "Con il low-rank a 8 bit: 242 token/s e richiesta completa in 0,47 s", { x: 5.6, y: 5.9, w: 7.15, h: 0.5, fontSize: S, color: C.text1 });

// ---------- 15. timing, large models ----------
s = nuova("Costi di esecuzione", "Quando il modello non entra in memoria",
  "Su modelli grandi il vantaggio diventa decisivo. Il prototipo F10, una versione più grande di STERA non addestrata e usata solo per misurare i costi, occupa 4 GB in forma compatta e resta in memoria: genera circa 12 token al secondo. La sua forma densa occuperebbe 17,3 GiB, più dei 16 GiB del Mac: andrebbe letta dal disco, 7,8 secondi per passaggio. Qwen3.5-9B, un modello reale da 16,7 GiB, eseguito leggendo i layer dal disco produce 0,15 token al secondo. Restare in memoria vale circa due ordini di grandezza.");
const scala = 6.2 / 20, x0 = 3.55, righe = [["F10 compatto", 3.73, HEX.stera, "12,3 token/s", "in memoria"],
  ["Qwen3.5-9B (BF16)", 16.68, HEX.ts33, "0,15 token/s", "layer letti dal disco"], ["F10 forma densa", 17.32, HEX.grigio, "≤ 0,13 token/s", "7,8 s per leggere i pesi"]];
testo(s, "Pesi (GiB)", { x: x0, y: 1.5, w: 3, h: 0.4, fontSize: S, color: C.accent4 });
testo(s, "Generazione", { x: 10.25, y: 1.5, w: 2.5, h: 0.4, fontSize: S, color: C.accent4 });
righe.forEach(([nm, v, col, tps, nota], k) => {
  const y = 2.05 + k * 1.3;
  testo(s, nm, { x: M, y, w: 2.85, h: 0.75, valign: "middle", bold: true, color: C.text2 });
  s.addShape(pres.shapes.RECTANGLE, { x: x0, y: y + 0.1, w: v * scala, h: 0.55, fill: { color: col, transparency: 15 }, line: { type: "none" }, objectName: nome("barra") });
  testo(s, v.toFixed(2).replace(".", ","), { x: x0 + v * scala + 0.1, y, w: 0.9, h: 0.75, valign: "middle", fontSize: S, color: C.text1 });
  testo(s, [{ text: tps, options: { bold: true, breakLine: true, color: k === 0 ? HEX.stera : HEX.testo } }, { text: nota, options: { fontSize: S, color: HEX.grigio } }],
    { x: 10.25, y: y - 0.05, w: 2.6, h: 0.95, valign: "middle" });
});
s.addShape(pres.shapes.LINE, { x: x0 + 16 * scala, y: 1.85, w: 0.001, h: 3.95, line: { color: HEX.navy, width: 1.75, dashType: "dash" }, objectName: nome("limite") });
testo(s, "16 GiB di memoria", { x: x0 + 16 * scala - 1.0, y: 5.85, w: 2.0, h: 0.4, fontSize: S, color: C.text2, align: "center" });
testo(s, "F10: prototipo più grande di STERA, non addestrato, usato solo per misurare i costi di esecuzione", { x: M, y: 6.35, w: 9.5, h: 0.45, fontSize: S, color: C.accent4 });

// ---------- 16. limitations and future work ----------
pres.addSection({ title: "Conclusioni" });
s = nuova("Conclusioni", "Limiti e sviluppi futuri",
  "Limiti: un solo training con un solo seme; i modelli di riferimento hanno dati e budget diversi; il vantaggio è sui byte da memorizzare e trasferire, non sul numero di operazioni; l'elaborazione del prompt è più lenta del motore denso. Sviluppi: un confronto controllato con stessi dati e più semi, modelli più grandi e altri domini, un kernel diretto anche per il prompt, coordinate quantizzate già in training.");
[["Limiti", ["Un solo training, un solo seme", "Riferimenti con dati e budget diversi", "Studi delle componenti a posteriori", "Misure su un solo dispositivo"], HEX.ts33],
 ["Sviluppi futuri", ["Confronto controllato: stessi dati, più semi", "Rango scelto prima del training", "Prefill diretto sulle coordinate", "Coordinate quantizzate in training"], HEX.stera]].forEach(([h, voci, col], k) => {
  const x = M + k * 6.25; riquadro(s, x, 1.6, 5.85, 4.75);
  testo(s, h, { x: x + 0.4, y: 1.85, w: 5.0, h: 0.6, bold: true, color: col });
  testo(s, voci.map((v, i) => ({ text: v, options: { bullet: true, breakLine: i < voci.length - 1, paraSpaceAfter: 14 } })), { x: x + 0.4, y: 2.65, w: 5.1, h: 3.5 });
});

// ---------- 17. thanks ----------
s = pres.addSlide({ masterName: "CHIUSURA", sectionTitle: "Conclusioni" });
s.addText("Grazie per l'attenzione", { placeholder: "title" });
s.addText([{ text: "Andrea Nardoni", options: { breakLine: true } }, { text: "Relatore: Prof. Fabio Gasparetti" }], { placeholder: "body" });
s.addNotes("Grazie. Sono a disposizione per le domande.");

(async () => {
  const out = path.join(__dirname, "STERA_graduation_talk.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log(out);
})();
