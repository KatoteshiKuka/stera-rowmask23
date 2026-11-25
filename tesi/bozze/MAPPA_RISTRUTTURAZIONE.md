# Mappa della ristrutturazione della tesi (passo 5)

Numerazione dei capitoli come nel PDF. Per ogni punto: **togliere**, **aggiungere**, **aggiornare**.
Tutti i numeri nuovi vengono dai file in `valutazione/varianti/` e `sistemi/risultati/campagne_20261124/`.

## Sommario (italiano e inglese)
- **Aggiornare** "il tempo per token si riduce del 27%" → valore attuale (−34,8% contro la forma densa).
- **Aggiungere** una frase: il formato garantisce −80,6% sulla parte strutturata; con il basso rango a 8 bit STERA 30M scende a 86 MB (−74% rispetto alla forma densa) con NLL invariata (+0,007%) e decode più rapido.

## Introduzione
- **Aggiornare H2**: "la parametrizzazione garantisce −80,6% sulla parte strutturata; il risparmio complessivo dipende dal rango (−63,3% per STERA 30M)".
- **Aggiornare "Risultati principali"**: già senza INT8 delle coordinate e senza archivio; aggiungere quantizzazione del basso rango (byte, NLL, giudizio, tempi), confronto alla pari su MLX, prefill con espansione al volo.
- **Aggiornare "Organizzazione del testo"** con le nuove sezioni.

## Cap. 3 – Stato dell'arte
- **Aggiungere** nella sezione sulla quantizzazione e in "Dove si colloca RowMask23" una frase: struttura e quantizzazione agiscono su grandezze diverse (numero di valori e bit per valore) e si possono combinare; la tesi lo verifica sul basso rango.

## Cap. 4 – La parametrizzazione RowMask23
- **Aggiungere** la nuova sezione *Bilancio dei parametri* (bozza `passo2_bilancio_parametri.tex`): definizione della proiezione STERA, proposizione sul risparmio $s(m,n,r)$, limiti 80,61% / 79,05%, strato con $r=\kappa d$ ($80{,}61\% - 1{,}5\kappa$), tabella della crescita e **nuova figura** del risparmio al variare di $d$ e del rango.

## Cap. 5 – STERA: parametrizzazione e implementazione
- **Aggiornare** "Come si costruiscono le matrici": frase esplicita "tutti i parametri, compreso il basso rango, sono memorizzati e addestrati in FP32"; rimando alla tabella delle componenti (coordinate 51,8%, basso rango 42,4%).
- **Aggiornare** "Una generazione, dall'ingresso al token successivo" (prefill): il prefill può usare l'*espansione al bisogno* strato per strato (già presente nella figura delle tre strategie) e il decode il *prodotto diretto*; spiegare perché (prefill limitato dal calcolo, decode dalla lettura dei pesi).

## Cap. 6 – Metodologia di validazione
- **Aggiungere** in "Misure di memoria e tempo": protocollo delle 5 campagne a sistema scarico; confronto alla pari con i TinyStories nello stesso motore denso MLX (parità dei token verificata) accanto al motore standard Transformers; lunghezze dell'input da 32 a 384 token; blocco di prefill.
- **Aggiungere** una sezione *Analisi del termine a basso rango*: riduzione a posteriori con la SVD (k = 64…0), quantizzazione a 8 bit (schema e formato), errore sulla matrice densa equivalente, protocollo del giudizio cieco per le varianti (insiemi mescolati, stesso giudice e stessa rubrica, sessioni separate).

## Cap. 7 – Risultati: apprendimento e qualità generativa
- **Aggiungere** la sezione *Contributo del termine a basso rango*: tabella rango → NLL → racconti corretti (96: 44,0%; 64: 20,0%; ≤48: 0%) e **nuova figura**; il modello addestrato usa tutto il rango.
- **Aggiungere** la variante con basso rango a 8 bit nella tabella del giudizio, come riga a parte con la nota "valutata in una sessione separata con lo stesso giudice e la stessa rubrica" (41,3%, differenza −2,7 punti, p = 0,52) e NLL (+0,007% nel formato del motore).

## Cap. 8 – Risultati: memoria e tempi di esecuzione
- **Togliere** le righe INT8 delle coordinate e l'archivio senza perdita dalla tabella e dal testo; titolo della sezione "Peso di STERA 30M".
- **Aggiungere** nella sezione sul peso: basso rango a 8 bit, errore 0,48% (formato MLX) e 0,61% sulla matrice densa equivalente (schema per colonna), 86,1 MB, −74,0% rispetto alla forma densa.
- **Sostituire** la tabella dei tre modelli con il confronto alla pari nello stesso motore MLX (STERA compatto, STERA denso, TinyStories-8M, TinyStories-33M), tenendo il motore standard Transformers come riferimento d'uso.
- **Aggiungere** lo studio al variare del contesto (tabella e **figura**: decode costante, riduzione −35,4…−36,3%; primo token).
- **Aggiungere** lo studio del prefill: causa (proiezioni compatte 56 ms contro 19 ms su 256 token), blocco 64, espansione al volo strato per strato (risultati delle campagne in corso).
- **Aggiungere** i tempi della variante con basso rango a 8 bit (decode 198,9 → 241,9 token/s, −17,8%; primo token invariato; memoria GPU 137 → 104 MiB; picco del processo invariato).
- **Aggiornare** la Sintesi.

## Cap. 9 – Conclusioni
- **Aggiornare** la tabella delle ipotesi (H2 e H3) e i paragrafi.
- **Aggiungere** il paragrafo "STERA e la quantizzazione".
- **Aggiornare i limiti**: contributo del rango all'apprendimento misurato solo a posteriori; misure di sistema su un solo dispositivo; giudizio delle varianti in sessioni separate; togliere o ridimensionare "prompt più lento" alla luce del prefill con espansione.
- **Aggiornare gli sviluppi futuri**: rango scelto prima dell'addestramento ($\kappa$), quantizzazione delle coordinate in addestramento, porting su GPU NVIDIA.

## Appendice
- **Aggiornare** "Riproducibilità" con i nuovi script (`valuta_modello.py --rango-bit/--rango-ridotto`, `stera_rango8_mlx.py`, `prefill_espansione.py`, `gptneo_denso_mlx.py`, campagne).

## Fuori dalla tesi
- README, `docs/RISULTATI.md`, `docs/RIPRODUZIONE.md`, `docs/VERIFICA_RIPRODUCIBILITA.md`.
- Presentazione: slide su memoria, tempi e limiti; eventualmente una slide su rango e quantizzazione.

## Nuove figure
1. Risparmio al variare di $d$ e del rango (Cap. 4).
2. Rango → NLL e racconti corretti (Cap. 7).
3. Decode e primo token al variare della lunghezza dell'input (Cap. 8).
4. Primo token: kernel compatto, espansione al volo, denso (Cap. 8).
