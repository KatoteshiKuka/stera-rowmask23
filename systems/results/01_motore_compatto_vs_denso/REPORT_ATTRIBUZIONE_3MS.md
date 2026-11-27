> **Misure precedenti, non usate nella tesi.** Questo report riguarda i pesi del checkpoint a 80.000 aggiornamenti. I tempi e la memoria riportati nella tesi provengono dal checkpoint finale: `sistemi/risultati/campagne_20261124/protocollo/riepilogo.json`.

# Da dove viene il vantaggio nel decode di STERA H80000

## Risultato

Nel decoder controllato il percorso solo Dense impiega in media **9,45 ms/token** e il percorso tutto STERA **6,67 ms/token**. La differenza è **2,78 ms/token**. La mediana delle differenze appaiate è **2,75 ms/token**. Questa campagna misura il decode dopo lo stesso prefill Dense per tutti i bracci. La misura end-to-end con prefill ottimizzato separatamente è disponibile nel [report precedente](REPORT_H80000_MOTORI_OTTIMIZZATI.md).

| Gruppo convertito al percorso compatto | Payload Dense (MB) | Payload compatto (MB) | Risparmio attribuito (ms/token) |
|---|---:|---:|---:|
| Embedding e proiezione finale sul vocabolario, matrice condivisa | 154,53 | 50,99 | 2,17 |
| Proiezioni FFN di salita e discesa nei sei strati | 113,25 | 40,72 | 0,56 |
| Proiezioni Q, K, V e O dei sei strati | 56,62 | 25,73 | 0,04 |

Un valore negativo nella colonna del risparmio indica un aumento del tempo. I contributi sono differenze di tempo del decoder intero durante gli interventi. La somma coincide con il vantaggio medio complessivo prima degli arrotondamenti.

## Meccanismo

La proiezione di uscita deve calcolare un punteggio per ogni voce del vocabolario. La sua matrice FP32 è di 50.304 × 768 elementi, con maschera sulle voci in padding. In uscita il Dense moltiplica la matrice esplicita per il vettore nascosto. STERA lavora sulle coordinate RowMask23, sui termini Walsh–Hadamard e sulla correzione di rango 96. È lo stesso peso condiviso con l'embedding in ingresso. La tabella attribuisce l'effetto al gruppo congiunto ingresso/uscita. L'ingresso legge la riga del token, mentre l'uscita proietta su tutto il vocabolario.

La stima dei payload usa le forme dei tensori FP32. Nel compatto include le 3.176 coordinate, la scala e i 128 coefficienti WHT per tile, i guadagni e i fattori di rango 96. Esclude bias condivisi, cache KV e temporanei. Questi byte descrivono i dati dei pesi accessibili dall'operatore. I trasferimenti effettivi dalla memoria unificata, il riuso nelle cache e il tempo interno dei singoli kernel richiedono contatori GPU. Il blocco di attenzione conserva SDPA in tutte le varianti. Il ramo Dense usa QKV fusa, il ramo STERA tre proiezioni distinte. Il contributo di quel gruppo include questa differenza di implementazione.

## Disegno sperimentale

Stessi pesi effettivi H80000, FP32, GPU M1, MLX/Metal, batch 1, cache KV di capacità 512, greedy. Per ciascuno di sei input da 256 token si crea una sola cache iniziale condivisa. Si provano tutte le otto combinazioni dei tre gruppi, alternandole in ordine pseudocasuale, con due ripetizioni e 100 passi sincronizzati dopo un warm-up. In tutte le 96 misure, tutti i 100 ID coincidono con il riferimento Dense sullo stesso input. I due formati dei pesi sono caricati nello stesso processo per alternare rapidamente i bracci. Il download Qwen era attivo durante la prova.

L'attribuzione, per ciascuna ripetizione, calcola il guadagno marginale di ogni gruppo nei sei ordini possibili di sostituzione. Fa poi la media degli ordini e delle 12 ripetizioni appaiate. Questa procedura ripartisce fra i gruppi le interazioni osservate, per esempio quando due sostituzioni insieme producono un effetto diverso dalla somma delle sostituzioni singole.

| Gruppi compatti | Mediana ms/token |
|---|---:|
| vocab | 8,014 |
| attention+vocab | 7,141 |
| ffn | 8,887 |
| dense | 9,424 |
| attention+ffn | 9,014 |
| ffn+vocab | 6,674 |
| attention | 9,373 |
| attention+ffn+vocab | 6,467 |

[Script del decoder controllato](../../benchmark/attribuzione.py) · [96 misure grezze](attribuzione_grezzi.jsonl) · [Calcolo dell’attribuzione](../../benchmark/analizza_attribuzione.py) · [Contributi e dati appaiati](attribuzione_effetti.json)

## Controllo aggiuntivo della testa sul vocabolario

Tenendo FFN e proiezioni di attenzione in versione STERA, abbiamo eseguito quattro configurazioni: ingresso e uscita Dense, solo ingresso compatto, solo uscita compatta, entrambi compatti. Le 24 misure sui sei input conservano tutti i 100 ID. L'effetto medio della proiezione finale è **+2,508 ms/token** di risparmio e quello della sola lettura embedding è **-0,387 ms/token**. È una verifica in uno specifico contesto di operatori e spiega dove cade il guadagno del gruppo vocabolario. Le stime vengono da una campagna distinta rispetto alla tabella principale.

Il kernel compatto riordina i residui per permettere letture contigue tra thread contigui, divide la riduzione sulle colonne tra più gruppi GPU e calcola il contributo strutturato con WHT. La correzione di rango 96 si valuta come due prodotti più piccoli. Queste operazioni fanno parte dell'intervento misurato. Il dato di 154,53 contro 50,99 MB rappresenta la quantità di pesi per le due implementazioni della stessa testa e offre un meccanismo compatibile con il guadagno osservato.

[Script del controllo](../../benchmark/isola_head.py) · [Dati](head_grezzi.jsonl) · [Riepilogo](head_riepilogo.json)
