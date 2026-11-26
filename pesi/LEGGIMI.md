# Pesi

I pesi non sono versionati su GitHub. Questa cartella contiene, sulla macchina dell'autore:

| File | Contenuto | SHA-256 |
|---|---|---|
| `stera30m_step122000.ckpt.gz` | Checkpoint finale di STERA 30M (122.000 aggiornamenti), con pesi compatti e stato dell'ottimizzatore | `418becfbcaddcefd56f828f295138bbc865f9405a9060930c060ff7e6ced0887` |

Per riottenerlo:
- scaricarlo dalla Release `v1.0` del repository (file `stera30m_step122000.ckpt.gz`) e metterlo in questa cartella:
  `gh release download v1.0 --pattern stera30m_step122000.ckpt.gz --dir pesi`
- oppure rieseguire `addestramento/addestra_stera30m.sh` (circa 35 ore su due P100) e copiare `TS68V2_STEP122000.ckpt.gz` qui con questo nome.
Gli script verificano l'impronta prima di caricare il file.
