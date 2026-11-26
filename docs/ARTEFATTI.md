# Impronte SHA-256 degli artefatti

## Esterni (non versionati)

| Artefatto | SHA-256 |
|---|---|
| Corpus TinyStoriesV2-GPT4-train.txt (revisione f54c09fd) | `6418d412de72888f52b5142c761ac21a582f7d1166f0bfbdb5f03ccfdec90443` |
| Tokenizer GPT-Neo (TinyStories-33M, revisione 2ad0a164) | `f6ed3d307010c244c22aeffbde05f419cf277c23e64cf98b673cac5449cfeff5` |
| Corpus tokenizzato: pacchetto / indice | `5ae58177e37ddc66f22bc5ac26c73b82d9b7b2702a1871a5cbfd517c1ae39066` / `bbadffc0b26a2dcf13c396691880e9655a8646212356317c95a96e8d70c69774` |
| STERA 30M, checkpoint finale (122.000 aggiornamenti) | `418becfbcaddcefd56f828f295138bbc865f9405a9060930c060ff7e6ced0887` |
| Pesi FP32 del checkpoint finale per le misure di sistema (`H122000_fp32_weights.npz`) | `864d03dc63b5d5a8977eba0c7c8bec60530a33a1ef52762ab5b96aeb2201018c` |
| Matrici dense equivalenti del checkpoint finale (`H122000_DENSE_FP32.safetensors`) | `ce96ce4b54a7a094279d9c7f6447a0563edb8142d89634e8de4da1e624e52f7f` |
| TinyStories-33M, pytorch_model.bin | `41316d3bad3cf7766cbc36443a71c5ef05321a1c4fa14855df537998c6dad302` |
| TinyStories-8M, pytorch_model.bin | `22c355bfabebc1f6c861b3f5d7a801e96c7f6da4af4bb0f7780096ab82ea6716` |

## Nel repository

| File | SHA-256 |
|---|---|
| `valutazione/protocollo/incipit_test300.json` | `95bcffdbc25502ddc137eccf94eb935b805c9478989b46902bae4522902032ec` |
| `valutazione/generazioni/stera30m.jsonl` | `ad13df91561a71c4e2741859626552dd69472a80ee320ae08bef80cfd10b751a` |
| `valutazione/generazioni/tinystories33m.jsonl` | `0b465edd6f47e352b26046a0a883a6fb0d65ea156267d99a828c238f5bd2f414` |
| `valutazione/generazioni/tinystories8m.jsonl` | `5d8de12c38d3366ede9c99227cccf2d66ccb685b6b36d1ddccfb479a95b2dea4` |
| `valutazione/giudizio_cieco/racconti_anonimi_900.json` | `9d90fb3ed1d834d04921c265e41337a8a505b8f3819c38cc2f29916a21724f76` |
| `valutazione/giudizio_cieco/abbinamento_modelli.json` | `6cbb11347d20abc92b7bf5bc4c719d90697f59f402b4122e53bac2931ab89235` |
| `valutazione/risultati_3modelli.json` | `369de10eca2a058b72504310dc9521ab5f7b1c55ab3f3dfd0cf1cbc7d14bda87` |
| `addestramento/log_addestramento_0_122k.txt` | `f5cb546dd9bcf330db910be0bcbfce6dc3cd85ff7b3b1e8cc3a33c7c8462ca4c` |

Impronta dei 900 giudizi (calcolata da `aggrega_e_decodifica.py` sul contenuto ordinato): `40760acaa40fbe7c68bbe165aab757fbc72d7820de8b353a560e2f3ba4a5e8ea`.

## Misure di sistema della tesi

| File | SHA-256 |
|---|---|
| `sistemi/risultati/campagne_20261124/protocollo/riepilogo.json` (tempi e memoria, Capitolo 8) | `6e1afd9f5c4f05284a52441c43180f385b3d8087499656503d33c8a96a16bac1` |
| `sistemi/modelli/PROMPTS_FRESH_TEST100.json` (ingressi delle misure di tempo) | `362f0c50045175cd141b422f80376e303ecc5b42d14496bd4adc7ff5af8e656f` |

I file `.npz` e `.safetensors` si rigenerano dal checkpoint finale con `esporta_pesi.py` ed `esporta_denso.py`; il controllo determinante è l'uguaglianza bit per bit dei tensori, che gli script verificano.
