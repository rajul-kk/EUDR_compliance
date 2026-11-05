# Scoop check (2026-11-05)

Evidence tags: **abstract** = arXiv abstract page only; **full PDF** = full text read;
**README** = Zenodo record/README/metadata read; **snippet** = search-result snippet only.
No source below was read as a full scientific PDF except where stated; claims about
methods inside papers that I only saw as abstracts are NOT made.

| Source | What I read | What it establishes | Overlap with this audit |
|---|---|---|---|
| arXiv 2604.23776, Kuapanich et al., oil palm maps Malaysia/Indonesia 2020-2024 | abstract (arXiv page) + Zenodo 17768444 file list (README-level: 3 GeoTIFFs 2020/2022/2024, ~700 MB each, CC BY 4.0) | U-Net + determinant mutual information trained on noisy 100 m historical labels; OA 70.64 / 63.53 / 60.06 % (2020/2022/2024) on 2,058 manual points. Neither Hansen nor EUDR mentioned in the abstract. | Same region, noisy-label theme, but labels noised are historical oil-palm maps, not Hansen loss; no farm-level compliance decisions. Their maps are themselves only 60-71 % accurate, so they are a weak reference (see data_inventory/protocol). |
| arXiv 2602.17372, Jiang et al., tree crops South America | abstract only | 10 m tree-crop map; 23 % of tree-crop area linked to 2000-2020 forest loss; says EUDR-supporting maps often classify established agriculture/agroforestry as forest. No Hansen mention in abstract, no accuracy numbers seen. | Supports the premise (plantation-as-forest) in a different continent; does not quantify Hansen loss noise or farm decisions. |
| arXiv 2509.08303, Wafiq et al., oil palm benchmark Indonesia | abstract + arXiv HTML summary (fetched via summariser, not a line-by-line read) + Zenodo 23874371 record metadata/file list (README-level) | 52,225 polygons / 259,991 ha, Riau and West Sulawesi (v0.1); validation 4,800 CEO plots, OA 0.83, kappa 0.76, oil palm PA 0.96 / UA 0.88. No Hansen comparison performed. Note: the latest Zenodo version (23874371, 2026-11-02) has 72 cells / 47,646 polygons over ten provinces and is licensed **CC BY-NC-SA 4.0**, not CC BY 4.0 as in the arXiv abstract. | Best available reference polygons; licence discrepancy must be respected (non-commercial). |
| GIZ/EEAS report (Vietnam case study of GFC2020 v2) | full PDF downloaded (19 MB), text extracted, key tables read (Table 8, executive summary) | Vietnam, 4 pilot communes. GFC2020 v2 (a *forest-extent* map, not Hansen loss) OA 65.67 %, forest PA 60.6 %, UA 72.6 %; >80 % in natural forest but ~20 % in fragmented/short-rotation plantations. | Not Hansen GFC loss, not Malaysia/Indonesia. Same failure mode (plantation vs forest). Do not quote "65 %" as Hansen accuracy. |
| Search: Hansen GFC accuracy on oil palm | search snippets only (Jakarta Post 2013, Ekuatorial, a Nature Sustainability 2022 listing, one unvetted self-hosted page) | Hansen states the product measures tree cover loss, not land use; tree-cover loss includes plantation harvest. Ad hoc figures (e.g. 43 % of Indonesian loss inside plantation) came from an unvetted page and are **not used**. | Confirms conceptual noise source; I found no peer-reviewed quantitative test of Hansen loss against oil palm polygons. Absence of evidence from a limited search, not proof. |

## Verdict
**Partly scooped on premise, not on the specific question.** The qualitative claim (global tree-cover/forest maps misread
plantations) is established. Not found in what I read: (a) a Hansen loss-label transition-matrix estimate against oil palm
reference polygons, (b) noise-robust training evaluated on that, (c) farm-level false-flag/missed-flag rates with bootstrap
intervals on an EUDR farm list. Limits: web search was shallow (a handful of queries), only abstracts for two papers;
I could have missed a closer paper. A deeper search (Google Scholar, "GFC" + "oil palm" + "commission error") is recommended
before any preprint claim of novelty.
