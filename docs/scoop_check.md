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

## Second pass (2026-11-13, U3)

Same evidence tags. Added **catalog** = dataset catalogue page read. Searches: Hansen GFC loss with oil palm rotation,
replanting and commission error in Indonesia; EUDR farm screening with Hansen false positives; forest-loss driver maps.
No Google Scholar access from here, and PMC returned a browser check, which was not worked around.

| Source | What I read | What it establishes | Overlap with this audit |
|---|---|---|---|
| Sims et al. 2025, "Global drivers of forest loss at 1 km resolution", Environ. Res. Lett. 20: 074027, doi:10.1088/1748-9326/add606 (WRI / Google DeepMind, CC BY 4.0) | catalog (Earth Engine page for v1.2, 2001-2024) | Classes loss drivers at 1 km. The page says the product "does not distinguish between the loss of natural forest and planted trees" and that "clearing and replanting an orchard would be included in the permanent agriculture class", which "is not deforestation of a natural forest". | Closest. It names the same confusion and leaves it inside one class; it does not measure how much loss is replanting. Its "permanent agriculture" share for Indonesia cannot be read as deforestation for oil palm. Accuracy figures are in the paper, not read. |
| "High-resolution maps show that rubber causes substantial deforestation", Nature 2023, doi:10.1038/s41586-023-06642-z (first author recalled as Wang; not checked) | snippet | States that rotational clearing and replanting of plantations and tree crops may be wrongly classed as deforestation, likely most in Indonesia, Malaysia and Thailand. | Premise, for rubber; no measured share of Hansen loss on existing plantations seen. |
| Riau oil palm age structure 1990-2020 (authors not seen), Environ. Res. Lett., doi:10.1088/1748-9326/ac8a61 | snippet | About 44 % of oil palm 20 years after planting had been replanted (20 %) or converted (23 %); says age data help interpret gain and loss in Hansen. | Supports the mechanism (replanting cycles) in one province; no Hansen confusion estimate seen. |
| Comment on Hansen et al. 2013, Science 2014 (first author recalled as Tropek; not checked) | snippet (ResearchGate listing) | Early examples of plantations mapped as forest and their harvest as loss. | Premise only. |
| "Slowing deforestation in Indonesia follows declining oil palm expansion and lower oil prices" (PMC8963565; authors recalled as Gaveau et al. 2022, not checked) | not read (browser check) | Unknown here. Known to map industrial plantation expansion separately from forest loss; whether it reports Hansen loss inside existing plantations was not checked. | **Read before any novelty claim.** |
| Vendor and advocacy pages (Nadar, TraceBean, tracextech), a MODIS loss-detection study, EUSPA coffee report | snippets | Repeat that Hansen loss is not EUDR deforestation; a cited Forclime figure (deforestation overestimated "by up to 347 %") is second-hand. | Not used for numbers. |

### Verdict after the second pass

Unchanged in substance, firmer on premise. The confusion between plantation replanting and deforestation in Hansen loss
is stated in a 2025 driver map's own documentation and in a 2023 Nature paper; the audit must cite both and must not
present the confusion itself as new. Not found in what was read: a measured, confusion-corrected share of Hansen loss on
land that was already plantation at the EUDR cut-off, against independent reference polygons, with farm-level flag
rates. Before a preprint: read PMC8963565 and the Sims et al. 2025 accuracy section in full, and run a Google
Scholar search for "tree cover loss" with "oil palm" and "replanting".
