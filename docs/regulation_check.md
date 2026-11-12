# Regulation check for E2 (decision rule) and E5 (plot outlines)

Read 13 Nov 2026. **Status: waiting for the owner's confirmation against the official text.** Until then nothing in
`src/` changes because of this note.

## Source and how it was read

- The official text is on EUR-Lex (CELEX 32023R1115; latest consolidation listed there is 18 Sep 2026). EUR-Lex
  returned an empty page to the automated reader, so it could not be read here.
- Read instead: the compiled full text at https://lawplayer.com/eu/act/32023R1115, which states it is in force from
  29 June 2023, last amended 18 September 2026, data as of 2 November 2026, and that the EUR-Lex text prevails.
- Article and point numbers below are as given by that compilation. The owner should check each quote and number
  against EUR-Lex before any wording in the README or the product relies on it.

## What the text says (quotes from the compilation)

| Article 2 | Quote |
|---|---|
| (3) deforestation | "the conversion of forest to agricultural use, whether human-induced or not" |
| (4) forest | "land spanning more than 0,5 hectares with trees higher than 5 metres and a canopy cover of more than 10 %, ..." "excluding land that is predominantly under agricultural or urban land use" |
| (5) agricultural use | "the use of land for the purpose of agriculture, including for agricultural plantations and set-aside agricultural areas" |
| (6) agricultural plantation | "it includes all plantations of relevant commodities other than wood;" "agricultural plantations are excluded from the definition of 'forest'" (oil palm plantations are among the examples) |
| (13) deforestation-free | "(a) that the relevant products contain, have been fed with or have been made using, relevant commodities that were produced on land that has not been subject to deforestation after 31 December, 2020; and (b) ..." (wood: harvested without forest degradation) |
| (28) geolocation | "using at least six decimal digits;" ... "for plots of land of more than four hectares used for the production of the relevant commodities other than cattle, this shall be provided using polygons with sufficient latitude and longitude points" |

Article 9(1)(d) asks for the geolocation of all plots of land where the commodities were produced. The compilation
paraphrases it as saying that any deforestation or forest degradation on the given plots disqualifies all commodities
from those plots.

The only percentages in the text that relate to forests are in the definitions: canopy cover over 10 % for forest,
5 to 10 % for other wooded land, and over 50 % of growing stock for planted forest. Article 16(9) sets inspection
coverage for authorities. The compilation shows no percentage tolerance of deforestation on a plot.

## What this would mean for the repo (if the owner confirms)

1. **E2.** The product's levels (loss as a share of the farm's 2020 forest: COMPLIANT up to 5 %, WARNING above 5 %,
   VIOLATION above 10 %; one rule since the merge of `fix/state-machine-bugs`, `classify_forest_loss` in
   `src/change_detector.py`) have no basis in the text: one converted forest area disqualifies the plot. Note that
   "forest" itself needs more than 0.5 ha, which matches the audit's flag at 0.5 ha of loss (`hansen_audit/decisions.py`,
   `farm_flag`). The audit's rule (flag, clear, uncertain) is the closer reading.
2. **The audit's central point holds in the text.** Oil palm plantations are agricultural plantations, which are
   excluded from "forest", so loss of tree cover on land that was already plantation on 31 Dec 2020 is not
   deforestation. This is the reading behind the PLANT share (0.49, 95 % interval 0.30-0.65).
3. **E5.** Plots over 4 ha (commodities other than cattle) must be given as polygons with at least six decimal
   digits. The farm boxes (0.01 degree, about 120 ha) are not plot geolocation in that sense; real outlines are needed
   for any DDS-like export, and for counting loss inside the plot.

## For the owner

- Check the six quotes and the numbering against EUR-Lex (consolidated text of 18 Sep 2026).
- Then decide for E2: replace the percentage levels with the audit's rule, and whether to keep the old levels behind
  a named option for continuity.
