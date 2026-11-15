# E10 diagnosis (15 Nov 2026, before PROTOCOL_E10 was frozen)

Prompted by E9: the label rasters call 3 % of box cells forest in 2020 where GFC2020 calls 20 %.

1. Ten farms drawn with seed 0 from the 996 (IT Cattle, 4 ID Oil Palm, BR Soy, 3 CM Cocoa, GH Cocoa). For each, the label was
   recomputed on Earth Engine with the logic of `generate_labels.py` on `UMD/hansen/global_forest_change_2025_v1_13`, on the
   file's own grid. Cell agreement with the file on disk: 1.000 on all ten. Yet tree cover >= 10 % covers 0.78-1.00 of the box
   on the oil palm and cocoa farms while classes 1 + 2 cover 0.03-0.15.
2. Two farms (relation_10746463 CM Cocoa, way_549536190 ID Oil Palm): `lossyear` is masked on 0.70 and 0.65 of the cells and
   holds no 0 value anywhere (the same with v1_11). Every class-1 cell on disk has lossyear 25. True stable forest
   (tree cover >= 10 %, lossyear masked) is 0.70 and 0.43 of these boxes, all written as class 0.
