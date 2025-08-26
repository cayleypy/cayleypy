# /home/username/.bun/bin/bun /home/username/Desktop/cayley/cubing.js/src/bin/puzzle-geometry-bin.ts --gap dino
# PuzzleGeometry 0.1 Copyright 2018 Tomas Rokicki.
# 
M_M_DRF:=(5,18,10)(6,17,9);
M_M_UBL:=(7,20,14)(8,19,13);
M_M_DFL:=(11,22,18)(12,21,17);
M_M_URB:=(3,24,13)(4,23,14);
M_M_DBR:=(3,15,6)(4,16,5);
M_M_ULF:=(1,21,8)(2,22,7);
M_M_DLB:=(11,16,20)(12,15,19);
M_M_UFR:=(1,24,9)(2,23,10);
Gen:=[
M_M_DRF,M_M_UBL,M_M_DFL,M_M_URB,M_M_DBR,M_M_ULF,M_M_DLB,M_M_UFR
];
ip:=[[1],[3],[5],[7],[9],[11],[13],[15],[17],[19],[21],[23]];
# Size(Group(Gen));
# Size(Stabilizer(Group(Gen), ip, OnTuplesSets));

