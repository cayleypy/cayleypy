# /home/username/.bun/bin/bun /home/username/Desktop/cayley/cubing.js/src/bin/puzzle-geometry-bin.ts --gap compy cube
# PuzzleGeometry 0.1 Copyright 2018 Tomas Rokicki.
# 
M_M_DRF:=(5,18,10)(6,17,9)(43,45,44);
M_M_UBL:=(7,20,14)(8,19,13)(31,33,32);
M_M_DFL:=(11,22,18)(12,21,17)(34,36,35);
M_M_URB:=(3,24,13)(4,23,14)(40,42,41);
M_M_DBR:=(3,15,6)(4,16,5)(28,30,29);
M_M_ULF:=(1,21,8)(2,22,7)(37,39,38);
M_M_DLB:=(11,16,20)(12,15,19)(46,48,47);
M_M_UFR:=(1,24,9)(2,23,10)(25,27,26);
Gen:=[
M_M_DRF,M_M_UBL,M_M_DFL,M_M_URB,M_M_DBR,M_M_ULF,M_M_DLB,M_M_UFR
];
ip:=[[1],[3],[5],[7],[9],[11],[13],[15],[17],[19],[21],[23],[25],[28],[31],[34],[37],[40],[43],[46]];
# Size(Group(Gen));
# Size(Stabilizer(Group(Gen), ip, OnTuplesSets));

