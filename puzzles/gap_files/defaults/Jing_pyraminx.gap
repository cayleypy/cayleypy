# /home/username/.bun/bin/bun /home/username/Desktop/cayley/cubing.js/src/bin/puzzle-geometry-bin.ts --gap Jing pyraminx
# PuzzleGeometry 0.1 Copyright 2018 Tomas Rokicki.
# 
M_M_F:=(1,6,8)(2,4,9)(3,5,7)(13,16,19)(14,15,20);
M_M_B:=(10,12,11)(17,23,22)(18,24,21)(26,28,27);
M_M_D:=(1,7,10)(2,8,11)(3,9,12)(17,19,24)(18,20,23);
M_M_U:=(4,6,5)(13,21,15)(14,22,16)(25,28,26);
M_M_BL:=(1,12,4)(2,10,5)(3,11,6)(13,18,22)(14,17,21);
M_M_R:=(7,9,8)(15,23,19)(16,24,20)(25,26,27);
M_M_BR:=(4,11,7)(5,12,8)(6,10,9)(15,22,24)(16,21,23);
M_M_L:=(1,3,2)(13,20,17)(14,19,18)(25,27,28);
Gen:=[
M_M_F,M_M_B,M_M_D,M_M_U,M_M_BL,M_M_R,M_M_BR,M_M_L
];
ip:=[[1],[4],[7],[10],[13],[15],[17],[19],[21],[23],[25],[26],[27],[28]];
# Size(Group(Gen));
# Size(Stabilizer(Group(Gen), ip, OnTuplesSets));

