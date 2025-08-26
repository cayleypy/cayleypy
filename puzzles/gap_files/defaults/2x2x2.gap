# /home/username/.bun/bin/bun /home/username/Desktop/cayley/cubing.js/src/bin/puzzle-geometry-bin.ts --gap 2x2x2
# PuzzleGeometry 0.1 Copyright 2018 Tomas Rokicki.
# 
M_M_F:=(1,14,16,11)(2,15,17,12)(3,13,18,10);
M_M_B:=(4,9,19,24)(5,7,20,22)(6,8,21,23);
M_M_D:=(13,22,19,16)(14,23,20,17)(15,24,21,18);
M_M_U:=(1,10,7,4)(2,11,8,5)(3,12,9,6);
M_M_L:=(7,12,16,21)(8,10,17,19)(9,11,18,20);
M_M_R:=(1,6,22,15)(2,4,23,13)(3,5,24,14);
Gen:=[
M_M_F,M_M_B,M_M_D,M_M_U,M_M_L,M_M_R
];
ip:=[[1],[4],[7],[10],[13],[16],[19],[22]];
# Size(Group(Gen));
# Size(Stabilizer(Group(Gen), ip, OnTuplesSets));

