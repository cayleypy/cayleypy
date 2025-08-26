# /home/username/.bun/bin/bun /home/username/Desktop/cayley/cubing.js/src/bin/puzzle-geometry-bin.ts --gap pyramorphix
# PuzzleGeometry 0.1 Copyright 2018 Tomas Rokicki.
# 
M_M_DF:=(7,12)(8,10)(9,11)(13,14);
M_M_RL:=(1,4)(2,5)(3,6)(15,16);
M_M_LF:=(1,9)(2,7)(3,8)(13,15);
M_M_RD:=(4,11)(5,12)(6,10)(14,16);
M_M_DL:=(4,9)(5,7)(6,8)(14,15);
M_M_FR:=(1,11)(2,12)(3,10)(13,16);
Gen:=[
M_M_DF,M_M_RL,M_M_LF,M_M_RD,M_M_DL,M_M_FR
];
ip:=[[1],[4],[7],[10],[13],[14],[15],[16]];
# Size(Group(Gen));
# Size(Stabilizer(Group(Gen), ip, OnTuplesSets));

