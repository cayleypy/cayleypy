# /home/username/.bun/bin/bun /home/username/Desktop/cayley/cubing.js/src/bin/puzzle-geometry-bin.ts --gap tetraminx
# PuzzleGeometry 0.1 Copyright 2018 Tomas Rokicki.
# 
M_M_R:=(5,9,11)(6,10,12)(19,21,20);
M_M_BL:=(1,8,4)(2,7,3)(13,22,18)(14,23,16)(15,24,17);
M_M_L:=(1,6,7)(2,5,8)(22,24,23);
M_M_BR:=(3,11,10)(4,12,9)(13,17,19)(14,18,20)(15,16,21);
M_M_B:=(3,8,12)(4,7,11)(16,18,17);
M_M_F:=(1,10,5)(2,9,6)(13,21,23)(14,19,24)(15,20,22);
M_M_U:=(1,3,9)(2,4,10)(13,15,14);
M_M_D:=(5,12,7)(6,11,8)(16,22,19)(17,23,20)(18,24,21);
Gen:=[
M_M_R,M_M_BL,M_M_L,M_M_BR,M_M_B,M_M_F,M_M_U,M_M_D
];
ip:=[[1],[3],[5],[7],[9],[11],[13],[16],[19],[22]];
# Size(Group(Gen));
# Size(Stabilizer(Group(Gen), ip, OnTuplesSets));

