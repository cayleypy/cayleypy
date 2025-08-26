# /home/username/.bun/bin/bun /home/username/Desktop/cayley/cubing.js/src/bin/puzzle-geometry-bin.ts --gap --moves F,2F,D,2D,BL,2BL,BR,2BR tetraminx
# PuzzleGeometry 0.1 Copyright 2018 Tomas Rokicki.
# 
M_BL:=(1,8,4)(2,7,3)(13,22,18)(14,23,16)(15,24,17);
M_2BL:=(5,11,9)(6,12,10)(19,20,21);
M_BR:=(3,11,10)(4,12,9)(13,17,19)(14,18,20)(15,16,21);
M_2BR:=(1,7,6)(2,8,5)(22,23,24);
M_F:=(1,10,5)(2,9,6)(13,21,23)(14,19,24)(15,20,22);
M_2F:=(3,12,8)(4,11,7)(16,17,18);
M_D:=(5,12,7)(6,11,8)(16,22,19)(17,23,20)(18,24,21);
M_2D:=(1,9,3)(2,10,4)(13,14,15);
Gen:=[
M_BL,M_2BL,M_BR,M_2BR,M_F,M_2F,M_D,M_2D
];
ip:=[[1],[3],[5],[7],[9],[11],[13],[16],[19],[22]];
# Size(Group(Gen));
# Size(Stabilizer(Group(Gen), ip, OnTuplesSets));

