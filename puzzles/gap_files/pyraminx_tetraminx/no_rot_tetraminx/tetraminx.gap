# /home/username/.bun/bin/bun /home/username/Desktop/cayley/cubing.js/src/bin/puzzle-geometry-bin.ts --gap --moves 2F,2D,2BL,2BR tetraminx
# PuzzleGeometry 0.1 Copyright 2018 Tomas Rokicki.
# 
M_2BL:=(5,11,9)(6,12,10)(19,20,21);
M_2BR:=(1,7,6)(2,8,5)(22,23,24);
M_2F:=(3,12,8)(4,11,7)(16,17,18);
M_2D:=(1,9,3)(2,10,4)(13,14,15);
Gen:=[
M_2BL,M_2BR,M_2F,M_2D
];
ip:=[[1],[3],[5],[7],[9],[11],[13],[16],[19],[22]];
# Size(Group(Gen));
# Size(Stabilizer(Group(Gen), ip, OnTuplesSets));

