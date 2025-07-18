# /home/username/.bun/bin/bun /home/username/Desktop/cayley/cubing.js/src/bin/puzzle-geometry-bin.ts --gap --moves 2F,3F,2D,3D,2BL,3BL,2BR,3BR pyraminx
# PuzzleGeometry 0.1 Copyright 2018 Tomas Rokicki.
# 
M_2BL:=(5,11,9)(6,12,10)(19,20,21);
M_3BL:=(31,32,33);
M_2BR:=(1,7,6)(2,8,5)(22,23,24);
M_3BR:=(34,35,36);
M_2F:=(3,12,8)(4,11,7)(16,17,18);
M_3F:=(28,29,30);
M_2D:=(1,9,3)(2,10,4)(13,14,15);
M_3D:=(25,26,27);
Gen:=[
M_2BL,M_3BL,M_2BR,M_3BR,M_2F,M_3F,M_2D,M_3D
];
ip:=[[1],[3],[5],[7],[9],[11],[13],[16],[19],[22],[25],[28],[31],[34]];
# Size(Group(Gen));
# Size(Stabilizer(Group(Gen), ip, OnTuplesSets));

