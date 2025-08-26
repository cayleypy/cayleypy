# /home/username/.bun/bin/bun /home/username/Desktop/cayley/cubing.js/src/bin/puzzle-geometry-bin.ts --gap pyraminx
# PuzzleGeometry 0.1 Copyright 2018 Tomas Rokicki.
# 
M_M_r:=(31,33,32);
M_M_BL:=(1,8,4)(2,7,3)(13,22,18)(14,23,16)(15,24,17)(25,34,30)(26,35,28)(27,36,29);
M_M_l:=(34,36,35);
M_M_BR:=(3,11,10)(4,12,9)(13,17,19)(14,18,20)(15,16,21)(25,29,31)(26,30,32)(27,28,33);
M_M_b:=(28,30,29);
M_M_F:=(1,10,5)(2,9,6)(13,21,23)(14,19,24)(15,20,22)(25,33,35)(26,31,36)(27,32,34);
M_M_u:=(25,27,26);
M_M_D:=(5,12,7)(6,11,8)(16,22,19)(17,23,20)(18,24,21)(28,34,31)(29,35,32)(30,36,33);
Gen:=[
M_M_r,M_M_BL,M_M_l,M_M_BR,M_M_b,M_M_F,M_M_u,M_M_D
];
ip:=[[1],[3],[5],[7],[9],[11],[13],[16],[19],[22],[25],[28],[31],[34]];
# Size(Group(Gen));
# Size(Stabilizer(Group(Gen), ip, OnTuplesSets));

