# /home/username/.bun/bin/bun /home/username/Desktop/cayley/cubing.js/src/bin/puzzle-geometry-bin.ts --gap Christopher's jewel
# PuzzleGeometry 0.1 Copyright 2018 Tomas Rokicki.
# 
M_M_DBRRF:=(5,8,11,18)(6,7,12,17)(33,36,35,34);
M_M_UBBBLL:=(13,24,21,20)(14,23,22,19)(41,44,43,42);
M_M_DFLBL:=(1,6,15,24)(2,5,16,23)(45,48,47,46);
M_M_URBRBB:=(3,20,9,12)(4,19,10,11)(29,32,31,30);
M_M_DBLBBBR:=(9,22,15,18)(10,21,16,17)(37,40,39,38);
M_M_ULFR:=(1,14,3,8)(2,13,4,7)(25,28,27,26);
Gen:=[
M_M_DBRRF,M_M_UBBBLL,M_M_DFLBL,M_M_URBRBB,M_M_DBLBBBR,M_M_ULFR
];
ip:=[[1],[3],[5],[7],[9],[11],[13],[15],[17],[19],[21],[23],[25],[29],[33],[37],[41],[45]];
# Size(Group(Gen));
# Size(Stabilizer(Group(Gen), ip, OnTuplesSets));

