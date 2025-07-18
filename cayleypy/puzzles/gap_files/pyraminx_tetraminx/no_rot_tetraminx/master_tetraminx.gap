# /home/username/.bun/bin/bun /home/username/Desktop/cayley/cubing.js/src/bin/puzzle-geometry-bin.ts --gap --moves 2F,3F,2D,3D,2BL,3BL,2BR,3BR master tetraminx
# PuzzleGeometry 0.1 Copyright 2018 Tomas Rokicki.
# 
M_2BL:=(7,23,13)(8,24,14)(29,35,33)(30,36,34)(49,51,50);
M_3BL:=(5,21,17)(6,22,18)(43,44,45);
M_2BR:=(1,15,6)(2,16,5)(25,31,30)(26,32,29)(49,52,51);
M_3BR:=(7,20,12)(8,19,11)(46,47,48);
M_2F:=(9,22,12)(10,21,11)(27,36,32)(28,35,31)(50,51,52);
M_3F:=(3,24,16)(4,23,15)(40,41,42);
M_2D:=(3,19,17)(4,20,18)(25,33,27)(26,34,28)(49,50,52);
M_3D:=(1,13,9)(2,14,10)(37,38,39);
Gen:=[
M_2BL,M_3BL,M_2BR,M_3BR,M_2F,M_3F,M_2D,M_3D
];
ip:=[[1],[3],[5],[7],[9],[11],[13],[15],[17],[19],[21],[23],[25],[27],[29],[31],[33],[35],[37],[40],[43],[46],[49],[50],[51],[52]];
# Size(Group(Gen));
# Size(Stabilizer(Group(Gen), ip, OnTuplesSets));

