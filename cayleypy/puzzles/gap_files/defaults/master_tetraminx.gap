# /home/username/.bun/bin/bun /home/username/Desktop/cayley/cubing.js/src/bin/puzzle-geometry-bin.ts --gap master tetraminx
# PuzzleGeometry 0.1 Copyright 2018 Tomas Rokicki.
# 
M_M_R:=(5,17,21)(6,18,22)(43,45,44);
M_M_BL:=(1,12,4)(2,11,3)(9,20,15)(10,19,16)(25,32,28)(26,31,27)(37,46,42)(38,47,40)(39,48,41);
M_M_L:=(7,12,20)(8,11,19)(46,48,47);
M_M_BR:=(3,21,14)(4,22,13)(9,23,18)(10,24,17)(27,35,34)(28,36,33)(37,41,43)(38,42,44)(39,40,45);
M_M_B:=(3,16,24)(4,15,23)(40,42,41);
M_M_F:=(1,18,7)(2,17,8)(5,19,14)(6,20,13)(25,34,29)(26,33,30)(37,45,47)(38,43,48)(39,44,46);
M_M_U:=(1,9,13)(2,10,14)(37,39,38);
M_M_D:=(5,24,11)(6,23,12)(7,22,15)(8,21,16)(29,36,31)(30,35,32)(40,46,43)(41,47,44)(42,48,45);
Gen:=[
M_M_R,M_M_BL,M_M_L,M_M_BR,M_M_B,M_M_F,M_M_U,M_M_D
];
ip:=[[1],[3],[5],[7],[9],[11],[13],[15],[17],[19],[21],[23],[25],[27],[29],[31],[33],[35],[37],[40],[43],[46]];
# Size(Group(Gen));
# Size(Stabilizer(Group(Gen), ip, OnTuplesSets));

