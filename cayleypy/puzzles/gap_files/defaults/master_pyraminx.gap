# /home/username/.bun/bin/bun /home/username/Desktop/cayley/cubing.js/src/bin/puzzle-geometry-bin.ts --gap master pyraminx
# PuzzleGeometry 0.1 Copyright 2018 Tomas Rokicki.
# 
M_M_R:=(7,9,8);
M_M_2R:=(17,29,33)(18,30,34)(55,57,56);
M_M_2BL:=(19,35,25)(20,36,26)(41,47,45)(42,48,46)(61,63,62);
M_M_BL:=(1,12,4)(2,10,5)(3,11,6)(13,24,16)(14,23,15)(21,32,27)(22,31,28)(37,44,40)(38,43,39)(49,58,54)(50,59,52)(51,60,53);
M_M_L:=(1,3,2);
M_M_2L:=(19,24,32)(20,23,31)(58,60,59);
M_M_2BR:=(13,27,18)(14,28,17)(37,43,42)(38,44,41)(61,64,63);
M_M_BR:=(4,11,7)(5,12,8)(6,10,9)(15,33,26)(16,34,25)(21,35,30)(22,36,29)(39,47,46)(40,48,45)(49,53,55)(50,54,56)(51,52,57);
M_M_B:=(10,12,11);
M_M_2B:=(15,28,36)(16,27,35)(52,54,53);
M_M_2F:=(21,34,24)(22,33,23)(39,48,44)(40,47,43)(62,63,64);
M_M_F:=(1,6,8)(2,4,9)(3,5,7)(13,30,19)(14,29,20)(17,31,26)(18,32,25)(37,46,41)(38,45,42)(49,57,59)(50,55,60)(51,56,58);
M_M_U:=(4,6,5);
M_M_2U:=(13,21,25)(14,22,26)(49,51,50);
M_M_2D:=(15,31,29)(16,32,30)(37,45,39)(38,46,40)(61,62,64);
M_M_D:=(1,7,10)(2,8,11)(3,9,12)(17,36,23)(18,35,24)(19,34,27)(20,33,28)(41,48,43)(42,47,44)(52,58,55)(53,59,56)(54,60,57);
Gen:=[
M_M_R,M_M_2R,M_M_2BL,M_M_BL,M_M_L,M_M_2L,M_M_2BR,M_M_BR,M_M_B,M_M_2B,M_M_2F,M_M_F,M_M_U,M_M_2U,M_M_2D,M_M_D
];
ip:=[[1],[4],[7],[10],[13],[15],[17],[19],[21],[23],[25],[27],[29],[31],[33],[35],[37],[39],[41],[43],[45],[47],[49],[52],[55],[58],[61],[62],[63],[64]];
# Size(Group(Gen));
# Size(Stabilizer(Group(Gen), ip, OnTuplesSets));

