# /home/username/.bun/bin/bun /home/username/Desktop/cayley/cubing.js/src/bin/puzzle-geometry-bin.ts --gap 2x2x2 + little chop
# PuzzleGeometry 0.1 Copyright 2018 Tomas Rokicki.
# 
M_M_F:=(1,2,11,22)(3,13,6,16)(4,20,7,18)(25,26,35,46)(27,37,30,40)(28,44,31,42);
M_M_B:=(5,12,24,15)(8,17,10,19)(9,21,23,14)(29,36,48,39)(32,41,34,43)(33,45,47,38);
M_M_D:=(4,13,5,14)(6,11,9,19)(8,15,22,18)(28,37,29,38)(30,35,33,43)(32,39,46,42);
M_M_U:=(1,23,17,3)(2,20,10,12)(7,16,24,21)(25,47,41,27)(26,44,34,36)(31,40,48,45);
M_M_L:=(1,4,19,24)(6,15,23,20)(10,16,22,14)(25,28,43,48)(30,39,47,44)(34,40,46,38);
M_M_R:=(2,21,8,13)(3,12,9,18)(5,11,7,17)(26,45,32,37)(27,36,33,42)(29,35,31,41);
M_M_DF:=(1,5)(2,14)(3,15)(4,11)(6,18)(13,22)(25,29)(26,38)(28,35)(30,42)(33,44)(37,46);
M_M_UB:=(7,19)(8,16)(9,20)(10,21)(12,23)(17,24)(27,39)(31,43)(32,40)(34,45)(36,47)(41,48);
M_M_FL:=(1,6)(2,15)(4,16)(11,23)(13,24)(20,22)(25,30)(26,39)(28,40)(31,38)(35,47)(44,46);
M_M_BR:=(3,19)(5,21)(7,14)(8,12)(9,17)(10,18)(27,43)(29,45)(32,36)(33,41)(34,42)(37,48);
M_M_DB:=(4,17)(5,19)(8,14)(9,15)(10,13)(18,23)(28,41)(29,43)(30,36)(32,38)(33,39)(34,37);
M_M_UF:=(1,7)(2,16)(3,20)(6,12)(11,24)(21,22)(25,31)(26,40)(27,44)(35,48)(42,47)(45,46);
M_M_DL:=(1,8)(4,15)(5,20)(6,14)(13,23)(19,22)(28,39)(29,44)(30,38)(34,35)(37,47)(43,46);
M_M_UR:=(2,17)(3,21)(7,12)(9,16)(10,11)(18,24)(25,32)(26,41)(27,45)(31,36)(33,40)(42,48);
M_M_BL:=(4,21)(6,17)(8,20)(10,15)(14,24)(19,23)(29,40)(30,41)(32,44)(34,39)(38,48)(43,47);
M_M_FR:=(1,9)(2,18)(3,11)(5,16)(7,13)(12,22)(25,33)(26,42)(27,35)(28,45)(31,37)(36,46);
M_M_UL:=(1,10)(2,19)(6,21)(7,15)(16,23)(20,24)(25,34)(30,45)(31,39)(40,47)(41,46)(44,48);
M_M_DR:=(3,14)(4,12)(5,18)(8,11)(9,13)(17,22)(26,43)(27,38)(28,36)(29,42)(32,35)(33,37);
Gen:=[
M_M_F,M_M_B,M_M_D,M_M_U,M_M_L,M_M_R,M_M_DF,M_M_UB,M_M_FL,M_M_BR,M_M_DB,M_M_UF,M_M_DL,M_M_UR,M_M_BL,M_M_FR,M_M_UL,M_M_DR
];
ip:=[[1,2,11,22],[3,9,12,18],[4,5,13,14],[6,15,20,23],[7,16,21,24],[8,10,17,19],[25,26,35,46],[27,33,36,42],[28,29,37,38],[30,39,44,47],[31,40,45,48],[32,34,41,43]];
# Size(Group(Gen));
# Size(Stabilizer(Group(Gen), ip, OnTuplesSets));

