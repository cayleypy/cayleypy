# /home/username/.bun/bin/bun /home/username/Desktop/cayley/cubing.js/src/bin/puzzle-geometry-bin.ts --gap mastermorphix
# PuzzleGeometry 0.1 Copyright 2018 Tomas Rokicki.
# 
M_M_DF:=(7,12)(8,10)(9,11)(25,26)(29,30)(33,34)(37,38);
M_M_RL:=(1,4)(2,5)(3,6)(27,28)(31,32)(35,36)(39,40);
M_M_LF:=(1,9)(2,7)(3,8)(13,14)(21,23)(29,31)(33,35);
M_M_RD:=(4,11)(5,12)(6,10)(15,16)(22,24)(30,32)(34,36);
M_M_DL:=(4,9)(5,7)(6,8)(19,20)(22,23)(34,35)(38,39);
M_M_FR:=(1,11)(2,12)(3,10)(17,18)(21,24)(33,36)(37,40);
Gen:=[
M_M_DF,M_M_RL,M_M_LF,M_M_RD,M_M_DL,M_M_FR
];
ip:=[[1],[4],[7],[10],[13],[15],[17],[19],[21],[22],[23],[24],[25],[27],[29],[30],[31],[32],[33],[34],[35],[36],[37],[38],[39],[40]];
# Size(Group(Gen));
# Size(Stabilizer(Group(Gen), ip, OnTuplesSets));

