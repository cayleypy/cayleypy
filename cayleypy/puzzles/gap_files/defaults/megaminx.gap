# /home/username/.bun/bin/bun /home/username/Desktop/cayley/cubing.js/src/bin/puzzle-geometry-bin.ts --gap megaminx
# PuzzleGeometry 0.1 Copyright 2018 Tomas Rokicki.
# 
M_M_U:=(1,4,19,22,7)(2,5,20,23,8)(3,6,21,24,9)(61,63,75,73,77)(62,64,76,74,78);
M_M_D:=(37,55,58,52,40)(38,56,59,53,41)(39,57,60,54,42)(93,115,117,111,95)(94,116,118,112,96);
M_M_F:=(1,9,31,35,12)(2,7,32,36,10)(3,8,33,34,11)(61,66,87,68,80)(62,65,88,67,79);
M_M_B:=(25,41,52,47,30)(26,42,53,48,28)(27,40,54,46,29)(97,112,113,108,99)(98,111,114,107,100);
M_M_L:=(1,11,15,16,5)(2,12,13,17,6)(3,10,14,18,4)(63,79,69,81,71)(64,80,70,82,72);
M_M_DR:=(43,47,54,59,51)(44,48,52,60,49)(45,46,53,58,50)(101,103,113,117,119)(102,104,114,118,120);
M_M_BL:=(4,17,27,28,20)(5,18,25,29,21)(6,16,26,30,19)(71,84,99,86,76)(72,83,100,85,75);
M_M_FR:=(31,50,60,56,36)(32,51,58,57,34)(33,49,59,55,35)(87,109,120,115,90)(88,110,119,116,89);
M_M_BR:=(19,29,48,43,23)(20,30,46,44,24)(21,28,47,45,22)(73,86,107,103,106)(74,85,108,104,105);
M_M_FL:=(10,35,57,38,15)(11,36,55,39,13)(12,34,56,37,14)(67,90,94,91,70)(68,89,93,92,69);
M_M_R:=(7,24,43,50,33)(8,22,44,51,31)(9,23,45,49,32)(65,77,105,101,109)(66,78,106,102,110);
M_M_DL:=(13,38,40,26,18)(14,39,41,27,16)(15,37,42,25,17)(81,91,95,97,83)(82,92,96,98,84);
Gen:=[
M_M_U,M_M_D,M_M_F,M_M_B,M_M_L,M_M_DR,M_M_BL,M_M_FR,M_M_BR,M_M_FL,M_M_R,M_M_DL
];
ip:=[[1],[4],[7],[10],[13],[16],[19],[22],[25],[28],[31],[34],[37],[40],[43],[46],[49],[52],[55],[58],[61],[63],[65],[67],[69],[71],[73],[75],[77],[79],[81],[83],[85],[87],[89],[91],[93],[95],[97],[99],[101],[103],[105],[107],[109],[111],[113],[115],[117],[119]];
# Size(Group(Gen));
# Size(Stabilizer(Group(Gen), ip, OnTuplesSets));

