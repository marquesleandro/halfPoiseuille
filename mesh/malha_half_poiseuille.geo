// Gmsh project created on Wed Dec 21 22:42:40 2016

lc = DefineNumber[ 0.02, Name "Parameters/lc" ]; 

// removing 1.0 from y coordinate for symmetry, since 
// normalV should be at Y=0.0 and everything else above. 
// This number makes the average channel diameter equal to 1.
xMax = 10.0000;
symY = 1.0;
Point(1)={0.000000,symY-0,0,lc}; 
Point(2)={1.725,symY-0,0,lc}; 
Point(3)={8.275,symY-0,0,lc}; 
Point(4)={xMax,symY-0,0,lc}; 
Point(5)={xMax,symY-1,0,lc}; 
Point(6)={0,symY-1,0,lc}; 

Line(1)={1,2}; 
Line(2)={2,3}; 
Line(3)={3,4}; 
Line(4)={4,5}; 
Line(5)={5,6}; 
Line(6)={6,1}; 

Line Loop(11) = {3, 4, 5, 6, 1, 2};
Plane Surface(11) = {11};

Physical Line("dirichlet1 noslip top") = {1,2,3};
Physical Line("neumann1 outflow") = {4};
Physical Line("dirichlet1 inflow") = {6};
Physical Line("neumann1 symmetric") = {5};
Physical Line("dirichlet2 noslip top") = {1,2,3};
Physical Line("neumann2 outflow") = {4};
Physical Line("dirichlet2 inflow") = {6};
Physical Line("dirichlet2 symmetric") = {5};
Physical Line("dirichlet3 symmetric") = {5};
Physical Line("neumann3 outflow") = {4};
Physical Line("dirichlet3 inflow") = {6};
Physical Line("dirichlet3 top") = {1,2,3};

Physical Surface(11) = {11};
