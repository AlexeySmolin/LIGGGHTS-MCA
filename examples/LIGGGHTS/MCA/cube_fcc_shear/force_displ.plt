#!/bin/gnuplot

# Requires data file "line.dat" from this directory,
# so change current working directory to this directory before running.
#
#set xrange[0.0:0.001005]
#set yrange[-0.0:0.5e8]
#set xtics 0.0, 0.001, 0.005
#set ytics 0.0, 1.0e8, 2.0e9
set grid
set autoscale

#X0=1.6164461
X0=3.5921
#H0=3.2328922
H0=7.1842
A=H0*H0

Y=6.895e10
p=0.3
G=Y/(2*(1+p))
K=Y/(3*(1-2.0*p))
#print "2G=",2*G


set xlabel "dZ/L, %"
set ylabel "F/A, MPa"
set title "Loading diagram"

plot \
     2.0*G*x*1.0E-8 title "2* shear modulus" with lines, \
     G*x*1.0E-8 title "shear modulus" with lines, \
     './cube.dat' using (100.*($3-X0)/H0):(-$6*1.E-6/A) not with points

pause -1 "Hit return "

set xlabel "t, s"
set ylabel "z, m"
set title "Z vs time"

plot \
     './cube.dat' using ($1):($4) not with lines

pause -1 "Hit return "

#pause mouse button2 "Click... "

# undo what we have done above
set title
set autoscale x
set xtics