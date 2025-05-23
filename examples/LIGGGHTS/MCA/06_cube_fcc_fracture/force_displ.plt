#!/bin/gnuplot

# Requires data file "line.dat" from this directory,
# so change current working directory to this directory before running.
#
#set xrange[0.0:0.001005]
#set xtics 0.0, 0.001, 0.005
#set ytics 0.0, 1.0e8, 2.0e9
set grid
set autoscale
set yrange[-0.0:3.0]

Z0=7.36381
X0=7.54342
#7.1842
Y=6.895e10
#A=0.0912394*121
A=(X0+0.254*1.4142)*(X0+0.254*1.4142)

set xlabel "dZ/L, %"
set ylabel "F/A, MPa"
set title "Loading diagram"

plot \
     Y*x*1.0E-8 title "Young" with lines, \
     './cube-orig.dat' using (100.*($4-Z0)/Z0):(-$7*1.E-6/A) t "orig" with l, \
     './cube.dat' using (100.*($4-Z0)/Z0):(-$7*1.E-6/A) not with linespoints

pause -1 "Hit return "

set autoscale
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
