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

X0=5.08
Y=6.895e10
A=(X0+0.254*2)*(X0+0.254*2)
#A=0.258064*121

set xlabel "dZ/L, %"
set ylabel "F/A, MPa"
set title "Loading diagram"

plot \
     Y*x*1.0E-8 title "Young" with lines, \
     './cube-orig.dat' using (100.*($4-X0)/X0):(-$7*1.E-6/A) t "original" with lines ,\
     './cube.dat' using (100.*($4-X0)/X0):(-$7*1.E-6/A) not with points pt 6

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
