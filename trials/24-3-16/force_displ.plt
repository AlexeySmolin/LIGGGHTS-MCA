#
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
A=0.258064

set xlabel "dX/L, %"
set ylabel "F/A, MPa"
set title "Loading diagram"

plot \
     Y*x*1.0E-8 title "Young" with lines, \
     './line.dat' using (100.*($2-X0)/X0):(-$5*1.E-6/A) not with points

pause -1 "Hit return "

set xlabel "t, s"
set ylabel "x, m"
set title "X vs time"

plot \
     './line.dat' using ($1):($2) not with lines

pause -1 "Hit return "

#pause mouse button2 "Click... "

# undo what we have done above
set title
set autoscale x
set xtics
