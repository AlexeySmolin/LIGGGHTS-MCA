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

set xlabel "t, s"
set ylabel "x, m"
set title "X vs time"

plot \
     './line_if0.dat' using ($1):($2) t "0" with lines, \
     './line_if0.1.dat' using ($1):($2) t "0.1" with lines, \
     './line_if0.25.dat' using ($1):($2) t "0.25" with lines, \
     './line.dat' using ($1):($2) t "0.5" with lines, \
     './line_if0.75.dat' using ($1):($2) t "0.75" with lines, \
     './line_if1.dat' using ($1):($2) t "1" with lines

pause -1 "Hit return "

set xlabel "dX/L, %"
set ylabel "F, kN"
set title "Loading diagram"

plot './line.dat' using (($2-X0)/X0):($5*1.E-3) title "" with lines

pause -1 "Hit return "

#pause mouse button2 "Click... "

# undo what we have done above
set title
set autoscale x
set xtics
