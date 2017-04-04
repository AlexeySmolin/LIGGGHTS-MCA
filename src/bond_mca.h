/* ----------------------------------------------------------------------
    This is the

    ██╗     ██╗ ██████╗  ██████╗  ██████╗ ██╗  ██╗████████╗███████╗
    ██║     ██║██╔════╝ ██╔════╝ ██╔════╝ ██║  ██║╚══██╔══╝██╔════╝
    ██║     ██║██║  ███╗██║  ███╗██║  ███╗███████║   ██║   ███████╗
    ██║     ██║██║   ██║██║   ██║██║   ██║██╔══██║   ██║   ╚════██║
    ███████╗██║╚██████╔╝╚██████╔╝╚██████╔╝██║  ██║   ██║   ███████║
    ╚══════╝╚═╝ ╚═════╝  ╚═════╝  ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝®

    DEM simulation engine, released by
    DCS Computing Gmbh, Linz, Austria
    http://www.dcs-computing.com, office@dcs-computing.com

    LIGGGHTS® is part of CFDEM®project:
    http://www.liggghts.com | http://www.cfdem.com

    Core developer and main author:
    Christoph Kloss, christoph.kloss@dcs-computing.com

    LIGGGHTS® is open-source, distributed under the terms of the GNU Public
    License, version 2 or later. It is distributed in the hope that it will
    be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
    of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. You should have
    received a copy of the GNU General Public License along with LIGGGHTS®.
    If not, see http://www.gnu.org/licenses . See also top-level README
    and LICENSE files.

    LIGGGHTS® and CFDEM® are registered trade marks of DCS Computing GmbH,
    the producer of the LIGGGHTS® software and the CFDEM®coupling software
    See http://www.cfdem.com/terms-trademark-policy for details.

-------------------------------------------------------------------------
    Contributing author and copyright for this file:

    Alexey Smolin (ISPMS SB RAS, Tomsk, Russia, http://www.ispms.ru)

    Copyright 2016-     ISPMS SB RAS, Tomsk, Russia
------------------------------------------------------------------------- */

#ifdef BOND_CLASS

BondStyle(mca,BondMCA)

#else

#ifndef LMP_BOND_MCA_H
#define LMP_BOND_MCA_H

#include "stdio.h"
#include "bond.h"

namespace LAMMPS_NS {

class BondMCA : public Bond {
 public:
  BondMCA(class LAMMPS *);
  ~BondMCA();
  void init_style();
  void compute(int, int);
  void build_bond_index();
  void coeff(int, char **);
  double equilibrium_distance(int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  //double single(int, double, int, int);
  double single(int, double, int, int, double &);

 protected:
  int *breakmode;
  int *bindmode;
  double *crackVelo; ///AS Crack propagation velocity
  double *breakVal1,*breakVal2; // Ultimate value for bond braking. It may be Equivalent Strain OR Equivalent Stress OR Tensile Strength and Compression Strength
  double *shapeDrPr,*volumeDrPr; // Coefficients for computing Drucker-Prager criterion
  double *bindPressure,*bindPlastHeat; // Ultimate value for binding new bond.
  void allocate();

  class FixPropertyAtom *fix_Temp; ///AS TODO We do not use it for now in MCA
  double *Temp;

};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Incorrect args for bond coefficients

Self-explanatory.  Check the input script or data file.

*/
