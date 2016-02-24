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
    Nadia Salman (iT-CDT, Leeds, UK)

    Copyright 2016-     ISPMS SB RAS, Tomsk, Russia
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
FixStyle(wall/mca,FixWallMCA)
#else

#ifndef LMP_FIX_WALL_MCA_H
#define LMP_FIX_WALL_MCA_H

#include "fix.h"

namespace LAMMPS_NS {

class FixWallMCA : public Fix {
 public:
  FixWallMCA(class LAMMPS *, int, char **);
  ~FixWallMCA();

  /* INHERITED FROM Fix */

  virtual int setmask();
  virtual void init();
  virtual void setup(int vflag);
//  virtual void post_create();
//  virtual void pre_delete(bool unfixflag);
  virtual void post_force(int vflag);
  virtual void post_force_respa(int, int, int); ///??

  /* PUBLIC ACCESS FUNCTIONS */

 protected:

  int nlevels_respa_;
  int wallstyle;
  double lo,hi,cylradius;
  double r0,D;

  // class variables for atom properties
  int nlocal_;
  double **x_, **f_, *radius_, *rmass_, **wallforce_, r0_;

};

}

#endif
#endif
