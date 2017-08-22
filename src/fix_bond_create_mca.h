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

#ifdef FIX_CLASS

FixStyle(bond/create/mca,FixBondCreateMCA)

#else

#ifndef LMP_FIX_BOND_CREATE_MCA_H
#define LMP_FIX_BOND_CREATE_MCA_H

#include "fix.h"
#include "pair_mca.h"

namespace LAMMPS_NS {

class FixBondCreateMCA : public Fix {
  friend class PairMCA;
 public:
  FixBondCreateMCA(class LAMMPS *, int, char **);
  ~FixBondCreateMCA();
//  void post_create(); registernig of the fixes are moved to AtomVecMCA::init() because they should be registered once
  int setmask();
  void init();
  void init_list(int, class NeighList *);
  void setup(int);
//  void pre_force(int);- moved it to BondMCA::build_bond_index() caled from Neighbor::bond_all()
  void post_integrate();
  void post_integrate_respa(int, int);
  int modify_param(int,char**);
  //virtual
  int pack_comm(int, int *, double *, int, int *);
  //virtual
  void unpack_comm(int, int, double *);
  //virtual
  int pack_reverse_comm(int, int, double *);
  //virtual
  void unpack_reverse_comm(int, int *, double *);
  void grow_arrays(int);
  void copy_arrays(int, int);
  //virtual
  int pack_exchange(int, double *);
  //virtual
  int unpack_exchange(int, double *);
  double compute_vector(int);
  double memory_usage();

 private:
  bool already_bonded(int,int);

  int me;
  int init_state;        // initial state of the bond, 1 - unbonded; 0 - bonded (default)
  int iatomtype,jatomtype;
  int btype,seed;
  int imaxbond,jmaxbond;
  int inewtype,jnewtype;
  double cutsq,fraction;

  int createcount,createcounttotal;   // bond formation stats

  int nmax;
  int maxbondsperatom;   // max bonds per atom
  int *bondcount;        // count of created bonds this atom is part of
  int *npartner;         // # of preferred atoms for this atom to bond to
  int **partner;         // IDs of preferred atoms for this atom to bond to
  double *probability;   // random # to use in decision to form bond

  class RanMars *random;
  class NeighList *list;
  int countflag,commflag;
  int nlevels_respa;

};

}

#endif
#endif
