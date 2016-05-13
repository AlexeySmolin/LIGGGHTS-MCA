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

#ifdef ATOM_CLASS

AtomStyle(mca,AtomVecMCA)

#else

#ifndef LMP_ATOM_VEC_MCA_H
#define LMP_ATOM_VEC_MCA_H

#include "atom_vec.h"
#include "fix_bond_exchange_mca.h"

namespace LAMMPS_NS {

class AtomVecMCA : public AtomVec {
 public:
  AtomVecMCA(class LAMMPS *);
  void settings(int narg, char **arg);
  ~AtomVecMCA(); //{}; fbe - delete?
  void init();
  void grow(int);
  void grow_reset();
  void copy(int, int, int);
  int pack_comm(int, int *, double *, int, int *);
  int pack_comm_vel(int, int *, double *, int, int *);
  int pack_comm_vel_wedge(int, int *, double *, int, int *);
  int pack_comm_hybrid(int, int *, double *);
  void unpack_comm(int, int, double *);
  void unpack_comm_vel(int, int, double *);
  int unpack_comm_hybrid(int, int, double *);
  int pack_reverse(int, int, double *);
  int pack_reverse_hybrid(int, int, double *);
  void unpack_reverse(int, int *, double *);
  int unpack_reverse_hybrid(int, int *, double *);
  int pack_border(int, int *, double *, int, int *);
  int pack_border_vel(int, int *, double *, int, int *);
  int pack_border_vel_wedge(int, int *, double *, int, int *);
  int pack_border_hybrid(int, int *, double *);
  void unpack_border(int, int, double *);
  void unpack_border_vel(int, int, double *);
  int unpack_border_hybrid(int, int, double *);
  int pack_exchange(int, double *);
  int unpack_exchange(double *);
  int size_restart();
  int pack_restart(int, double *);
  int unpack_restart(double *);
  void create_atom(int, double *);
  void data_atom(double *, tagint, char **);
  int data_atom_hybrid(int, char **);
  void data_vel(int, char **);
  int data_vel_hybrid(int, char **);
  void pack_data(double **);
  void pack_data(double **buf,int tag_offset); 
  int pack_data_hybrid(int, double *);
  void write_data(FILE *, int, double **);
  int write_data_hybrid(FILE *, double *);
  void pack_vel(double **);
  void pack_vel(double **buf,int tag_offset);
  int pack_vel_hybrid(int, double *);
  void write_vel(FILE *, int, double **);
  int write_vel_hybrid(FILE *, double *);
  bigint memory_usage();
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);

 private:
   //!! All these are in Sphere
  int *tag,*type,*mask; //atom ID, atom type?, deform_groupbit?
  tagint *image;        // IMGMAX-IMG2BITS in new version of LIGGGHTS?? 
  double **x,**v,**f;
//!!  double *radius;
  double *density,*rmass; //!!Later we should use  'int mass_type;  // 1 if per-type masses'
  double **omega,**torque;
//!!  int radvary; //!! We do not need it

  int packing;   //!! Packing of movable callular automata: 'sc' or 'fcc' or 'hcp'
  int coord_num; //!! Coordination number is defined by packing (6 for cubic or 12 for fcc and hcp)
  double mca_radius; //!! Change from array to single variable: all automata have the same radius
  double contact_area; //!! Initial contact area defined by packing. Remember about heat transfer through contact_area in granular!!

  double *q; //!! mca_inertia moment of inertia is a scalar as for sphere
  double **mu;    //!! We need orientation vector to describe rotation as a first approximation
                  //!! Later we must use Rodrigues rotation vector or quaternions, as in AtomVecEllipsoid 'struct Bonus {double quat[4]...};'
  double *p;  //!! 'pressure == -mean_stress' - is used for many-body interaction
  double *s0; //!! equiv_stress~ equivalent (or von Mises, shear) stress - is used for plasticity
  double *e; //!! equiv_strain~ equivalent (shear) strain - is used for plasticity
/*
For the MCA style, the number of mca bonds per atom is stored, and the information associated to it:
 the type of each bond, the ID of the bonded atom and the so-called bond history.
The bond history is similar to the contact history for granulars, it stores the internal state of the bond.
What exactly is stored in this internal state is defined by the MCA style used.

In atom_style command it need 4 args: 
the automaton radius, the packing of automata,
the number of bond types, and the _maximum_ number of bonds that each atom can have.

An example for the sytnax is given below:
atom_style mca radius 0.0001 packing fcc n_bondtypes 1 bonds_per_atom 6  

Ususally atom_vec_mca has number bonds defined by packing (6 for cubic packing, 12 for fcc packing), 
but during deformation other atoms can be in contact with it and the total number of interacting 
neighbours may be greater than coordination number of the packing .

For each bond type, the parameters have to be specified via the bond_coeff command.
These parameters define bond formation and breaking rules
bond_coeff 	1 0.0025 10000000000 10000000000 ${simplebreak} 0.002501

*/
  int *molecule; //!! This allows to have bonds. Do we really need it?
  int **nspecial,**special;
  int *num_bond; //!! Do we really need it?
  int **bond_type,**bond_atom;
  int num_bondhist;
  double ***bond_hist; //???

  class FixBondExchangeMCA *fbe; //!! This is used for MPI eschange as I understand. But there is no '#include ...' Why?

  double get_init_volume(); //!! compute initial volume of cellular automaton based on radius and packing
  double get_contact_area(); //!! compute initial contact_area between cellular automata based on radius and packing

};

}

#endif
#endif
