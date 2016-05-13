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

#include "math.h"
#include "stdlib.h"
#include "bond_mca.h"
#include "atom.h"
#include "neighbor.h"
#include "domain.h"
#include "comm.h"
#include "force.h"
#include "pair_mca.h"
#include "memory.h"
#include "modify.h"
#include "fix_property_atom.h"
#include "error.h"
#include "update.h"
#include "vector_liggghts.h"

using namespace LAMMPS_NS;

/*AS
 TODO list for mca bonds:
+ check whether run this bond style w/ or w/o mca pair style active,
  (neigh_modify command)
+ parallel debugging and testing not yet done
+ need evtally implemetation
*/

enum{
     BREAKSTYLE_SIMPLE,
     BREAKSTYLE_STRESS,
     BREAKSTYLE_STRESS_TEMP
    };

enum{
  R,      // 0 distance to neighbor,
  R_PREV, // 1 distance to neighbor at previous time step,
  A,      // 2 contact area
  E,      // 3 normal strain of i
///can be computed !!  QI, 4 distance to contact point of i 
  P,      // 4 normal force of i
  P_PREV, // 5 normal force of i at previous time step,
  NX,     // 6 unit vector from i to j
  NY,     // 7 unit vector from i to j
  NZ,     // 8 unit vector from i to j
  NX_PREV,// 9 unit vector from i to j at previous time step,
  NY_PREV,// 10 unit vector from i to j at previous time step,
  NZ_PREV,// 11 unit vector from i to j at previous time step,
  YX,     // 12 history of shear force of i
  YY,     // 13 history of shear force of i
  YZ,     // 14 history of shear force of i
  YX_PREV,// 15 history of shear force of i at previous time step,
  YY_PREV,// 16 history of shear force of i at previous time step,
  YZ_PREV,// 17 history of shear force of i at previous time step,
  SHX,    // 18 shear strain of i
  SHY,    // 19 shear strain of i
  SHZ,    // 20 shear strain of i
  SHX_PREV,// 21 shear strain of i at previous time step,
  SHY_PREV,// 22 shear strain of i at previous time step,
  SHZ_PREV,// 23 shear strain of i at previous time step,
  MX,     // 24 bending-torsion torque of i
  MY,     // 25 bending-torsion torque of i
  MZ,     // 26 bending-torsion torque of i
  SX,     // 27 shear force of i
  SY,     // 28 shear force of i
  SZ,     // 29 shear force of i
  };      // 30 in total
/* in case of newton is 'on' we need also these
  EJ,     // 4 normal strain of j
  PJ,     // 7 normal force of j
  PJ_PREV,// 8 normal force of j at previous time step,
  YJX,    // 21 history of shear force of j
  YJY,    // 22 history of shear force of j
  YjZ,    // 23 history of shear force of j
  YJX_PREV,// 24 history of shear force of j at previous time step,
  YJY_PREV,// 25 history of shear force of j at previous time step,
  YJZ_PREV,// 26 history of shear force of J at previous time step,
  SHJX,   // 33 shear strain of j
  SHJY,   // 34 shear strain of j
  SHJZ,   // 35 shear strain of J
  SHJX_PREV,// 36 shear strain of j at previous time step,
  SHJY_PREV,// 37 shear strain of j at previous time step,
  SHJZ_PREV,// 38 shear strain of J at previous time step,
  MJX,    // 42 bending-torsion torque of j
  MJY,    // 43 bending-torsion torque of j
  MJZ,    // 44 bending-torsion torque of j
  SJX,    // 48 shear force of j
  SJY,    // 49 shear force of j
  SJZ     // 50 shear force of j
*/

/* ---------------------------------------------------------------------- */

BondMCA::BondMCA(LAMMPS *lmp) : Bond(lmp)
{
    /* number of entries in bondhistlist. bondhistlist[number of bond][number of value (from 0 to number given here)]
     * so with this number you can modify how many pieces of information you savae with every bond
     * following dependencies and processes for saving,copying,growing the bondhistlist:
     * For mca we need 51 history values
     */
    n_granhistory(51);
     
    /* allocation of memory are here:
     * neighbor.cpp:       memory->create(bondhistlist,maxbond,atom->n_bondhist,"neigh:bondhistlist");
     * neigh_bond.cpp:     memory->grow(bondhistlist,maxbond,atom->n_bondhist,"neighbor:bondhistlist");
     * bond.cpp: void Bond::n_granhistory(int nhist) {ngranhistory = nhist;     atom->n_bondhist = ngranhistory; if(){FLERR}}
     * atom_vec_bond_gran.cpp:  memory->grow(atom->bond_hist,nmax,atom->bond_per_atom,atom->n_bondhist,"atom:bond_hist");
     */
    if(!atom->style_match("mca"))
      error->all(FLERR,"A granular bond style can only be used together with atom style mca");
    if(comm->me == 0)
        error->warning(FLERR,"BondMCA: This is a beta version - be careful!");
    fix_Temp = NULL;
}

/* ---------------------------------------------------------------------- */

BondMCA::~BondMCA()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(rb);
    memory->destroy(Sn);
    memory->destroy(St);
    memory->destroy(r_break);
    memory->destroy(sigman_break);
    memory->destroy(tau_break);
    memory->destroy(T_break);
  }
}

/* ---------------------------------------------------------------------- */

void  BondMCA::init_style()
{
    if(breakmode == BREAKSTYLE_STRESS_TEMP)
       fix_Temp = static_cast<FixPropertyAtom*>(modify->find_fix_property("Temp","property/atom","scalar",1,0,"mca bond"));
///                                                                                                           "bond mca")); ????
}

/* ---------------------------------------------------------------------- */

void BondMCA::compute_total_force(int eflag, int vflag)
{

  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = 0;

  double mca_radius  = atom->mca_radius;
  double contact_area  = atom->contact_area;
  int *tag = atom->tag; // tag of atom is their ID number
  double **x = atom->x;
  double **f = atom->f;
  double **torque = atom->torque;
  double *mean_stress  = atom->mean_stress;
  int **bondlist = neighbor->bondlist;
///AS I do not whant to use it, because it requires to be copied every step///  double **bondhistlist = neighbor->bondhistlist;
  double ***bond_hist = atom->bond_hist;
  int *num_bond = atom->num_bond;
  int **bond_atom = atom->bond_atom;

  int nbondlist = neighbor->nbondlist;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;
  PairMCA *mca_pair = (PairMCA*) force->pair;

// fprintf(logfile, "BondMCA::compute_total_force \n"); ///AS DEBUG

  for (int n = 0; n < nbondlist; n++) {
	//1st check if bond is broken,
    if(bondlist[n][3])
    {
	//fprintf(screen,"bond %d allready broken\n",n);
        continue;
    }

    int i1,i2,n1,n2;
    double rsq,r,rinv;
    double vt1,vt2,vt3;
    double tor1,tor2,tor3;
    double delx,dely,delz;
    double dnforce[3],dtforce[3],nv[3];
    double dttorque[3];
    double A;
    double q1,q2;// distance to contact point
    
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
// fprintf(logfile, "BondMCA::compute_total_force n=%d i1=%d i2=%d (tags= %d %d)\n", n, i1, i2, tag[i1], tag[i2]); ///AS DEBUG

    for (n1 = 0; n1 < num_bond[i1]; n1++) {
      if (bond_atom[i1][n1]==tag[i2]) break;
// fprintf(logfile, "BondMCA::compute_total_force bond_atom[i1][%d]=%d \n", n1, bond_atom[i1][n1]); ///AS DEBUG
    }
    if (n1 == num_bond[i1]) error->all(FLERR,"Internal error in BondMCA: n1 not found");

    for (n2 = 0; n2 < num_bond[i2]; n2++) {
      if (bond_atom[i2][n2]==tag[i1]) break;
// fprintf(logfile, "BondMCA::compute_total_force bond_atom[i2][%d]=%d \n", n2, bond_atom[i2][n2]); ///AS DEBUG
    }
    if (n2 == num_bond[i2]) error->all(FLERR,"Internal error in BondMCA: n2 not found");

    double *bond_hist1 = bond_hist[i1][n1];
    double *bond_hist2 = bond_hist[i2][n2];

    double pi = bond_hist1[P];
    double pj = bond_hist2[P];
// fprintf(logfile, "BondMCA::compute_total_force pi=%f pj=%f\n", pi, pj); ///AS DEBUG
    if ( bondlist[n][3] && ((pi>0.)||(pj>0.)) ) {
      error->warning(FLERR,"BondMCA::compute_total_force (pi>0.)||(pj>0.) - be careful!");
      continue;
    }

    q1 = mca_radius*(1. + bond_hist1[E]);
    q2 = mca_radius*(1. + bond_hist2[E]);

    //int type = bondlist[n][2];

    double rD0 = 2.0*mca_radius;
    double rDij = bond_hist1[R];
    int itype = atom->type[i1];
    double rKi =  mca_pair->K[itype][itype];
    int jtype = atom->type[i2];
    double rKj =  mca_pair->K[jtype][jtype];
    double rDStress = 0.5*(mean_stress[i1]/rKi + mean_stress[i2]/rKj);
    A = contact_area * (1. + rDStress) * rD0 / rDij; // Новая площадь контакта по пирамидкам;
    //bond_hist1[A] = A;
    //bond_hist2[A] = A;

    delx = x[i1][0] - x[i2][0];
    dely = x[i1][1] - x[i2][1];
    delz = x[i1][2] - x[i2][2];
    //domain->minimum_image(delx,dely,delz); ??

    rsq = delx*delx + dely*dely + delz*delz;
    r = sqrt(rsq);
    rinv = -1. / r; // "-" means that unit vector is from i1 to i2

    // normal unit vector
    nv[0] = delx*rinv;
    nv[1] = dely*rinv;
    nv[2] = delz*rinv;
    //bond_hist1[NX] = nv[0];
    //bond_hist1[NY] = nv[1];
    //bond_hist1[NZ] = nv[2];
    //bond_hist2[NX] = -nv[0];
    //bond_hist2[NY] = -nv[1];
    //bond_hist2[NZ] = -nv[2];

    // change in normal forces
    pi += pj; pi *= 0.5;
    vectorScalarMult3D(nv, pi, dnforce);
    //dnforce[0] = pi * nv[0];
    //dnforce[1] = pi * nv[1];
    //dnforce[2] = pi * nv[2];

    // tangential force
    vt1 = (bond_hist1[SX] - bond_hist2[SX])*0.5;
    vt2 = (bond_hist1[SY] - bond_hist2[SY])*0.5;
    vt3 = (bond_hist1[SZ] - bond_hist2[SZ])*0.5;

    // change in shear forces
    dtforce[0] = vt1;
    dtforce[1] = vt2;
    dtforce[2] = vt3;

    // torque due to tangential force

    vectorCross3D(nv,dtforce,dttorque);

    // torque due to torsion and bending

    tor1 = 0.5 * A * (bond_hist1[MX] - bond_hist2[MX]);
    tor2 = 0.5 * A * (bond_hist1[MY] - bond_hist2[MY]);
    tor3 = 0.5 * A * (bond_hist2[MZ] - bond_hist2[MZ]);

    // energy
    //if (eflag) error->all(FLERR,"MCA bonds currently do not support energy calculation");

    // apply force to each of 2 atoms

    if (newton_bond || i1 < nlocal) {
      f[i1][0] += (dnforce[0] + dtforce[0]) * A;
      f[i1][1] += (dnforce[1] + dtforce[1]) * A;
      f[i1][2] += (dnforce[2] + dtforce[2]) * A;
      torque[i1][0] += q1*dttorque[0] - tor1;
      torque[i1][1] += q1*dttorque[1] - tor2;
      torque[i1][2] += q1*dttorque[2] - tor3;
    }

    if (newton_bond || i2 < nlocal) {
      f[i2][0] -= (dnforce[0] + dtforce[0]) * A;
      f[i2][1] -= (dnforce[1] + dtforce[1]) * A;
      f[i2][2] -= (dnforce[2] + dtforce[2]) * A;
      torque[i2][0] -= q2*dttorque[0] + tor1;
      torque[i2][1] -= q2*dttorque[1] + tor2;
      torque[i2][2] -= q2*dttorque[2] + tor3;
    }

    //if (evflag) ev_tally(i1,i2,nlocal,newton_bond,ebond,0./*fbond*/,delx,dely,delz);
  }
}

/* ---------------------------------------------------------------------- */

void BondMCA::compute_bond_state()
{
  int *num_bond = atom->num_bond;
  int **bond_atom = atom->bond_atom;
  int *tag = atom->tag; // tag of atom is their ID number
  double ***bond_hist = atom->bond_hist;
  double **x = atom->x;

  int nbondlist = neighbor->nbondlist;
  int **bondlist = neighbor->bondlist;
  double cutoff=neighbor->skin;
  ///AS I do not whant to use it, because it requires to be copied every step/// double **bondhistlist = neighbor->bondhistlist;

fprintf(logfile, "BondMCA::compute_bond_state \n"); ///AS DEBUG

  if(breakmode == BREAKSTYLE_STRESS_TEMP)
  {
      if(!fix_Temp) error->all(FLERR,"Internal error in BondMCA::compute_bond_state");
      Temp = fix_Temp->vector_atom;
  }

  for (int n = 0; n < nbondlist; n++) {
    
	//1st check if bond is broken,
    if(bondlist[n][3])
    {
		//printf("bond %d allready broken\n",n);
        continue;
    }

    int i1,i2,type;
    double delx,dely,delz;
    double rsq,r;

    i1 = bondlist[n][0];
    i2 = bondlist[n][1];

    //2nd check if bond overlap the box-borders
    if (x[i1][0]<(domain->boxlo[0]+cutoff)) {
	bondlist[n][3]=1;
	continue;
    } else if (x[i1][0]>(domain->boxhi[0]-cutoff)) {
	bondlist[n][3]=1;
	continue;
    } else if (x[i1][1]<(domain->boxlo[1]+cutoff)) {
	bondlist[n][3]=1;
	continue;
    } else if (x[i1][1]>(domain->boxhi[1]-cutoff)) {
	bondlist[n][3]=1;
	continue;
    } else if (x[i1][2]<(domain->boxlo[2]+cutoff)) {
	bondlist[n][3]=1;
	continue;
    } else if (x[i1][2]>(domain->boxhi[2]-cutoff)) {
	bondlist[n][3]=1;
	continue;
    } 
    if (x[i2][0]<(domain->boxlo[0]+cutoff)) {
	bondlist[n][3]=1;
	continue;
    } else if (x[i2][0]>(domain->boxhi[0]-cutoff)) {
	bondlist[n][3]=1;
	continue;
    } else if (x[i2][1]<(domain->boxlo[1]+cutoff)) {
	bondlist[n][3]=1;
	continue;
    } else if (x[i2][1]>(domain->boxhi[1]-cutoff)) {
	bondlist[n][3]=1;
	continue;
    } else if (x[i2][2]<(domain->boxlo[2]+cutoff)) {
	bondlist[n][3]=1;
	continue;
    } else if (x[i2][2]>(domain->boxhi[2]-cutoff)) {
	bondlist[n][3]=1;
	continue;
    }

    type = bondlist[n][2];
    //rbmin=rb[type]*MIN(q1,q2); //lamda * min(rA,rB) see P.Cundall, "A bonded particle model for rock"

    delx = x[i1][0] - x[i2][0];
    dely = x[i1][1] - x[i2][1];
    delz = x[i1][2] - x[i2][2];
    //domain->minimum_image(delx,dely,delz); ??

    rsq = delx*delx + dely*dely + delz*delz;
    r = sqrt(rsq);

    //flag breaking of bond if criterion met
    if(breakmode == BREAKSTYLE_SIMPLE)
    {
        if(r > 2. * r_break[type])
        {
            //NP fprintf(screen,"r %f, 2. * r_break[type] %f \n",r,2. * r_break[type]);
            bondlist[n][3] = 1;
            //NP error->all(FLERR,"broken");
        }
    }
    else //NP stress or stress_temp
    {
      int n1;

      for (n1 = 0; n1 < num_bond[i1]; n1++) {
        if (bond_atom[i1][n1]==tag[i2]) break;
fprintf(logfile, "BondMCA::compute_bond_state bond_atom[i1][%d]=%d \n", n1, bond_atom[i1][n1]); ///AS DEBUG
      }
      if (n1 == num_bond[i1]) error->all(FLERR,"Internal error in BondMCA::compute_bond_state n1 not found");

      double *bond_hist1 = bond_hist[i1][n1];
      double nforce_mag = bond_hist1[P];
      double tforce_mag = vectorMag3D(&bond_hist1[SX]);

      bool nstress = sigman_break[type] < (nforce_mag/* + 2.*ttorque_mag/J*rbmin*/);
      bool tstress = tau_break[type]    < (tforce_mag/* +    ntorque_mag/J*rbmin*/);
      bool toohot = false;
      if(breakmode == BREAKSTYLE_STRESS_TEMP)
      {
         toohot = 0.5 * (Temp[i1] + Temp[i2]) > T_break[type];
         //NL //fprintf(screen,"Temp[i1] %f Temp[i2] %f, T_break[type] %f\n",Temp[i1],Temp[i2],T_break[type]);
      }

      if(nstress || tstress || toohot)
      {
         bondlist[n][3] = 1;
         //fprintf(screen,"broken bond %d at step %d\n",n,update->ntimestep);
         //NL //if(toohot)fprintf(screen,"   it was too hot\n");
         //NL //if(nstress)fprintf(screen,"   it was nstress\n");
         //NL //if(tstress)fprintf(screen,"   it was tstress\n");
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void BondMCA::compute(int eflag, int vflag)
{

  compute_total_force(eflag, vflag);
  ///AS compute_bond_state();

}

/* ---------------------------------------------------------------------- */

void BondMCA::allocate()
{
  allocated = 1;
  int n = atom->nbondtypes;

  memory->create(rb,n+1,"bond:rb");
  memory->create(Sn,n+1,"bond:Sn");
  memory->create(St,n+1,"bond:St");

  memory->create(r_break,(n+1),"bond:r_break");
  memory->create(sigman_break,(n+1),"bond:sigman_break");
  memory->create(tau_break,(n+1),"bond:tau_break");
  memory->create(T_break,(n+1),"bond:T_break");

  memory->create(setflag,(n+1),"bond:setflag");
  for (int i = 1; i <= n; i++)
    setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   set coeffs for one or more types
------------------------------------------------------------------------- */

void BondMCA::coeff(int narg, char **arg)
{
  if(narg < 4)  error->all(FLERR,"Small number of args for MCA bond coefficients (<4)");

  double rb_one = force->numeric(FLERR,arg[1]);
  double Sn_one = force->numeric(FLERR,arg[2]);
  double St_one = force->numeric(FLERR,arg[3]);

  /*NL*///fprintf(screen,"Sn %f, St%f\n",Sn_one,St_one);

  if(force->numeric(FLERR,arg[4]) == 0. )
  {
      breakmode = BREAKSTYLE_SIMPLE;
      if (narg != 6) error->all(FLERR,"Incorrect SIMPLE arg for MCA bond coefficients");
  }
  else if(force->numeric(FLERR,arg[4]) == 1. )
  {
      breakmode = BREAKSTYLE_STRESS;
      if (narg != 7) error->all(FLERR,"Incorrect STRESS arg for MCA bond coefficients");
  }
  else if(force->numeric(FLERR,arg[4]) == 2. )
  {
      breakmode = BREAKSTYLE_STRESS_TEMP;
      if (narg != 8) error->all(FLERR,"Incorrect STRESS_TEMP arg for MCA bond coefficients");
  }
  else  error->all(FLERR,"Incorrect args for MCA bond coefficients");

  if (!allocated) allocate();

  double r_break_one,sigman_break_one,tau_break_one,T_break_one;

  if(breakmode == BREAKSTYLE_SIMPLE) r_break_one = force->numeric(FLERR,arg[5]);
  else
  {
      sigman_break_one = force->numeric(FLERR,arg[5]);
      tau_break_one = force->numeric(FLERR,arg[6]);
      if(breakmode == BREAKSTYLE_STRESS_TEMP) T_break_one = force->numeric(FLERR,arg[7]);
  }

  int ilo,ihi;
  force->bounds(arg[0],atom->nbondtypes,ilo,ihi);
  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    rb[i] = rb_one;
    Sn[i] = Sn_one;
    St[i] = St_one;
    if(breakmode == BREAKSTYLE_SIMPLE) r_break[i] = r_break_one;
    else
    {
        sigman_break[i] = sigman_break_one;
        tau_break[i] = tau_break_one;
        if(breakmode == BREAKSTYLE_STRESS_TEMP) T_break[i] = T_break_one;
    }
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR,"Incorrect args for bond coefficients - or the bonds are not initialized in create_atoms");
}

/* ----------------------------------------------------------------------
   return an equilbrium bond length
------------------------------------------------------------------------- */

double BondMCA::equilibrium_distance(int i)
{
  //NP ATTENTION: this is _not_ correct - and has implications on fix shake, pair_lj_cut_coul_long and pppm
  //NP it is not possible to define a general equilibrium distance for this bond model
  //NP as rotational degree of freedom is present
  return 0.;
}

/* ----------------------------------------------------------------------
   proc 0 writes out coeffs to restart file
------------------------------------------------------------------------- */

void BondMCA::write_restart(FILE *fp)
{
  fwrite(&rb[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&Sn[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&St[1],sizeof(double),atom->nbondtypes,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them
------------------------------------------------------------------------- */

void BondMCA::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
    fread(&rb[1],sizeof(double),atom->nbondtypes,fp);
    fread(&Sn[1],sizeof(double),atom->nbondtypes,fp);
    fread(&St[1],sizeof(double),atom->nbondtypes,fp);
  }
  MPI_Bcast(&rb[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&Sn[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&St[1],atom->nbondtypes,MPI_DOUBLE,0,world);

  for (int i = 1; i <= atom->nbondtypes; i++) setflag[i] = 1;
}

/* ---------------------------------------------------------------------- */

double BondMCA::single(int type, double rsq, int i, int j, double &fforce)
{
  error->all(FLERR,"MCA bond does not support this feature");
  /*double r = sqrt(rsq);
  double dr = r - r0[type];
  double rk = k[type] * dr;
  return rk*dr;*/
  return 0.;
}
