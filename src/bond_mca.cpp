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
#include "atom_vec_mca.h"

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

using namespace MCAAtomConst;

/* ---------------------------------------------------------------------- */

BondMCA::BondMCA(LAMMPS *lmp) : Bond(lmp)
{
    /* number of entries in bondhistlist. bondhistlist[number of bond][number of value (from 0 to number given here)]
     * so with this number you can modify how many pieces of information you save with every bond.
     * following dependencies and processes for saving,copying,growing the bondhistlist:
     * For mca we need BOND_HIST_LEN history values
     * see bond.cpp: void Bond::n_granhistory(int nhist) {ngranhistory = nhist; atom->n_bondhist = ngranhistory; if(){FLERR}}
     */
    n_granhistory(BOND_HIST_LEN);
///AS TODO I do not whant to use it, because it requires LIGGGHTS bond.cpp

    /* allocation of memory are here:
     * neighbor.cpp:       memory->create(bondhistlist,maxbond,atom->n_bondhist,"neigh:bondhistlist");
     * neigh_bond.cpp:     memory->grow(bondhistlist,maxbond,atom->n_bondhist,"neighbor:bondhistlist");
     * atom_vec_bond_gran.cpp:  memory->grow(atom->bond_hist,nmax,atom->bond_per_atom,atom->n_bondhist,"atom:bond_hist");
     * atom_vec_mca.cpp:  memory->grow(atom->bond_hist,nmax,atom->bond_per_atom,atom->n_bondhist,"atom:bond_hist");
     */
    if(!atom->style_match("mca"))
      error->all(FLERR,"A bond style 'mca' can only be used together with atom style 'mca'");
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
///AS TODO                                                                                                    "bond mca")); ????
}

/* ---------------------------------------------------------------------- */

void BondMCA::compute(int eflag, int vflag)
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

//fprintf(logfile, "BondMCA::compute \n"); ///AS DEBUG

  if(breakmode == BREAKSTYLE_STRESS_TEMP) {
    if(!fix_Temp) error->all(FLERR,"Internal error in BondMCA::compute");
    Temp = fix_Temp->vector_atom;
  }

//fprintf(logfile,"boxlo[0]=%g boxhi[0]=%g cutoff=%g \n",domain->boxlo[0],domain->boxhi[0],cutoff);
//fprintf(logfile,"boxlo[1]=%g boxhi[1]=%g cutoff=%g \n",domain->boxlo[1],domain->boxhi[1],cutoff);
//fprintf(logfile,"boxlo[2]=%g boxhi[2]=%g cutoff=%g \n",domain->boxlo[2],domain->boxhi[2],cutoff);

  for (int n = 0; n < nbondlist; n++) {

    //1st check if bond is broken,
    if(bondlist[n][3]) {
      fprintf(logfile,"BondMCA::compute bond %d allready broken\n",n);
      continue;
    }

    int i1,i2,type;
    double delx,dely,delz;
    double rsq,r;

    i1 = bondlist[n][0];
    i2 = bondlist[n][1];

    //2nd check if bond overlap the box-borders
    if (x[i1][0]<(domain->boxlo[0]-cutoff)) {
      bondlist[n][3]=1;
    } else if (x[i1][0]>(domain->boxhi[0]+cutoff)) {
      bondlist[n][3]=1;
    } else if (x[i1][1]<(domain->boxlo[1]-cutoff)) {
      bondlist[n][3]=1;
    } else if (x[i1][1]>(domain->boxhi[1]+cutoff)) {
      bondlist[n][3]=1;
    } else if (x[i1][2]<(domain->boxlo[2]-cutoff)) {
      bondlist[n][3]=1;
    } else if (x[i1][2]>(domain->boxhi[2]+cutoff)) {
      bondlist[n][3]=1;
    }
    if(bondlist[n][3]==1)fprintf(logfile,"BondMCA::compute bond %d broken by domain overlap x[%d]= %g %g %g\n",n,i1,x[i1][0],x[i1][1],x[i1][2]);
    continue;

    if (x[i2][0]<(domain->boxlo[0]-cutoff)) {
      bondlist[n][3]=1;
    } else if (x[i2][0]>(domain->boxhi[0]+cutoff)) {
      bondlist[n][3]=1;
    } else if (x[i2][1]<(domain->boxlo[1]-cutoff)) {
      bondlist[n][3]=1;
    } else if (x[i2][1]>(domain->boxhi[1]+cutoff)) {
      bondlist[n][3]=1;
    } else if (x[i2][2]<(domain->boxlo[2]-cutoff)) {
      bondlist[n][3]=1;
    } else if (x[i2][2]>(domain->boxhi[2]+cutoff)) {
      bondlist[n][3]=1;
    }
    if(bondlist[n][3]==1)fprintf(logfile,"BondMCA::compute bond %d broken by domain overlap x[%d]= %g %g %g\n",n,i2,x[i2][0],x[i2][1],x[i2][2]);
    continue;

    type = bondlist[n][2];

    delx = x[i1][0] - x[i2][0];
    dely = x[i1][1] - x[i2][1];
    delz = x[i1][2] - x[i2][2];
    //domain->minimum_image(delx,dely,delz); ??

    rsq = delx*delx + dely*dely + delz*delz;
    r = sqrt(rsq);

    // breaking the bond if criterion met
    if(breakmode == BREAKSTYLE_SIMPLE) {
      if(r > (2. * r_break[type])) {
        fprintf(logfile,"r= %f > 2.*r_break[%d]= %f \n",r,type,2.*r_break[type]);
        bondlist[n][3] = 1;
        error->all(FLERR,"broken");
      }
    }
    else // stress or stress_temp
    {
      int n1;

      for (n1 = 0; n1 < num_bond[i1]; n1++) {
        if (bond_atom[i1][n1]==tag[i2]) break;
//fprintf(logfile, "BondMCA::compute bond_atom[%d][%d]=%d \n", i1, n1, bond_atom[i1][n1]); ///AS DEBUG
      }
      if (n1 == num_bond[i1]) error->all(FLERR,"Internal error in BondMCA::compute n1 not found");

      double *bond_hist1 = bond_hist[i1][n1];
      double nforce_mag = bond_hist1[P]; // - atom->mean_stress[i1]
      double tforce_mag = sqrt(vectorMag3D(&bond_hist1[SX]));

      bool nstress = sigman_break[type] < (nforce_mag/* + 2.*ttorque_mag/J*rbmin*/);
      bool tstress = tau_break[type]    < (tforce_mag/* +    ntorque_mag/J*rbmin*/);
      bool toohot = false;
      if(breakmode == BREAKSTYLE_STRESS_TEMP) {
        toohot = 0.5 * (Temp[i1] + Temp[i2]) > T_break[type];
        //fprintf(screen,"Temp[i1] %f Temp[i2] %f, T_break[type] %f\n",Temp[i1],Temp[i2],T_break[type]);
      }

      if(nstress || tstress || toohot) {
        bondlist[n][3] = 1;
        fprintf(logfile,"broken bond %d at step %d\n",n,update->ntimestep);
        if(toohot) fprintf(logfile,"   it was too hot\n");
        if(nstress) fprintf(logfile,"   it was nstress: sigman_break[%d]=%g < nforce_mag=%g\n", type, sigman_break[type], nforce_mag);
        if(tstress) fprintf(logfile,"   it was tstress: tau_break[%d]=%g < tstress=%g\n", type, tau_break[type], tforce_mag);
      }
    }
  }
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

  if(force->numeric(FLERR,arg[4]) == 0. ) {
      breakmode = BREAKSTYLE_SIMPLE;
      if (narg != 6) error->all(FLERR,"Incorrect SIMPLE arg for MCA bond coefficients");
  }
  else if(force->numeric(FLERR,arg[4]) == 1. ) {
      breakmode = BREAKSTYLE_STRESS;
      if (narg != 7) error->all(FLERR,"Incorrect STRESS arg for MCA bond coefficients");
  }
  else if(force->numeric(FLERR,arg[4]) == 2. ) {
      breakmode = BREAKSTYLE_STRESS_TEMP;
      if (narg != 8) error->all(FLERR,"Incorrect STRESS_TEMP arg for MCA bond coefficients");
  }
  else  error->all(FLERR,"Incorrect args for MCA bond coefficients");

  if (!allocated) allocate();

  double r_break_one,sigman_break_one,tau_break_one,T_break_one;

  if(breakmode == BREAKSTYLE_SIMPLE) r_break_one = force->numeric(FLERR,arg[5]);
  else {
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
    else {
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
