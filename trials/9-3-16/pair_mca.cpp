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

#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "pair_mca.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "update.h"
#include "neigh_list.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairMCA::PairMCA(LAMMPS *lmp) : Pair(lmp)
{
  writedata = 1;
  single_enable = 0;
}

/* ---------------------------------------------------------------------- */

PairMCA::~PairMCA()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(G);
    memory->destroy(K);
  }
}

/* ---------------------------------------------------------------------- */

void PairMCA::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,fpair;
  double evdwl,factor_lj; ///AS
  double r,rsq;
  double e_ij;
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double **omega = atom->omega;
  double **torque = atom->torque;
  double *rmass = atom->rmass;
  double *inertia = atom->q;
  double **theta = atom->mu;
  int coord_num  = atom->coord_num;
  double mca_radius  = atom->mca_radius;
  double contact_area  = atom->contact_area;
  double *mca_inertia  = atom->q;
  double *mean_stress  = atom->p;
  double *equiv_stress  = atom->s0;
  double *equiv_strain  = atom->e;

  const double dtImpl = 0.5*update->dt; // for implicit estimation of displacement
  const double cutsq = 5.76*mca_radius*mca_radius; // 1.44*d*d

  int *type = atom->type;
  int nlocal = atom->nlocal;
 ///AS  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

//fprintf(stderr, "PairMCA::compute dtImpl= %g\n", dtImpl);

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
///AS      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0] + dtImpl*(v[i][0] - v[j][0]);
      dely = ytmp - x[j][1] + dtImpl*(v[i][1] - v[j][1]);
      delz = ztmp - x[j][2] + dtImpl*(v[i][2] - v[j][2]);
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

//if(ii == (inum-2)) fprintf(stderr, "PairMCA::compute i= %d j= %d rsq= %g cutsq = %g\n", i, j, rsq, cutsq);
      if (rsq < cutsq) {
        r = sqrt(rsq);
        e_ij = (0.5*r - mca_radius) / mca_radius;
        fpair = contact_area * e_ij * 2.0 * G[itype][jtype]
                / r; //this allows to get cos of normal direction by mult on delx 
//if(ii == (inum-2)) fprintf(stderr, "PairMCA::compute i= %d j= %d e_ij= %g fpair = %g G=%g mass[i]= %g\n", i, j, e_ij, fpair*r,G[itype][jtype],rmass[i]);

        f[i][0] -= delx*fpair;
        f[i][1] -= dely*fpair;
        f[i][2] -= delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] += delx*fpair;
          f[j][1] += dely*fpair;
          f[j][2] += delz*fpair;
        }

        if (eflag)
          evdwl = 2.0 * G[itype][jtype] * e_ij * e_ij;///AS ??? energy?

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             evdwl,0.0,fpair,delx,dely,delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairMCA::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(G,n+1,n+1,"pair:G");
  memory->create(K,n+1,n+1,"pair:K");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairMCA::settings(int narg, char **arg)
{
/*
  if (narg != 1) error->all(FLERR,"Illegal pair_style command");

  cut_global = force->numeric(FLERR,arg[0]); ///AS Later we will change it

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i+1; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }*/
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairMCA::coeff(int narg, char **arg)
{
  if (narg < 4)
    error->all(FLERR,"Incorrect args for mca pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(arg[0],atom->ntypes,ilo,ihi);
  force->bounds(arg[1],atom->ntypes,jlo,jhi);

  double G_one = force->numeric(FLERR,arg[2]);

  double K_one = force->numeric(FLERR,arg[3]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      G[i][j] = G_one;
      K[i][j] = K_one;
      setflag[i][j] = 1; // 0/1 = whether each i,j has been set
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for mca pair coefficients");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairMCA::init_one(int i, int j)
{
  // always mix Gs geometrically

  if (setflag[i][j] == 0) {
    G[i][j] =(G[i][i] * G[j][j]) / (G[i][i] + G[j][j]);
    K[i][j] = (K[i][i] * K[j][j]) / (K[i][i] + K[j][j]);
  }

  G[j][i] = G[i][j];
  K[j][i] = K[i][j];

  return 1.2*atom->mca_radius; // have to return cut_global
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairMCA::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&G[i][j],sizeof(double),1,fp);
        fwrite(&K[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairMCA::read_restart(FILE *fp)
{
  read_restart_settings(fp);

  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) fread(&setflag[i][j],sizeof(int),1,fp);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          fread(&G[i][j],sizeof(double),1,fp);
          fread(&K[i][j],sizeof(double),1,fp);
        }
        MPI_Bcast(&G[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&K[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairMCA::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairMCA::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    fread(&cut_global,sizeof(double),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
  }
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairMCA::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp,"%d %g\n",i,G[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairMCA::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp,"%d %d %g %g\n",i,j,G[i][j],K[i][j]);
}

/* ---------------------------------------------------------------------- (

double PairMCA::single(int i, int j, int itype, int jtype, double rsq,
                        double factor_coul, double factor_lj,
                        double &fforce)
{
  double r,arg,philj;

  r = sqrt(rsq);
  arg = MY_PI*r/cut[itype][jtype];
  fforce = factor_lj * G[itype][jtype] * sin(arg) * MY_PI/cut[itype][jtype]/r;

  philj = G[itype][jtype] * (1.0+cos(arg));
  return factor_lj*philj;
}*/

/* ---------------------------------------------------------------------- */

void *PairMCA::extract(const char *str, int &dim)
{
  dim = 2;
  if (strcmp(str,"a") == 0) return (void *) G; ///AS ???
  return NULL;
}
