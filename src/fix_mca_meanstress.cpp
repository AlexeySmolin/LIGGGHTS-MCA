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
#include "mpi.h"
#include "string.h"
#include "stdlib.h"
#include "update.h"
#include "domain.h"
#include "respa.h"
#include "atom.h"
#include "force.h"
#include "modify.h"
#include "pair_mca.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "memory.h"
#include "error.h"
#include "sph_kernels.h"
#include "fix_property_atom.h"
#include "timer.h"
#include "fix_mca_meanstress.h"
#include "atom_vec_mca.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MCAAtomConst;

///#define NO_MEANSTRESS

/* ---------------------------------------------------------------------- */

FixMCAMeanStress::FixMCAMeanStress(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
//    fprintf(logfile,"constructor FixMCAMeanStress ###########\n");
    restart_global = 1; ///?
}

/* ---------------------------------------------------------------------- */

FixMCAMeanStress::~FixMCAMeanStress()
{

}

/* ---------------------------------------------------------------------- */

int FixMCAMeanStress::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE; /// ПРОБУЕМ
///  mask |= POST_INTEGRATE;
///  mask |= POST_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixMCAMeanStress::init()
{
  if(force->pair == NULL) error->all(FLERR,"Fix mca/meanstress force->pair is NULL");
  if(!(force->pair_match("mca",1)))
     error->all(FLERR,"Fix mca/meanstress can only be used together with dedicated 'mca' pair styles");

  if(!(force->bond_match("mca")))
     error->all(FLERR,"Fix mca/meanstress can only be used together with dedicated 'mca' bond styles");
}

/* ---------------------------------------------------------------------- */

inline void  FixMCAMeanStress::swap_prev()
{
  int i,k;
//fprintf(logfile,"FixMCAMeanStress::swap_prev\n"); ///AS DEBUG TRACE

  const int * const num_bond = atom->num_bond;
  double ***bond_hist = atom->bond_hist;
  const int nlocal = atom->nlocal;
///  const int nmax = atom->nmax;

  {
    double *tmp;
    tmp = atom->mean_stress;
    atom->mean_stress = atom->mean_stress_prev;
    atom->mean_stress_prev = tmp;

    tmp = atom->equiv_stress;
    atom->equiv_stress = atom->equiv_stress_prev;
    atom->equiv_stress_prev = tmp;
  }

#if defined (_OPENMP)
#pragma omp parallel for private(i,k) shared (bond_hist) default(none) schedule(static)
#endif
  for (i = 0; i < nlocal; i++) {/// i < nmax; i++) {///
    if (num_bond[i] == 0) continue;

    for(k = 0; k < num_bond[i]; k++)
    {
      double *bond_hist_ik = bond_hist[i][k];
///if(i == 13) fprintf(logfile,"PairMCA::swap_prev i=%d j=%d nv_pre= %20.12e %20.12e %20.12e nv= %20.12e %20.12e %20.12e \n",i,k,bond_hist_ik[NX_PREV],bond_hist_ik[NY_PREV],bond_hist_ik[NZ_PREV],bond_hist_ik[NX],bond_hist_ik[NY],bond_hist_ik[NZ]); ///AS DEBUG
      bond_hist_ik[R_PREV] = bond_hist_ik[R];
      bond_hist_ik[P_PREV] = bond_hist_ik[P];
      bond_hist_ik[NX_PREV] = bond_hist_ik[NX];
      bond_hist_ik[NY_PREV] = bond_hist_ik[NY];
      bond_hist_ik[NZ_PREV] = bond_hist_ik[NZ];
      bond_hist_ik[YX_PREV] = bond_hist_ik[YX];
      bond_hist_ik[YY_PREV] = bond_hist_ik[YY];
      bond_hist_ik[YZ_PREV] = bond_hist_ik[YZ];
      bond_hist_ik[SHX_PREV] = bond_hist_ik[SHX];
      bond_hist_ik[SHY_PREV] = bond_hist_ik[SHY];
      bond_hist_ik[SHZ_PREV] = bond_hist_ik[SHZ];
    }
  }
}

/* ---------------------------------------------------------------------- */

inline void  FixMCAMeanStress::predict_mean_stress()
{
  int i,j,k,jk,itype,jtype;

  double **x = atom->x;
  double **v = atom->v;
  double *mean_stress = atom->mean_stress;
  double *mean_stress_prev = atom->mean_stress_prev;
  const double mca_radius = atom->mca_radius;
  const int * const tag = atom->tag;
  const int * const type = atom->type;

  int **bond_atom = atom->bond_atom;
  const int * const num_bond = atom->num_bond;
  const int newton_bond = force->newton_bond;
  double ***bond_hist = atom->bond_hist;
  const int nbondlist = neighbor->nbondlist;
  int ** const bondlist = neighbor->bondlist;
  int ** const bond_index = atom->bond_index;
  const double dtImpl = (atom->implicit_factor)*update->dt; ///AS it is equivalent to damping force.
  const int Nc = atom->coord_num;
  const int nlocal = atom->nlocal;
///  const int nmax = atom->nmax;
  const PairMCA * const mca_pair = (PairMCA*) force->pair;

//fprintf(logfile,"FixMCAMeanStress::predict_mean_stress\n"); ///AS DEBUG TRACE
#if defined (_OPENMP)
#pragma omp parallel for private(i,j,k,jk,itype,jtype) shared(x,v,mean_stress,mean_stress_prev,bond_atom,bond_hist) default(shared) schedule(static)
#endif
  for (i = 0; i < nlocal; i++) {/// i < nmax; i++) {///
    if (num_bond[i] == 0) continue;

    double xtmp,ytmp,ztmp,delx,dely,delz,vxtmp,vytmp,vztmp,r,r0,rsq,rinv;
    double rKHi,rKHj;// (1-2*G)/(3*K) for atom i (j)
    double rHi,rHj;  // 2*G for atom i (j)
    double pi,pj;
    double d_p;
    double d_e,d_e0;
    double rdSgmi;
    double rK1, rKn;
    int Ni, Nj; // number of interacting neighbors (bonds) for i (j)

    Ni = num_bond[i];
    if (Ni > Nc) Ni = Nc; // not increse "rigidity"
    itype = type[i];
    rHi = 2.0*mca_pair->G[itype][itype];
    ///AS TODO make this property global as in 'fix_check_timestep_gran.cpp' :
    /// Y = static_cast<FixPropertyGlobal*>(modify->find_fix_property("youngsModulus","property/global","peratomtype",max_type,0,style));

    rKHi = mca_pair->K[itype][itype]; rKHi = 1. - rHi / (3. * rKHi);
    rK1 = (double)Nc / (Nc - rKHi);
    rKn = (Nc + rKHi/(1.0 - rKHi)) / ((double)Nc);
    rHi *= rK1 + ((rKn-rK1)*(Ni-1))/((double)(Nc-1)); // fix "rigidity" with accounting of # of bonds

    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    vxtmp = v[i][0];
    vytmp = v[i][1];
    vztmp = v[i][2];

    rdSgmi = 0.0;
    for(k = 0; k < num_bond[i]; k++)
    {
      j = atom->map(bond_atom[i][k]);
      if (j == -1) {
        char str[512];
        sprintf(str,
                "Bond atoms %d %d missing at step " BIGINT_FORMAT,
                tag[i],bond_atom[i][k],update->ntimestep);
        error->one(FLERR,str);
      }
      j = domain->closest_image(i,j);
/*{
{
  int *sametag = atom->sametag;
  double **x = atom->x;
  double *xi = x[i];

  int closest = j;
  double delx = xi[0] - x[j][0];
  double dely = xi[1] - x[j][1];
  double delz = xi[2] - x[j][2];
  double rsqmin = delx*delx + dely*dely + delz*delz;
  double rsq;

  while (sametag[j] >= 0) {
    j = sametag[j];
if(tag[i]==10) fprintf(logfile,"sametag[j]=%d (%g %g %g) \n",j,x[j][0],x[j][1],x[j][2]);
    delx = xi[0] - x[j][0];
    dely = xi[1] - x[j][1];
    delz = xi[2] - x[j][2];
    rsq = delx*delx + dely*dely + delz*delz;
    if (rsq < rsqmin) {
      rsqmin = rsq;
      closest = j;
    }
  }
  j = closest;
}
if(tag[i]==10) fprintf(logfile,"FixMCAMeanStress::predict_mean_stress bond_atom[%d][%d]=%d map()=%d \n",i,k,bond_atom[i][k],j);

}*/
      int found = 0;
///fprintf(logfile,"FixMCAMeanStress::mean_stress_predict: i=%d j=%d num_bond[j]=%d\n",i,j,num_bond[j]);
      for(jk = 0; jk < num_bond[j]; jk++) {
///fprintf(logfile,"FixMCAMeanStress::mean_stress_predict: i=%d j=%d jk=%d num_bond[j]=%d\n",i,j,jk,num_bond[j]);
        if(bond_atom[j][jk] == tag[i]) {found = 1; break; }
      }
      if (!found) error->all(FLERR,"FixMCAMeanStress::mean_stress_predict 'jk' not found");

      Nj = num_bond[j];
      if (Nj > Nc) Nj = Nc; // not increse "rigidity" if atom has more bonds
      jtype = type[j];
      rHj = 2.0*mca_pair->G[jtype][jtype];
      rKHj = mca_pair->K[jtype][jtype]; rKHj = 1. - rHj / (3. * rKHj);
      rK1 = (double)Nc / (Nc - rKHj);
      rKn = (Nc + rKHj/(1.0 - rKHj)) / ((double)Nc);
      rHj *= rK1 + ((rKn-rK1)*(Nj-1))/((double)(Nc-1)); // fix "rigidity" with accounting of on # of bonds

      delx = xtmp - x[j][0] + dtImpl*(vxtmp - v[j][0]); // Implicit update of the distance. It is equivalent to use of damping force.
      dely = ytmp - x[j][1] + dtImpl*(vytmp - v[j][1]);
      delz = ztmp - x[j][2] + dtImpl*(vztmp - v[j][2]);
      rsq = delx*delx + dely*dely + delz*delz;
      r = sqrt(rsq);
      rinv = -1. / r; // "-" means that unit vector is from i1 to i2

      double *bond_hist_ik = bond_hist[i][k];
      r0 = bond_hist_ik[R_PREV];
      pi = bond_hist_ik[P_PREV];
      if (newton_bond) {
        //pj = bond_hist_ik[PJ_PREV]; it means we store bonds only for i < j, but allocate memory for both. why?
        error->all(FLERR,"FixMCAMeanStress::mean_stress_predict does not support 'newton_bond on'");
      } else
        pj = bond_hist[j][jk][P_PREV];

      d_e0 = (r - r0) / mca_radius;
      d_e  = (pj - pi + rHj*d_e0) / (rHi + rHj);
      d_p = rHi*d_e;

      if (nbondlist != 0) {
        int bond_index_i = bond_index[i][k];
        if((bond_index_i >= nbondlist) || (bond_index_i < 0)) {
          char str[512];
          sprintf(str,"wrong bond_index[%d][%d()]=%d (nbondlist=%d) at step " BIGINT_FORMAT,
                  i,k,bond_index_i,nbondlist,update->ntimestep);
          error->one(FLERR,str);
        }
        int bond_state = bondlist[bond_index_i][3];
        if((!bond_state) && ((pi + d_p)>0.0)) { // if happens?!
          d_p = 0.0;
        }
      }
      rdSgmi += d_p;
      bond_hist_ik[R] = r;
//      bond_hist[j][jk][R] = r; //TODO do we need it for j?
      bond_hist_ik[NX] = delx*rinv; ///TODO will do it later in PairMCA::compute_total_force because here we use implicit distance
      bond_hist_ik[NY] = dely*rinv;
      bond_hist_ik[NZ] = delz*rinv;
    }
#ifndef NO_MEANSTRESS
    mean_stress[i] = mean_stress_prev[i] + rdSgmi / Nc;
#endif
  }
}

/*
void FixMCAMeanStress::post_integrate()
{
fprintf(logfile,"FixMCAMeanStress::post_integrate\n"); ///AS DEBUG TRACE
  swap_prev();
  predict_mean_stress();
} */

void FixMCAMeanStress::pre_force(int vflag)
{
//fprintf(logfile, "FixMCAMeanStress::pre_force \n");///AS DEBUG
  swap_prev();
  predict_mean_stress();

  comm->forward_comm_fix(this); // to exchange needed fields
}

int FixMCAMeanStress::pack_comm(int n, int *list, double *buf,
                             int pbc_flag, int *pbc)
{
  int i,j,m;
  int k,l;

  const int * const num_bond = atom->num_bond;
  double ** const theta = atom->theta;
  double ** const theta_prev = atom->theta_prev;
  const double * const mean_stress = atom->mean_stress;
  const double * const mean_stress_prev = atom->mean_stress_prev;
  const double * const equiv_stress = atom->equiv_stress;
  const double * const equiv_stress_prev = atom->equiv_stress_prev;
  const double * const equiv_strain = atom->equiv_strain;
  double *** const bond_hist = atom->bond_hist;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = theta[j][0];
    buf[m++] = theta[j][1];
    buf[m++] = theta[j][2];
    buf[m++] = theta_prev[j][0];
    buf[m++] = theta_prev[j][1];
    buf[m++] = theta_prev[j][2];
    buf[m++] = mean_stress[j];
    buf[m++] = mean_stress_prev[j];
    buf[m++] = equiv_stress[j];
    buf[m++] = equiv_stress_prev[j];
    buf[m++] = equiv_strain[j];

    if(atom->n_bondhist) {
      for (k = 0; k < num_bond[j]; k++)
        for (l = 0; l < MX; l++)
          buf[m++] = bond_hist[j][k][l];
    }
  }
//  fprintf(logfile,"FixMCAMeanStress::pack_comm m=%d n=%d [%d - %d]\n",m,n,list[0],list[n-1]);
  return m;
}

/* ---------------------------------------------------------------------- */

void FixMCAMeanStress::unpack_comm(int n, int first, double *buf)
{
  int i,m,last;
  int k,l;

  const int *num_bond = atom->num_bond;
  double **theta = atom->theta;
  double **theta_prev = atom->theta_prev;
  double *mean_stress = atom->mean_stress;
  double *mean_stress_prev = atom->mean_stress_prev;
  double *equiv_stress = atom->equiv_stress;
  double *equiv_stress_prev = atom->equiv_stress_prev;
  double *equiv_strain = atom->equiv_strain;
  double ***bond_hist = atom->bond_hist;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    if (i == atom->nmax) error->all(FLERR,"FixMCAMeanStress::unpack_comm i==atom->nmax");
    theta[i][0] = buf[m++];
    theta[i][1] = buf[m++];
    theta[i][2] = buf[m++];
    theta_prev[i][0] = buf[m++];
    theta_prev[i][1] = buf[m++];
    theta_prev[i][2] = buf[m++];
    mean_stress[i] = buf[m++];
    mean_stress_prev[i] = buf[m++];
    equiv_stress[i] = buf[m++];
    equiv_stress_prev[i] = buf[m++];
    equiv_strain[i] = buf[m++];

    if(atom->n_bondhist) {
      for (k = 0; k < num_bond[i]; k++)
        for (l = 0; l < MX; l++)
          bond_hist[i][k][l] = buf[m++];
    }
  }
//fprintf(logfile,"FixMCAMeanStress::unpack_comm m=%d n=%d [%d - %d]\n",m,n,first,last);
}

