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

///#define NO_MEANSTRESS // see also in pair_mca.cpp

/* ---------------------------------------------------------------------- */

FixMCAMeanStress::FixMCAMeanStress(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
//    if (logfile) fprintf(logfile,"constructor FixMCAMeanStress ###########\n");
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

  comm_forward = 10 + MAX_BONDS*(2 + BOND_HIST_LEN);  // theta[j][3] + theta_prev[j][3] + mean_stress[j] +
                                                      // mean_stress_prev[j] + equiv_stress_prev[j] + equiv_strain[j];
}

/* ---------------------------------------------------------------------- */

inline void  FixMCAMeanStress::swap_prev()
{
  int i,k;
//if (logfile) fprintf(logfile,"FixMCAMeanStress::swap_prev\n"); ///AS DEBUG TRACE

  const int * const num_bond = atom->num_bond;
  double ***bond_hist = atom->bond_hist;
  const int * const tag = atom->tag;
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
      double *bond_hist_ik = bond_hist[tag[i]-1][k];
///if(i == 13) if (logfile) fprintf(logfile,"PairMCA::swap_prev i=%d j=%d nv_pre= %20.12e %20.12e %20.12e nv= %20.12e %20.12e %20.12e \n",i,k,bond_hist_ik[NX_PREV],bond_hist_ik[NY_PREV],bond_hist_ik[NZ_PREV],bond_hist_ik[NX],bond_hist_ik[NY],bond_hist_ik[NZ]); ///AS DEBUG
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
  int i,j,k,jk,itype;

  double **x = atom->x;
  double **v = atom->v;
  double *mean_stress = atom->mean_stress;
  double *plastic_heat = atom->plastic_heat;
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

//if (logfile) fprintf(logfile,"FixMCAMeanStress::predict_mean_stress\n"); ///AS DEBUG TRACE
  // first loop for computing distance (R) normal vector (NX,NY,NZ) and mean strain increment
#if defined (_OPENMP)
#pragma omp parallel for private(i,j,k,jk,itype) shared(x,v,mean_stress,bond_atom,bond_hist) default(shared) schedule(static)
#endif
  for (i = 0; i < nlocal; i++) {/// i < nmax; i++) {///
    if (num_bond[i] == 0) continue;

    int ** const bond_mca = atom->bond_mca;
    double xtmp,ytmp,ztmp,vxtmp,vytmp,vztmp;
    double rHi;  // 2*G for atom i (j)
    double rK1, rKn;
    int Ni = num_bond[i]; // number of interacting neighbors (bonds) for i (j)
    if (Ni > Nc) Ni = Nc; // not increase "rigidity"
    double rM_I = (double)(Ni-1) / (double)(Nc-1);
    itype = type[i];
    double r3Ki = 3.0 * mca_pair->K[itype][itype];
    rK1 = mca_pair->ModulusPredictOne[itype][itype];
    rKn = mca_pair->ModulusPredictAll[itype][itype];
    rHi = rK1*(1.0-rM_I) + rKn*rM_I; // "rigidity" with accounting for # of bonds

    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    vxtmp = v[i][0];
    vytmp = v[i][1];
    vztmp = v[i][2];

    double rDeltaEpsMean_I = 0.0;
    for(k = 0; k < num_bond[i]; k++)
    {
      j = bond_mca[i][k];
/*      j = atom->map(bond_atom[i][k]);
      if (j == -1) {
        char str[512];
        sprintf(str,
                "Bond atoms %d %d missing at step " BIGINT_FORMAT,
                tag[i],bond_atom[i][k],update->ntimestep);
        error->one(FLERR,str);
      }
      j = domain->closest_image(i,j);
*/
      int found = 0;
      int i_tag = tag[i];
      int Nj = num_bond[j];
///if (logfile) fprintf(logfile,"FixMCAMeanStress::mean_stress_predict: i=%d j=%d num_bond[j]=%d\n",i,j,Nj);
      for(jk = 0; jk < Nj; jk++) {
///if (logfile) fprintf(logfile,"FixMCAMeanStress::mean_stress_predict: i=%d j=%d jk=%d num_bond[j]=%d\n",i,j,jk,Nj);
        if(bond_atom[j][jk] == i_tag) {found = 1; break; }
      }
      if (!found) error->all(FLERR,"FixMCAMeanStress::mean_stress_predict 'jk' not found");

      if (Nj > Nc) Nj = Nc; // not increse "rigidity" if atom has more bonds
      double rM_J = (double)(Nj-1) / (double)(Nc-1);
      int jtype = type[j];
      double rK1j = mca_pair->ModulusPredictOne[jtype][jtype];
      double rKnj = mca_pair->ModulusPredictAll[jtype][jtype];
      double rHj = rK1j*(1.0-rM_J) + rKnj*rM_J; // "rigidity" with accounting for # of bonds
      double delx,dely,delz,r,rsq,rinv;
      delx = xtmp - x[j][0] + dtImpl*(vxtmp - v[j][0]); // Implicit update of the distance. It is equivalent to use of damping force.
      dely = ytmp - x[j][1] + dtImpl*(vytmp - v[j][1]);
      delz = ztmp - x[j][2] + dtImpl*(vztmp - v[j][2]);
      rsq = delx*delx + dely*dely + delz*delz;
      r = sqrt(rsq);
      rinv = -1. / r; // "-" means that unit vector is from i1 to i2

      double *bond_hist_ik = bond_hist[i_tag-1][k];
      double r0 = bond_hist_ik[R_PREV];
      double pi = bond_hist_ik[P_PREV];
      double pj = bond_hist[tag[j]-1][jk][P_PREV];
      int bond_state = bondlist[bond_index[i][k]][3];
      if((bond_state == UNBONDED) && ((pi > 0.0)||(pj > 0.0))) continue; // 01.08.18 this is free surface
      if (newton_bond) {
        //pj = bond_hist_ik[PJ_PREV]; it means we store bonds only for i < j, but allocate memory for both. why?
        error->all(FLERR,"FixMCAMeanStress::mean_stress_predict does not support 'newton_bond on'");
      }

      double d_e0 = (r - r0) / mca_radius;
      double d_e  = (pj - pi + rHj*d_e0) / (rHi + rHj);
      rDeltaEpsMean_I += d_e;

      bond_hist_ik[R] = r;
//      bond_hist[j][jk][R] = r; //TODO do we need it for j?
      bond_hist_ik[NX] = delx*rinv; ///TODO will do it later in PairMCA::compute_total_force because here we use implicit distance
      bond_hist_ik[NY] = dely*rinv;
      bond_hist_ik[NZ] = delz*rinv;
    }
#ifdef NO_MEANSTRESS
    mean_stress[i] = 0.0;
#else
    double rNc = (double)Nc;
    if( Ni < Nc ) { // correcting mean strain accounting for free surface (surrounding)
       double rKHi = 1.0 - 2.0*mca_pair->G[itype][itype] / r3Ki;
       double rKk = -rKHi / (1.0 - rKHi);
       double rTMult = (double)(Nc - Ni);
       rNc -= rTMult * rKk;
       rDeltaEpsMean_I /= rNc; // contribution to mean strain
//       rTMult *= (1.0 - rKk);
//       rTMult = rThermoElasticPart_I * rTMult / (Ni + rTMult);
//       rDeltaEpsMean_I += rTMult; // contribution to thermoelastic part
    } else
      rDeltaEpsMean_I /= rNc;

    mean_stress[i] = r3Ki * (rDeltaEpsMean_I);// - rThermoElasticPart_I);
#endif
  }

#ifndef NO_MEANSTRESS
   // Second loop for computing mean stress
#if defined (_OPENMP)
#pragma omp parallel for private(i,j,k,jk,itype) shared(x,v,mean_stress,plastic_heat,bond_atom,bond_hist) default(shared) schedule(static)
#endif
  for (i = 0; i < nlocal; i++) {/// i < nmax; i++) {///
    if (num_bond[i] == 0) continue;

    int ** const bond_mca = atom->bond_mca;
    double rKHi,rKHj;// 1-2*G/(3*K) for atom i (j)
    double rHi,rHj;  // 2*G for atom i (j)
    double pi,pj;
    double d_p;
    double d_e,d_e0;
    double rdSgmi,rdSgmj;
    double r,r0;

    itype = type[i];
    ///AS TODO make this property global as in 'fix_check_timestep_gran.cpp' :
    /// Y = static_cast<FixPropertyGlobal*>(modify->find_fix_property("youngsModulus","property/global","peratomtype",max_type,0,style));
    rHi = 2. * mca_pair->G[itype][itype];
    rKHi = mca_pair->K[itype][itype]; rKHi = 1. - rHi / (3. * rKHi);

    rdSgmi = rKHi*mean_stress[i];
    double rSum = 0.0;
    for(k = 0; k < num_bond[i]; k++)
    {
      j = bond_mca[i][k];
      int bond_index_i = bond_index[i][k];
      if(bond_index_i >= nbondlist) {
        char str[512];
        sprintf(str,"bond_index[%d][%d]=%d > nbondlist(%d) at step " BIGINT_FORMAT,
                i,j,bond_index_i,nbondlist,update->ntimestep);
        error->one(FLERR,str);
      }
      int bond_state = bondlist[bond_index_i][3];
      if(bond_state == NOT_INTERACT) { // pair does not interact
        continue;
      }

      double *bond_hist_ik = &(bond_hist[tag[i]-1][k][0]);
      int found = 0;
      for(jk = 0; jk < num_bond[j]; jk++)
        if(bond_atom[j][jk] == tag[i]) {found = 1; break; }
      if (!found) error->all(FLERR,"FixMCAMeanStress::mean_stress_predict 'jk' not found");
      double *bond_hist_jk = &(bond_hist[tag[j]-1][jk][0]);

      int jtype = type[j];
      rHj = 2. * mca_pair->G[jtype][jtype];
      rKHj = mca_pair->K[jtype][jtype]; rKHj = 1. - rHj / (3. * rKHj);

      r = bond_hist_ik[R];
      r0 = bond_hist_ik[R_PREV];
      pi = bond_hist_ik[P_PREV];
      if((bond_state == UNBONDED) && (pi > 0.0)) pi = 0.0; // 01.08.18 this is free surface
      pj = bond_hist_jk[P_PREV];
      if((bond_state == UNBONDED) && (pj > 0.0)) pj = 0.0; // 01.08.18 this is free surface
      if (newton_bond) {
        //pj = bond_hist_ik[PJ_PREV];
        error->all(FLERR,"FixMCAMeanStress::mean_stress_predict does not support 'newton on'");
      }

      d_e0 = (r - r0) / mca_radius;
      rdSgmj = rKHj*mean_stress[j]; // here we use mean_stress[j] so we can not write to it
      d_e  = (pj - pi + rHj*d_e0 + rdSgmj - rdSgmi) / (rHi + rHj);
      d_p = rHi*d_e + rdSgmi;
      pi += d_p;
//if (logfile) fprintf(logfile,"FixMCAMeanStress::mean_stress_predict: i=%d j=%d P=%g oNbrR_i.rE=%g IDi=%d IDj=%d Dij=%g D0ij=%g\n   dE=%g Pj=%g Pi=%g dSgmj=%g dSgmi=%g Hj=%g Hi=%g meanSi=%g meanSj=%g bond_state=%d\n",
//i,j,pi,bond_hist_ik[E],tag[i],bond_atom[i][k],r,r0,d_e,pj,(pi-d_p),rdSgmj,rdSgmi,rHj,rHi,mean_stress[i],mean_stress[j],bond_state);
      if((bond_state == UNBONDED) && (pi > 0.0)) {  // 01.08.18 this is free surface if happens that unbonded particles attract each other
          pi = 0.0;
      }
      rSum += pi;
    }
    double rNc = (double)Nc;
    plastic_heat[i] = rSum / rNc; // save mean stress to plastic_heat temporarily
  }

#ifdef _OPENMP
#pragma omp parallel for private(i) shared(mean_stress,plastic_heat) default(shared) schedule(static)
#endif
  for (i = 0; i < nlocal; i++) {/// i < nmax; i++) {///
    mean_stress[i] = plastic_heat[i];
//if (logfile) fprintf(logfile,"FixMCAMeanStress::mean_stress_predict: i=%d meanSi=%g PREV  meanSi=%g \n",i,mean_stress[i],atom->mean_stress_prev[i]);
  }
#endif // NO_MEANSTRESS
}

void FixMCAMeanStress::pre_force(int vflag)
{
//if (logfile) fprintf(logfile, "FixMCAMeanStress::pre_force \n");///AS DEBUG
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
  const int * const tag = atom->tag;
  double ** const theta = atom->theta;
  double ** const theta_prev = atom->theta_prev;
  const double * const mean_stress = atom->mean_stress;
  const double * const mean_stress_prev = atom->mean_stress_prev;
//  const double * const equiv_stress = atom->equiv_stress;
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
//    buf[m++] = equiv_stress[j];
    buf[m++] = equiv_stress_prev[j];
    buf[m++] = equiv_strain[j];

    if(atom->n_bondhist) {
      int tag_j = tag[j] - 1;
      for (k = 0; k < num_bond[j]; k++)
        for (l = 0; l < MX; l++)
          buf[m++] = bond_hist[tag_j][k][l];
    }
  }
//if (logfile) fprintf(logfile,"FixMCAMeanStress::pack_comm m=%d n=%d [%d - %d]\n",m,n,list[0],list[n-1]);
  return comm_forward; //m; for last versions of lammps !!!
}

/* ---------------------------------------------------------------------- */

void FixMCAMeanStress::unpack_comm(int n, int first, double *buf)
{
  int i,m,last;
  int k,l;

  const int *num_bond = atom->num_bond;
  const int * const tag = atom->tag;
  double **theta = atom->theta;
  double **theta_prev = atom->theta_prev;
  double *mean_stress = atom->mean_stress;
  double *mean_stress_prev = atom->mean_stress_prev;
//  double *equiv_stress = atom->equiv_stress;
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
//    equiv_stress[i] = buf[m++];
    equiv_stress_prev[i] = buf[m++];
    equiv_strain[i] = buf[m++];

    if(atom->n_bondhist) {
      int tag_i = tag[i] - 1;
      for (k = 0; k < num_bond[i]; k++)
        for (l = 0; l < MX; l++)
          bond_hist[tag_i][k][l] = buf[m++];
    }
  }
//if (logfile) fprintf(logfile,"FixMCAMeanStress::unpack_comm m=%d n=%d [%d - %d]\n",m,n,first,last);
}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for exchange with another proc
------------------------------------------------------------------------- *

int FixMCAMeanStress::pack_exchange(int i, double *buf)
{
  int m,k,l;

  const int * const num_bond = atom->num_bond;
  const int * const tag = atom->tag;
  double ** const theta = atom->theta;
  double ** const theta_prev = atom->theta_prev;
  const double * const mean_stress = atom->mean_stress;
  const double * const mean_stress_prev = atom->mean_stress_prev;
  const double * const equiv_stress = atom->equiv_stress;
  const double * const equiv_stress_prev = atom->equiv_stress_prev;
  const double * const equiv_strain = atom->equiv_strain;
  double *** const bond_hist = atom->bond_hist;

    m = 0;
    buf[m++] = theta[i][0];
    buf[m++] = theta[i][1];
    buf[m++] = theta[i][2];
    buf[m++] = theta_prev[i][0];
    buf[m++] = theta_prev[i][1];
    buf[m++] = theta_prev[i][2];
    buf[m++] = mean_stress[i];
    buf[m++] = mean_stress_prev[i];
//    buf[m++] = equiv_stress[i];
    buf[m++] = equiv_stress_prev[i];
    buf[m++] = equiv_strain[i];

    if(atom->n_bondhist) {
      int tag_i = tag[i] - 1;
      for (k = 0; k < num_bond[i]; k++)
        for (l = 0; l < MX; l++)
          buf[m++] = bond_hist[tag_i][k][l];
    }

if (logfile) fprintf(logfile,"FixMCAMeanStress::pack_exchange m=%d i=%d \n",m,i);
    return m;
}

* ----------------------------------------------------------------------
   unpack values into local atom-based arrays after exchange
------------------------------------------------------------------------- *

int FixMCAMeanStress::unpack_exchange(int nlocal, double *buf)
{
  int k,l;
  const int *num_bond = atom->num_bond;
  const int * const tag = atom->tag;
  double **theta = atom->theta;
  double **theta_prev = atom->theta_prev;
  double *mean_stress = atom->mean_stress;
  double *mean_stress_prev = atom->mean_stress_prev;
  double *equiv_stress = atom->equiv_stress;
  double *equiv_stress_prev = atom->equiv_stress_prev;
  double *equiv_strain = atom->equiv_strain;
  double ***bond_hist = atom->bond_hist;

  int m = 0;
    theta[nlocal][0] = buf[m++];
    theta[nlocal][1] = buf[m++];
    theta[nlocal][2] = buf[m++];
    theta_prev[nlocal][0] = buf[m++];
    theta_prev[nlocal][1] = buf[m++];
    theta_prev[nlocal][2] = buf[m++];
    mean_stress[nlocal] = buf[m++];
    mean_stress_prev[nlocal] = buf[m++];
//    equiv_stress[nlocal] = buf[m++];
    equiv_stress_prev[nlocal] = buf[m++];
    equiv_strain[nlocal] = buf[m++];

    if(atom->n_bondhist) {
      int tag_i = tag[nlocal] - 1;
      for (k = 0; k < num_bond[nlocal]; k++)
        for (l = 0; l < MX; l++)
          bond_hist[tag_i][k][l] = buf[m++];
    }

if (logfile) fprintf(logfile,"FixMCAMeanStress::unpack_exchange m=%d nlocal=%d\n",m,nlocal);
    return m;
}
*/

