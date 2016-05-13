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

    Andreas Aigner (JKU Linz)

    Copyright 2009-2012 JKU Linz
------------------------------------------------------------------------- */

#include "math.h"
#include "mpi.h"
#include "string.h"
#include "stdlib.h"
#include "fix_mca_meanstress.h"
#include "update.h"
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

#include "pair_mca.cpp"


using namespace LAMMPS_NS;
using namespace FixConst;



/* ---------------------------------------------------------------------- */

FixMCAMeanStress::FixMCAMeanStress(LAMMPS *lmp, int narg, char **arg) :
  FixSph(lmp, narg, arg)
{
  int iarg = 0;

  if (iarg+3 > narg) error->fix_error(FLERR,this,"Not enough input arguments");

  iarg += 3;

  while (iarg < narg) {
    // kernel style
    if (strcmp(arg[iarg],"sphkernel") == 0) {
          if (iarg+2 > narg) error->fix_error(FLERR,this,"Illegal use of keyword 'sphkernel'. Not enough input arguments");

          if(kernel_style) delete []kernel_style;
          kernel_style = new char[strlen(arg[iarg+1])+1];
          strcpy(kernel_style,arg[iarg+1]);

          // check uniqueness of kernel IDs

          int flag = SPH_KERNEL_NS::sph_kernels_unique_id();
          if(flag < 0) error->fix_error(FLERR,this,"Cannot proceed, sph kernels need unique IDs, check all sph_kernel_* files");

          // get kernel id

          kernel_id = SPH_KERNEL_NS::sph_kernel_id(kernel_style);
          if(kernel_id < 0) error->fix_error(FLERR,this,"Unknown sph kernel");

          iarg += 2;

    } else error->fix_error(FLERR,this,"Wrong keyword.");
  }
}

/* ---------------------------------------------------------------------- */

FixMCAMeanStress::~FixMCAMeanStress()
{


}

/* ---------------------------------------------------------------------- */

inline void  FixMCAMeanStress::swap_prev()
{
  int i,k;

  int *num_bond = atom->num_bond;
  double ***bond_hist = atom->bond_hist;
  int nlocal = atom->nlocal;

  for (i = 0; i < nlocal; i++) {
    double *tmp;
    tmp = atom->mean_stress;
    atom->mean_stress = atom->mean_stress_prev;
    atom->mean_stress_prev = tmp;

    tmp = atom->equiv_stress;
    atom->equiv_stress = atom->equiv_stress_prev;
    atom->equiv_stress_prev = tmp;

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
  int i,j,k,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,vxtmp,vytmp,vztmp,r,r0,rsq,rinv;

  int jk;

  double **x = atom->x;
  double **v = atom->v;
  double *mean_stress_prev = atom->mean_stress_prev;
  const double mca_radius = atom->mca_radius;
  int *tag = atom->tag;
  int *type = atom->type;

  int **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;
  int newton_bond = force->newton_bond;
  double ***bond_hist = atom->bond_hist;
  const double dtImpl = IMPLFACTOR*update->dt; ///AS it is equivalent to damping force. TODO set in coeffs some value 0.0..1 instead of 0.5

  int Nc = atom->coord_num;
  int nlocal = atom->nlocal;
  PairMCA *mca_pair = (PairMCA*) force->pair;

  for (i = 0; i < nlocal; i++) {
    if (num_bond[i] == 0) continue;

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

      double *bond_hist_ik = bond_hist[i][k];
      int found = 0;
      for(jk = 0; jk < num_bond[j]; jk++)
        if(bond_atom[j][jk] == tag[i]) {found = 1; break; }
      if (!found) error->all(FLERR,"PairMCA::mean_stress_predict 'jk' not found");

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

      r0 = bond_hist_ik[R_PREV];
      pi = bond_hist_ik[P_PREV];
      if (newton_bond) {
        //pj = bond_hist_ik[PJ_PREV]; it means we store bonds only for i < j, but allocate memory for both. why?
        error->all(FLERR,"PairMCA::mean_stress_predict does not support 'newton_bond on'");
      } else
        pj = bond_hist[j][jk][P_PREV];

      d_e0 = (r - r0) / mca_radius;
      d_e  = (pj - pi + rHj*d_e0) / (rHi + rHj);
      d_p = rHi*d_e;

      rdSgmi += d_p;
      bond_hist_ik[R] = r;
//      bond_hist[j][jk][R] = r; //TODO do we need it for j?
      bond_hist_ik[NX] = delx*rinv; ///TODO will do it later in BondMCA::compute_total_force because here we use implicit distance
      bond_hist_ik[NY] = dely*rinv;
      bond_hist_ik[NZ] = delz*rinv;
    }
#ifndef NO_MEANSTRESS
    mean_stress_prev[i] += rdSgmi / Nc; // optimized if use _prev as current here and in compute_elastic_force()
    //mean_stress[i] = mean_stress_prev[i] + rdSgmi / Nc;
#endif
  }
}



/* ---------------------------------------------------------------------- */



int FixMCAMeanStress::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= POST_INTEGRATE_RESPA;
  return mask;
}




/* ---------------------------------------------------------------------- */

void FixMCAMeanStress::init()
{
  FixSph::init();

  // check if there is an sph/pressure fix present
  // must come before me, because
  // a - it needs the rho for the pressure calculation
  // b - i have to do the forward comm to have updated ghost properties

  int pres = -1;
  int me = -1;
  for(int i = 0; i < modify->nfix; i++)
  {
    if(strcmp("sph/density/summation",modify->fix[i]->style)) {
      me = i;
    }
    if(strncmp("sph/pressure",modify->fix[i]->style,12) == 0) {
      pres = i;
      break;
    }
  }

  if(me == -1 && pres >= 0) error->fix_error(FLERR,this,"Fix sph/pressure has to be defined after sph/density/summation \n");
  if(pres == -1) error->fix_error(FLERR,this,"Requires to define a fix sph/pressure also \n");
}


/* ---------------------------------------------------------------------- */



void FixMCAMeanStress::post_integrate()
{
  //template function for using per atom or per atomtype smoothing length
  if (mass_type) post_integrate_eval<1>();
  else post_integrate_eval<0>();

  swap_prev();
  predict_mean_stress();
 
}

/* ---------------------------------------------------------------------- */

template <int MASSFLAG>
void FixMCAMeanStress::post_integrate_eval()
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq,r,s=0.0,W;
  double sli,sliInv,slj,slCom,slComInv,cut,imass,jmass;
  int *ilist,*jlist,*numneigh,**firstneigh;

  double **x = atom->x;
  int *mask = atom->mask;
  double *rho = atom->rho;
  int newton_pair = force->newton_pair;

  int *type = atom->type;
  double *mass = atom->mass;
  double *rmass = atom->rmass;

  updatePtrs(); // get sl

  // reset and add rho contribution of self

  int nlocal = atom->nlocal;
  for (i = 0; i < nlocal; i++) {
    if (MASSFLAG) {
      itype = type[i];
      sli = sl[itype-1];
      imass = mass[itype];
    } else {
      sli = sl[i];
      imass = rmass[i];
    }

    sliInv = 1./sli;

    // this gets a value for W at self, perform error check

    W = SPH_KERNEL_NS::sph_kernel(kernel_id,0.,sli,sliInv);
    if (W < 0.)
    {
      fprintf(screen,"s = %f, W = %f\n",s,W);
      error->one(FLERR,"Illegal kernel used, W < 0");
    }

    // add contribution of self
    rho[i] = W * imass;
  }

  // need updated ghost positions and self contributions
  timer->stamp();
  comm->forward_comm();

  timer->stamp(TIME_COMM);

  // loop over neighbors of my atoms

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];

    if (!(mask[i] & groupbit)) continue;
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    if (MASSFLAG) {
      itype = type[i];
      imass = mass[itype];
    } else {
      imass = rmass[i];
      sli = sl[i];
    }

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];

      if (!(mask[j] & groupbit)) continue;

      if (MASSFLAG) {
        jtype = type[j];
        jmass = mass[jtype];
        slCom = slComType[itype][jtype];
      } else {
        jmass = rmass[j];
        slj = sl[j];
        slCom = interpDist(sli,slj);
      }

      slComInv = 1./slCom;
      cut = slCom*kernel_cut;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;

      if (rsq >= cut*cut) continue;
      // calculate distance and normalized distance

      r = sqrt(rsq);
      slComInv = 1./slCom;
      s = r*slComInv;

      // this gets a value for W at self, perform error check

      W = SPH_KERNEL_NS::sph_kernel(kernel_id,s,slCom,slComInv);
      if (W < 0.)
      {
        fprintf(screen,"s = %f, W = %f\n",s,W);
        error->one(FLERR,"Illegal kernel used, W < 0");
      }

      // add contribution of neighbor
      // have a half neigh list, so do it for both if necessary

      rho[i] += W * jmass;

      if (newton_pair || j < nlocal)
        rho[j] += W * imass;
    }
  }

  // rho is now correct, send to ghosts
  timer->stamp();
  comm->forward_comm();
  timer->stamp(TIME_COMM);

}


