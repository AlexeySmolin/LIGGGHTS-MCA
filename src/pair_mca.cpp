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
#include "domain.h"
#include "neigh_list.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include "neighbor.h"
#include "bond_mca.h"
#include "atom_vec_mca.h"
#include "vector_liggghts.h"
#include "rotations_mca.h"


using namespace LAMMPS_NS;
using namespace MathConst;
using namespace MCAAtomConst;

/// see also in 'fix_mca_meanstress.cpp'
///#define NO_MEANSTRESS // see also in fix_mca_meanstress.cpp
///#define NO_ROTATIONS
#define REAL_NULL_CONST 5.0E-22 // 5.0E-14

/* ---------------------------------------------------------------------- */

PairMCA::PairMCA(LAMMPS *lmp) : Pair(lmp)
{
  writedata = 1;
  single_enable = 0;
  Sy = NULL;
  Eh = NULL;
}

/* ---------------------------------------------------------------------- */

PairMCA::~PairMCA()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cof);
    memory->destroy(G);
    memory->destroy(K);
    memory->destroy(ModulusPredictOne);
    memory->destroy(ModulusPredictAll);
    memory->destroy(Sy);
    memory->destroy(Eh);
  }
}

/* ---------------------------------------------------------------------- */

inline void  PairMCA::compute_elastic_force()
{
  int i,j,k,jk,itype,jtype;
///  double xtmp,ytmp,ztmp,delx,dely,delz,r,r0;
///  double rsq,rinv;
///  double vxtmp,vytmp,vztmp;

  double **x = atom->x;
  double **v = atom->v;
  double **theta = atom->theta;
  double **theta_prev = atom->theta_prev;
  double **omega = atom->omega;
  const double * const mean_stress = atom->mean_stress;
  const double * const mean_stress_prev = atom->mean_stress_prev;
  const double mca_radius = atom->mca_radius;
  const int * const tag = atom->tag;
  const int * const type = atom->type;

  int ** const bond_atom = atom->bond_atom;
  const int * const num_bond = atom->num_bond;
  const int newton_bond = force->newton_bond;
  double ***bond_hist = atom->bond_hist;
  const int nbondlist = neighbor->nbondlist;
  int ** const bondlist = neighbor->bondlist;
  int ** const bond_index = atom->bond_index;
  double * const cont_distance = atom->cont_distance;
  const double dtImpl = (atom->implicit_factor) * update->dt;

  const int nlocal = atom->nlocal;
///  const int nmax = atom->nmax;
  const PairMCA * const mca_pair = (PairMCA*) force->pair;

//fprintf(logfile,"PairMCA::compute_elastic_force\n"); ///AS DEBUG TRACE

#if defined (_OPENMP)
#pragma omp parallel for private(i,j,k,jk,itype,jtype) shared(x,v,omega,theta,theta_prev,bond_hist) default(shared) schedule(static)
#endif
  for (i = 0; i < nlocal; i++) {/// i < nmax; i++) {///
    if (num_bond[i] == 0) continue;

    int ** const bond_mca = atom->bond_mca;
    double rKHi,rKHj;// 1-2*G/(3*K) for atom i (j)
    double rHi,rHj;  // 2*G for atom i (j)
    double pi,pj;
    double d_p;
    double ei,d_e,d_e0;
    double rdSgmi,rdSgmj;
    double rGi,rGj;
    double qi,qj;
    double r,r0;

    itype = type[i];
    ///AS TODO make this property global as in 'fix_check_timestep_gran.cpp' :
    /// Y = static_cast<FixPropertyGlobal*>(modify->find_fix_property("youngsModulus","property/global","peratomtype",max_type,0,style));
    rGi = mca_pair->G[itype][itype];
    rHi = 2. * rGi;
    rKHi = mca_pair->K[itype][itype]; rKHi = 1. - rHi / (3. * rKHi);

    rdSgmi = rKHi*(mean_stress[i] - mean_stress_prev[i]);
    for(k = 0; k < num_bond[i]; k++)
    {
      j = bond_mca[i][k];
/*      j = atom->map(bond_atom[i][k]);
      if (j == -1) {
        char str[512];
        sprintf(str,"Bond atoms %d %d missing at step " BIGINT_FORMAT,
                tag[i],bond_atom[i][k],update->ntimestep);
        error->one(FLERR,str);
      }
      j = domain->closest_image(i,j);
///if(tag[i]==10) fprintf(logfile,"PairMCA::compute_elastic_force bond_atom[%d][%d]=%d map()=%d \n",i,k,bond_atom[i][k],j);
*/
      int bond_index_i = bond_index[i][k];
      if(bond_index_i >= nbondlist) {
        char str[512];
        sprintf(str,"bond_index[%d][%d]=%d > nbondlist(%d) at step " BIGINT_FORMAT,
                i,j,bond_index_i,nbondlist,update->ntimestep);
        error->one(FLERR,str);
      }
      int bond_state = bondlist[bond_index_i][3];
      if(bond_state == NOT_INTERACT) { // pair does not interact
//        fprintf(logfile,"PairMCA::compute_elastic_force bond %d (%d - %d) does not interact\n",bond_index_i,i,j);
        continue;
      }
//fprintf(logfile,"PairMCA::compute_elastic_force bond %d (%d - %d(k=%d)) has state %d\n",bond_index_i,i,j,k,bond_state);

      double *bond_hist_ik = &(bond_hist[tag[i]-1][k][0]);
      int found = 0;
      for(jk = 0; jk < num_bond[j]; jk++)
        if(bond_atom[j][jk] == tag[i]) {found = 1; break; }
      if (!found) error->all(FLERR,"PairMCA::compute_elastic_force 'jk' not found");
      double *bond_hist_jk = &(bond_hist[tag[j]-1][jk][0]);

      jtype = type[j];
      rGj = mca_pair->G[jtype][jtype];
      rHj = 2. * rGj;
      rKHj = mca_pair->K[jtype][jtype]; rKHj = 1. - rHj / (3. * rKHj);

      r = bond_hist_ik[R];
      r0 = bond_hist_ik[R_PREV];
      ei = bond_hist_ik[E];
      pi = bond_hist_ik[P_PREV];
      if (newton_bond) {
        //pj = bond_hist_ik[PJ_PREV];
        error->all(FLERR,"PairMCA::compute_elastic_force does not support 'newton on'");
      } else
        pj = bond_hist_jk[P_PREV];

      /// BEGIN central force
      d_e0 = (r - r0) / mca_radius;
      rdSgmj = rKHj*(mean_stress[j] - mean_stress_prev[j]);
      d_e  = (pj - pi + rHj*d_e0 + rdSgmj - rdSgmi) / (rHi + rHj);
      d_p = rHi*d_e + rdSgmi;
      ei += d_e;
      pi += d_p;
//fprintf(logfile,"PairMCA::compute_elastic_force: i=%d j=%d E=%g P=%g oNbrR_i.rE=%g IDi=%d IDj=%d Dij=%g D0ij=%g\n   dE=%g Pj=%g Pi=%g dSgmj=%g dSgmi=%g Hj=%g Hi=%g meanSi=%g meanSj=%g bond_state=%d\n",
//i,j,ei,pi,bond_hist_ik[E],tag[i],bond_atom[i][k],r,r0,d_e,pj,(pi-d_p),rdSgmj,rdSgmi,rHj,rHi,mean_stress[i],mean_stress[j],bond_state);
      if((mca_radius*(1.0 + ei)) > r) {
        if((bond_state == BONDED) || (pi <= 0.0)) {
          fprintf(logfile,"PairMCA::compute_elastic_force: 'Qij>Dij' E=%g oNbrR_i.rE=%g IDi=%d IDj=%d Dij=%g D0ij=%g\n   dE=%g Pj=%g Pi=%g dSgmj=%g dSgmi=%g Hj=%g Hi=%g bond_state=%d\n",
          ei,bond_hist_ik[E],tag[i],bond_atom[i][k],r,r0,d_e,pj,(pi-d_p),rdSgmj,rdSgmi,rHj,rHi,bond_state);
          continue;
        }
      }
///fprintf(logfile,"PairMCA::compute_elastic_force: E=%g P=%g oNbrR_i.rE=%g IDi=%d IDj=%d Dij=%g D0ij=%g\n   dE=%g Pj=%g Pi=%g dSgmj=%g dSgmi=%g Hj=%g Hi=%g meanSi=%g meanSj=%g bond_state=%d\n",
///ei,pi,bond_hist_ik[E],tag[i],bond_atom[i][k],r,r0,d_e,pj,(pi-d_p),rdSgmj,rdSgmi,rHj,rHi,mean_stress[i],mean_stress[j],bond_state);
      /// END central force

      double vShear[3], vSij[3], vYi[3], vMij[3];
      if((bond_state == UNBONDED) && (pi > 0.0)) { // if happens that unbonded particles attract each other
///        fprintf(logfile,"PairMCA::compute_elastic_force bond %d (%d - %d) is broken but attracts\n",bond_index_i,i,j);
        double rCDsum = cont_distance[i] + cont_distance[j];
        if (rCDsum > r) {
//fprintf(stderr,"CMCA3D_TEPModel::ElasticForce(): '(!oNbrR_i.bLinked) && (Pi>0.0)' IDi=%d IDj=%d CDsum=%g Dij=%g oAtR_i.iNCount=%d oAtR_j.iNCount=%d\n",oAtL_i.lID,oAtL_j.lID,rCDsum,rDij,oAtR_i.iNCount,oAtR_j.iNCount);
//fprintf(stderr,"Qi=%g Qj=%g Pi=%g Pj=%g dP=%g dSgmj=%g dSgmi=%g D0ij=%g dE0=%g dE=%g\n",
//rF_AutomataRadius*(1.+rE),rF_AutomataRadius*(1.+(oAtR_j.aNeighbors[kj].rE+rdE0-rdE)),
//(rPi-rdP),rPj,rdP,rdSgmj,rdSgmi,rD0ij,rdE0,rdE);
          ;// leave it as it is
        } else {
//fprintf(stderr,"CMCA3D_TEPModel::ElasticForce(): '(!oNbrR_i.bLinked) && (rPi>0.0)'' IDi=%d IDj=%d rCDsum=%g Dij=%g oAtR_i.iNCount=%d oAtR_j.iNCount=%d\n",oAtL_i.lID,oAtL_j.lID,rCDsum,rDij,oAtR_i.iNCount,oAtR_j.iNCount);
//fprintf(stderr,"Qi=%g Qj=%g Pi=%g Pj=%g dP=%g dSgmj=%g dSgmi=%g D0ij=%g dE0=%g dE=%g CD=%g new E=%g\n",
//rF_AutomataRadius*(1.+rE),rF_AutomataRadius*(1.+(oAtR_j.aNeighbors[kj].rE+rdE0-rdE)),
//(rPi-rdP),rPj,rdP,rdSgmj,rdSgmi,rD0ij,rdE0,rdE,oAtR_i.rCD,((oAtR_i.rCD - rF_AutomataRadius) / rF_AutomataRadius));
//fprintf(stderr,"OwnNode=%d iNode=%d jNode=%d \n", OwnNode, oAtL_i.iNode, oAtL_j.iNode);
          ei  = (cont_distance[i] - mca_radius) / mca_radius;
        }
        pi = 0.0;
        vSij[0]= vSij[1]= vSij[0]= 0.0;
        vYi[0]= vYi[1]= vYi[0]= 0.0;
        vMij[0]= vMij[1]= vMij[0]= 0.0;
      } else {
        /// BEGIN shear force

        double *nv = &(bond_hist_ik[NX]);       // normal unit vector
        double *nv0 = &(bond_hist_ik[NX_PREV]); // normal unit vector at prev time step
        double dYij[3];
        vectorCross3D(nv0, nv, dYij); // rotaion of the pair
///if(i == 13) fprintf(logfile,"PairMCA::compute_elastic_force i=%d j=%d nv0= %20.12e %20.12e %20.12e nv= %20.12e %20.12e %20.12e \n",i,j,nv0[0],nv0[1],nv0[2],nv[0],nv[1],nv[2]); ///AS DEBUG
///if(i == 13) fprintf(logfile,"PairMCA::compute_elastic_force i=%d j=%d nv0xnv= %20.12e %20.12e %20.12e \n",i,j,dYij[0],dYij[1],dYij[2]); ///AS DEBUG

        double vdLij[3];
        vectorCopy3D(dYij, vdLij);
        vectorScalarMult3D(vdLij, r); // tangent displacement of the pair

        qi = mca_radius * (1. + bond_hist_ik[E]); // distance to the contact point from i
        qj = mca_radius * (1. + bond_hist_jk[E]); // distance to the contact point from j
        double vdTHi[3], vdTHj[3];
        double vR1[3], vR2[3], vRsum[3];
#ifdef NO_ROTATIONS
        vdTHi[0]=vdTHi[2]=vdTHi[2]=vdTHj[0]=vdTHj[1]=vdTHj[2]= 0.0;
#else
        vectorCopy3D(&(theta[i][0]), vR1);
        vectorCopy3D(&(theta_prev[i][0]), vR2);
        vectorFlip3D(vR2); //vectorScalarMult3D(vR2, -1.)
        SmallRotationSum(vR1, vR2, vdTHi); // rotation increment for i
        vectorCopy3D(&(theta[j][0]), vR1);
        vectorCopy3D(&(theta_prev[j][0]), vR2);
        vectorFlip3D(vR2); //vectorScalarMult3D(vR2, -1.)
        SmallRotationSum(vR1, vR2, vdTHj); // rotation increment for j
// implicit vvvv
        vectorCopy3D(&(omega[i][0]), vR2);
        vectorScalarMult3D(vR2, dtImpl);
        vectorCopy3D(vdTHi, vR1);
        SmallRotationSum(vR1, vR2, vdTHi);
        vectorCopy3D(&(omega[j][0]), vR2);
        vectorScalarMult3D(vR2, dtImpl);
        vectorCopy3D(vdTHj, vR1);
        SmallRotationSum(vR1, vR2, vdTHj);
// inmplicit ^^^^
        vectorCopy3D(vdTHi, vR1); vectorScalarMult3D(vR1, qi);
        vectorCopy3D(vdTHj, vR2); vectorScalarMult3D(vR2, qj);
        SmallRotationSum(vR1, vR2, vRsum);

        vectorCopy3D(nv, vR1);
        double proj = vectorDot3D(vRsum, nv); // projection of rotational part to the normal
        vectorScalarMult3D(vR1, proj);
        vectorSubtract3D(vRsum, vR1, vR2);    // tangential component of rotation
        vectorCopy3D(vdLij, vR1);
        vectorFlip3D(vR2);
        SmallRotationSum(vR1, vR2, vdLij);
#endif

        double rKS = 1. / (qj*rGi + qi*rGj);
        double rQR = qi * 0.5 * rKS;
        double vYj[3], vYij[3];
        vectorCopy3D(&(bond_hist_ik[SHX_PREV]), vShear);
        vectorCopy3D(&(bond_hist_ik[YX_PREV]), vYi);
        vectorCopy3D(&(bond_hist_jk[YX_PREV]), vYj);
        vectorCopy3D(vYj, vR1); vectorScalarMult3D(vR1, rQR);
        vectorCopy3D(vYi, vR2); vectorScalarMult3D(vR2, -rQR);
        vectorAdd3D(vR1, vR2, vYij);
        vectorScalarMult3D(vdLij, -rGj*rKS);
        SmallRotationSum(vdLij, vYij, dYij);
        vectorCopy3D(vShear, vR1);
        RotationSum(vR1, dYij, vShear);
        vectorScalarMult3D(vYi, 1./rHi);
        vectorCopy3D(vYi, vR1);
        RotationSum(vR1, dYij, vYi);
        vectorScalarMult3D(vYi, rHi);

        double rFDij=1.0; // dry friction factor
        if(bond_state == UNBONDED) { // unlinked
          /// BEGIN correction due to sliding friction
          //int m1=oAtR_i.iMaterialID;
          //int m2=oAtR_j.iMaterialID;
          //const T_PairMaterial &oPair = aF_Pairs[m1][m2];
/* TODO   */
          double rCOF = mca_pair->cof[itype][jtype];
          if(rCOF <  REAL_NULL_CONST) {
            rFDij = 0.0;
            vShear[0]= vShear[1]= vShear[0]= 0.0;
            vSij[0]= vSij[1]= vSij[0]= 0.0;
            vYi[0]= vYi[1]= vYi[0]= 0.0;
            vMij[0]= vMij[1]= vMij[0]= 0.0;
          } else {
            double rSij;
            double rFDij; // dry friction
            if(pi > 0.0) rFDij = 0.0;  // just in case
            else rFDij = fabs(rCOF*pi);
            rSij = vectorMag3D(vYi);
            if(rSij > rFDij) {
              rFDij /= rSij;
              vectorScalarMult3D(vYi, rFDij);
            } else rFDij = 1.0;
            vectorCross3D(vYi, nv, vSij);
          }
          /// END correction due to sliding friction
        } else {
          vectorCross3D(vYi, nv, vSij);
        }
        /// END shear force

///        if(bond_state == BONDED) { // linked
        /// BEGIN bending-torsion torque
        double vdMij[3], vdMji[3];
#ifdef NO_ROTATIONS
vMij[0]= vMij[1]= vMij[2]= 0.;
#else
        vectorCopy3D(vdTHi, vdMij);
        vectorCopy3D(vdTHj, vdMji);
        vectorScalarMult3D(vdMij, qi);
        vectorScalarMult3D(vdMji, -qj);
        RotationSum(vdMij, vdMji, vMij);
        double rGmult = rGi*rGj/(rGi+rGj);
        vectorScalarMult3D(vMij, rGmult);

        vectorCopy3D(&(bond_hist_ik[MX]), vdMji);
        vectorScalarMult3D(vdMji, rKS);
        vectorCopy3D(vMij, vdMij);
        vectorScalarMult3D(vdMij, rKS);
        RotationSum(vdMij, vdMji, vMij);
        rGmult = qi*rGi + qj*rGj;
        if(rFDij != 1.0) rGmult *= rFDij;
        vectorScalarMult3D(vMij, rGmult);
///        }
#endif
        /// END bending-torsion torque
      } // end of else if Unlinked and P>0
      bond_hist_ik[E] = ei;
      bond_hist_ik[P] = pi;
      bond_hist_ik[SHX] = vShear[0];
      bond_hist_ik[SHY] = vShear[1];
      bond_hist_ik[SHZ] = vShear[2];
      bond_hist_ik[YX] = vYi[0];
      bond_hist_ik[YY] = vYi[1];
      bond_hist_ik[YZ] = vYi[2];
      bond_hist_ik[SX] = vSij[0];
      bond_hist_ik[SY] = vSij[1];
      bond_hist_ik[SZ] = vSij[2];
///      if(bond_state == BONDED) {
        bond_hist_ik[MX] = vMij[0];
        bond_hist_ik[MY] = vMij[1];
        bond_hist_ik[MZ] = vMij[2];
///      }
    }
  }
}

/* ---------------------------------------------------------------------- */

inline void  PairMCA::compute_equiv_stress()
{
  int i,k;

  int ** const bond_atom = atom->bond_atom;
  const int * const num_bond = atom->num_bond;
  double *** const bond_hist = atom->bond_hist;
  int ** const bondlist = neighbor->bondlist;
  int ** const bond_index = atom->bond_index;
  const int * const type = atom->type;
  const PairMCA * const mca_pair = (PairMCA*) force->pair;

  const int Nc = atom->coord_num;
  const int nlocal = atom->nlocal;
///  const int nmax = atom->nmax;
  const double mca_radius = atom->mca_radius;

//fprintf(logfile,"PairMCA::compute_equiv_stress\n"); ///AS DEBUG TRACE
#if defined (_OPENMP)
#pragma omp parallel for private(i,k) default(shared) schedule(static)
#endif
  for (i = 0; i < nlocal; i++) {/// i < nmax; i++) {///
    int numb = num_bond[i];
    if (numb == 0) continue;

    const int * const tag = atom->tag;
    const double * const theta = &(atom->theta[i][0]);
    double * const theta_prev = &(atom->theta_prev[i][0]);
    double ** const bond_hist_i = &(bond_hist[tag[i]-1][0]);
    double *mean_stress = atom->mean_stress;
    double *cont_distance = atom->cont_distance;
    double meanstr = 0.0;

    theta_prev[0] = theta[0]; // we need to swap 'theta' here, after using in 'compute_elastic_force()'
    theta_prev[1] = theta[1]; // but not before as for other '_prev' values
    theta_prev[2] = theta[2];

    double rdSumP = 0.0;
    double rdSumE = 0.0;
#ifndef NO_MEANSTRESS
    for(k = 0; k < numb; k++) {
      int bond_index_i = bond_index[i][k];
      int bond_state = bondlist[bond_index_i][3];
      if (bond_state == NOT_INTERACT) continue;

      double * const bond_hist_ik = &(bond_hist_i[k][0]);
      rdSumE += bond_hist_ik[E];
      double p = bond_hist_ik[P];
      if ((bond_state == UNBONDED) && (p > 0.0)) continue;
      rdSumP += p;
    }
    meanstr = rdSumP / Nc;
#endif
    double *equiv_stress = atom->equiv_stress;
    double rSumEqStr = 0.0;
    int iNC = 0;
    for(k = 0; k < numb; k++)
    {
      int bond_index_i = bond_index[i][k];
      int bond_state = bondlist[bond_index_i][3];
      if (bond_state == NOT_INTERACT) continue;

      double * const bond_hist_ik = &(bond_hist_i[k][0]);
      double p = bond_hist_ik[P];
      if ((bond_state == UNBONDED) && (p > 0.0)) continue;

      double xtmp,ytmp,ztmp,rEqStress;
      rEqStress = p - meanstr;
      rEqStress = rEqStress*rEqStress;
      xtmp = bond_hist_ik[SX];
      ytmp = bond_hist_ik[SY];
      ztmp = bond_hist_ik[SZ];
      rEqStress += xtmp*xtmp + ytmp*ytmp + ztmp*ztmp;
      rSumEqStr += rEqStress;
      iNC++;
    }
    int iFreeSlots = Nc > iNC ? (Nc - iNC) : 0;
    if (iNC < Nc) {
      rSumEqStr += iFreeSlots * meanstr*meanstr;
    }
    equiv_stress[i] = sqrt(4.5*(rSumEqStr) / Nc);;
    int itype = type[i];
    double rKi = 3.0 * mca_pair->K[itype][itype];
    double rE = (rdSumP / rKi - rdSumE); // predictor of the contact distance accounting of plastic strain
    if (iFreeSlots > 0) rE /= iFreeSlots;
    cont_distance[i] = mca_radius * (1.0 + rE);
    if(rE < -1.0) {
fprintf(stderr,"iFreeSlots=%d (iNC=%d) rdSumP=%g rdSumE=%g rE=%g\n", iFreeSlots,iNC,rdSumP,rdSumE,rE);
        char str[512];
        sprintf(str,"Wrong contact distance for atom# %d at step " BIGINT_FORMAT,
                i,update->ntimestep);
        error->one(FLERR,str);
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairMCA::correct_for_plasticity()
{
//fprintf(logfile, "PairMCA::correct_for_plasticity \n"); ///AS DEBUG TRACE
  int i,k;
  const int nlocal = atom->nlocal;
///  const int nmax = atom->nmax;
  const double mca_radius = atom->mca_radius;
  double ***bond_hist = atom->bond_hist;
  int ** const bondlist = neighbor->bondlist;
  int ** const bond_index = atom->bond_index;

#if defined (_OPENMP)
#pragma omp parallel for private(i,k) shared(bond_hist) default(shared) schedule(static)
#endif
  for (i = 0; i < nlocal; i++) {/// i < nmax; i++) {///
    const int * const num_bond = atom->num_bond;
    if (num_bond[i] == 0) continue;

    const int newton_bond = force->newton_bond;
    const int * const type = atom->type;
    const int * const tag = atom->tag;
    int ** const bond_atom = atom->bond_atom;
    const PairMCA * const mca_pair = (PairMCA*) force->pair;
    const double * const mean_stress = atom->mean_stress;
    const double * const equiv_stress_prev = atom->equiv_stress_prev;


    double **bond_hist_i = &(bond_hist[tag[i]-1][0]);
    double *equiv_stress = atom->equiv_stress;
    double *equiv_strain = atom->equiv_strain;
    double *plastic_heat = atom->plastic_heat;
    int itype = type[i];
    double rGi = mca_pair->G[itype][itype];
    double r3Gi = 3.0*rGi;
    double rSgmInt = equiv_stress[i];
    double rdSgm = rSgmInt - equiv_stress_prev[i];
    double rSR = equiv_strain[i] + rdSgm/r3Gi;
    double rSyi =  mca_pair->Sy[itype][itype];
    double rSR_Pla = rSyi > 0. ? rSyi/r3Gi : 10.0*rSR; // equivalent yeild strain, if no plasticity make rSR_Pla > rSR
    equiv_strain[i] = rSR;

    if((rSyi>0.)&&(rSgmInt>REAL_NULL_CONST)&&(rdSgm>0.0)&&(rSR>rSR_Pla)) {
      double rEhi =  mca_pair->Eh[itype][itype];
//fprintf(logfile,"PairMCA::correct_for_plasticity rEhi[%d][%d]= %g \n",itype,itype,rEhi);
      double rSgmPl = rSyi + (rSR - rSR_Pla) * rEhi; // + work harderning
//fprintf(logfile,"PairMCA::correct_for_plasticity rSR= %g rSR_Pla= %g rSgmPl=%g\n",rSR,rSR_Pla,rSgmPl);
      double rM = rSgmPl/rSgmInt; // radial return factor
      double rMpli = 1.0 - rM;
      if(rMpli > REAL_NULL_CONST) {
        equiv_stress[i] = rSgmPl; // set according to the yeild surface
        double rP = (rSyi*rSyi*0.5/r3Gi + (rSyi + rSgmPl)*0.5*(rSR - rSR_Pla)) - rSgmPl*rSgmPl*0.5/r3Gi; ///TODO plastic heat: now for bilinear harderning only
        if (rP > 0.0) plastic_heat[i] = rP; // store plastic heat
        for(k = 0; k < num_bond[i]; k++) {
/*          int j = atom->map(bond_atom[i][k]);
          if (j == -1) {
            char str[512];
            sprintf(str,
                "Bond atoms %d %d missing at step " BIGINT_FORMAT,
                tag[i],bond_atom[i][k],update->ntimestep);
            error->one(FLERR,str);
          }
          j = domain->closest_image(i,j);
///if(tag[i]==10) fprintf(logfile,"PairMCA::correct_for_plasticity bond_atom[%d][%d]=%d map()=%d \n",i,k,bond_atom[i][k],j);
*/
          int bond_index_i = bond_index[i][k];
          int bond_state = bondlist[bond_index_i][3];
          if (bond_state == NOT_INTERACT) continue;
          else
          {
            double *bond_hist_ik = &(bond_hist_i[k][0]);
/*            int found = 0;
            int jk;
            for(jk = 0; jk < num_bond[j]; jk++)
              if(bond_atom[j][jk] == tag[i]) {found = 1; break; }
            if (!found) error->all(FLERR,"PairMCA::correct_for_plasticity 'jk' not found");
            double *bond_hist_jk = &(bond_hist[tag[j]-1][jk][0]);
*/
            rP = bond_hist_ik[P];
            rP = rP * rM + mean_stress[i] * rMpli;
            double rMij = rM;
            if ((rP > 0.0) && (bond_state == UNBONDED)) {
              rP = 0.0;
              rMij = 0.0;
            }
            bond_hist_ik[P] = rP;
            bond_hist_ik[SX] *= rMij;
            bond_hist_ik[SY] *= rMij;
            bond_hist_ik[SZ] *= rMij;
            bond_hist_ik[YX] *= rMij;
            bond_hist_ik[YY] *= rMij;
            bond_hist_ik[YZ] *= rMij;
            bond_hist_ik[MX] *= rMij;
            bond_hist_ik[MY] *= rMij;
            bond_hist_ik[MZ] *= rMij;
          }
        }
      }
    }

//TODO    oAtL_i.rFrictionH = oAtR_i.rFrictionH;
    for(k = 0; k < num_bond[i]; k++) {
      int j = atom->map(bond_atom[i][k]);
      j = domain->closest_image(i,j);
      int bond_index_i = bond_index[i][k];
      int bond_state = bondlist[bond_index_i][3];
      if (bond_state == UNBONDED) { // broken bond - contacting
        //BEGIN dry friction force
        int jtype = type[j];
        double rCOF = mca_pair->cof[itype][jtype];
        double *bond_hist_ik = &(bond_hist_i[k][0]);
        double rP = bond_hist_ik[P];
        double *vSij = &(bond_hist_ik[SX]);
        double *vMij = &(bond_hist_ik[MX]);
        if (rP > 0.0) {
          rP = 0.0; bond_hist_ik[P] = 0.0;
        }
        if (rCOF < REAL_NULL_CONST) {
          vSij[0] = vSij[1] = vSij[2] = 0.0;
          vMij[0] = vMij[1] = vMij[2] = 0.0;
        } else {
          double rFDij = fabs(rCOF*rP); // dry friction force
          double rSij = vectorMag3D(vSij);
          if (rSij > rFDij) {
            double rKij = rFDij / rSij;  // correction factor
            vectorScalarMult3D(vSij, rKij);
          }
          rSij = vectorMag3D(vMij) / mca_radius;
          if (rSij > rFDij) {
            double rKij = rFDij / rSij;
            vectorScalarMult3D(vMij, rKij);
          }
/* TODO          rSij = vectorMag3D(vSij);
                    CVector3D vdS = oNbrL_i.vShear; vdS -= oNbrR_i.vShear;
                    rSij *= vdS.length();
                    double rFrictionHeat =  rSij;
                    vdS = oNbrL_i.vShear; vdS -= oNbrR_i.vShear;
                    CVector3D vR2, vdSi;
                    const CMCA3D_Automaton &oAtR_j = aRightA[j];
                    vR2 = oAtL_i.vTheta; vR2 *= -1.0; SmallRotationSum(oAtR_i.vTheta, vR2, vdSi);
                    vR2 = oAtL_j.vTheta; vR2 *= -1.0; SmallRotationSum(oAtR_j.vTheta, vR2, vdS);
                    vR2 = vdS; vR2 *= -1.0; SmallRotationSum(vdSi, vR2, vdS);
                    rSij *= vdS.length();
                    rFrictionHeat += rSij;
                    rFrictionHeat *=  oNbrL_i.rCA * oNbrL_i.rQ * 0.5; //делим пополам между двумя автоматами
                    oAtL_i.rFrictionH += rFrictionHeat; // Plastic heat
                    rTotalFrictionHeat += oAtL_i.rFrictionH; */
        }
        //END dry friction force
      }
    }
  }
  return;
}

/* ---------------------------------------------------------------------- */

void PairMCA::compute_total_force(int eflag, int vflag)
{
  const int nbondlist = neighbor->nbondlist;
  double **f = atom->f;
  double **torque = atom->torque;

  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = 0;

//fprintf(logfile, "PairMCA::compute_total_force \n");  ///AS DEBUG TRACE

///#if defined (_OPENMP)
//#pragma omp parallel for shared(x,f,torque,bondlist,bond_atom,bond_hist) default(none) schedule(static)
///#pragma omp parallel for default(shared) schedule(static)
///#endif
  for (int n = 0; n < nbondlist; n++) {
    const double mca_radius  = atom->mca_radius;
    const double contact_area  = atom->contact_area;
    const int * const tag = atom->tag; // tag of atom is their ID number
    double ** const x = atom->x;
    const double * const mean_stress  = atom->mean_stress;
    int ** const bondlist = neighbor->bondlist;
///AS I do not whant to use it, because it requires to be copied every step///  double **bondhistlist = neighbor->bondhistlist;
    double *** const bond_hist = atom->bond_hist;
    const int * const num_bond = atom->num_bond;
    int ** const bond_atom = atom->bond_atom;

    const int nlocal = atom->nlocal;
    const int newton_bond = force->newton_bond;
    const PairMCA * const mca_pair = (PairMCA*) force->pair;

    int bond_state = bondlist[n][3];
    //1st check if bond is broken,
    if(bond_state == NOT_INTERACT)
    {
       fprintf(logfile,"PairMCA::compute_total_force bond %d does not interact\n",n);
       continue;
    }

    int i1,i2,n1,n2;
    double rsq,r,rinv;
    double tor1,tor2,tor3;
    double delx,dely,delz;
    double dnforce[3],dtforce[3],nv[3];
    double dttorque[3];
    double A;
    double q1,q2;// distance to contact point

    i1 = bondlist[n][0];
    i2 = bondlist[n][1];

    for (n1 = 0; n1 < num_bond[i1]; n1++) {
      const int ib1 = bond_atom[i1][n1];
      if (ib1==tag[i2]) break;
    }
    if (n1 == num_bond[i1]) error->all(FLERR,"Internal error in PairMCA: n1 not found");

    for (n2 = 0; n2 < num_bond[i2]; n2++) {
      const int ib2 = bond_atom[i2][n2];
      if (ib2==tag[i1]) break;
    }
    if (n2 == num_bond[i2]) error->all(FLERR,"Internal error in PairMCA: n2 not found");

    double * const bond_hist1 = &(bond_hist[tag[i1]-1][n1][0]);
    double * const bond_hist2 = &(bond_hist[tag[i2]-1][n2][0]);

    double pi = bond_hist1[P];
    double pj = bond_hist2[P];
    if ( (bond_state == UNBONDED) && ((pi>0.) || (pj>0.)) ) {
      error->warning(FLERR,"PairMCA::compute_total_force (pi>0.)||(pj>0.) - be careful!");
      pi=0.0; pj=0.0;
      continue;
    }

    q1 = mca_radius*(1. + bond_hist1[E]);
    q2 = mca_radius*(1. + bond_hist2[E]);

///fprintf(logfile,"PairMCA::compute_total_force:  bond# %d i1=%d(tag=%d) n1=%d i2=%d(tag=%d) n2=%d pi=%g pj=%g ei=%g ej=%g\n",
///n,i1,tag[i1],n1,i2,tag[i2],n2,pi,pj,bond_hist1[E],bond_hist2[E]);
    //int type = bondlist[n][2];

    double rD0 = 2.0*mca_radius;
    double rDij = bond_hist1[R];
    int itype = atom->type[i1];
    double rKi =  mca_pair->K[itype][itype];
    int jtype = atom->type[i2];
    double rKj =  mca_pair->K[jtype][jtype];
    double rDStress = 0.5 * (mean_stress[i1]/rKi + mean_stress[i2]/rKj);
    A = contact_area * (1. + rDStress) * rD0 / rDij; // contact area updated for pyramid

    //bond_hist1[A] = A;
    //bond_hist2[A] = A;

    delx = x[i1][0] - x[i2][0];
    dely = x[i1][1] - x[i2][1];
    delz = x[i1][2] - x[i2][2];

    rsq = delx*delx + dely*dely + delz*delz;
    r = sqrt(rsq);
    rinv = -1. / r; // "-" means that unit vector is from i1 to i2

    // normal unit vector
    nv[0] = delx*rinv; // compute normal here because in bond_hist1[NX] it has implicit increment
    nv[1] = dely*rinv;
    nv[2] = delz*rinv;

    // normal force
    pi += pj; pi *= 0.5;
    vectorScalarMult3D(nv, pi, dnforce);

    // tangential force
    dtforce[0] = 0.5 * (bond_hist2[SX] - bond_hist1[SX]);
    dtforce[1] = 0.5 * (bond_hist2[SY] - bond_hist1[SY]);
    dtforce[2] = 0.5 * (bond_hist2[SZ] - bond_hist1[SZ]);

    // torque due to tangential force
    vectorCross3D(nv, dtforce, dttorque);

    // torque due to torsion and bending
#ifdef NO_ROTATIONS
    tor1 = tor2 = tor3 = 0.;
#else
    tor1 = 0.5 * A * (bond_hist2[MX] - bond_hist1[MX]);
    tor2 = 0.5 * A * (bond_hist2[MY] - bond_hist1[MY]);
    tor3 = 0.5 * A * (bond_hist2[MZ] - bond_hist1[MZ]);
#endif

    // energy
    //if (eflag) error->all(FLERR,"MCA bonds currently do not support energy calculation");

    // apply force to each of 2 atoms

///#if defined (_OPENMP)
///#pragma omp critical(tfI1)
///#endif
    if (newton_bond || i1 < nlocal) {
      f[i1][0] += (dnforce[0] + dtforce[0]) * A;
      f[i1][1] += (dnforce[1] + dtforce[1]) * A;
      f[i1][2] += (dnforce[2] + dtforce[2]) * A;
      torque[i1][0] += q1 * A * dttorque[0] + tor1;
      torque[i1][1] += q1 * A * dttorque[1] + tor2;
      torque[i1][2] += q1 * A * dttorque[2] + tor3;
    }

///#if defined (_OPENMP)
///#pragma omp critical(tfI2)
///#endif
    if (newton_bond || i2 < nlocal) {
      f[i2][0] -= (dnforce[0] + dtforce[0]) * A;
      f[i2][1] -= (dnforce[1] + dtforce[1]) * A;
      f[i2][2] -= (dnforce[2] + dtforce[2]) * A;
      torque[i2][0] += q2 * A * dttorque[0] - tor1;
      torque[i2][1] += q2 * A * dttorque[1] - tor2;
      torque[i2][2] += q2 * A * dttorque[2] - tor3;
    }

    //if (evflag) ev_tally(i1,i2,nlocal,newton_bond,ebond,0./*fbond*/,delx,dely,delz);
  }
}

/* ---------------------------------------------------------------------- */

void PairMCA::compute(int eflag, int vflag)
{

  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

//fprintf(logfile,"PairMCA::compute\n"); ///AS DEBUG TRACE
///  swap_prev(); Moved to fixMCAExchangeMeanStress::post_integrate()
///  predict_mean_stress(); Moved to fixMCAExchangeMeanStress::post_integrate()
  compute_elastic_force();
  compute_equiv_stress();
  correct_for_plasticity();
  compute_total_force(eflag,vflag);

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

  memory->create(cof,n+1,n+1,"pair:cof");
  memory->create(G,n+1,n+1,"pair:G");
  memory->create(K,n+1,n+1,"pair:K");
  memory->create(ModulusPredictOne,n+1,n+1,"pair:ModulusPredictOne");
  memory->create(ModulusPredictAll,n+1,n+1,"pair:ModulusPredictAll");
  memory->create(Sy,n+1,n+1,"pair:Sy");
  memory->create(Eh,n+1,n+1,"pair:Eh");
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

  double cof_one = force->numeric(FLERR,arg[2]);
  double G_one = force->numeric(FLERR,arg[3]);
  double K_one = force->numeric(FLERR,arg[4]);
  double rCoordNumber = (double) atom->coord_num;
  double ModulusPredictOne_one = 3.0*K_one*2.0*G_one*rCoordNumber /
                                 (3.0*K_one*(rCoordNumber-1.0) + 2.0*G_one);
  double ModulusPredictAll_one = (3.0*K_one + 2.0*G_one*(rCoordNumber-1.0)) / rCoordNumber;
fprintf(logfile,"Computing Predictor modulii using coordination number=%g :\n",rCoordNumber);
fprintf(logfile,"ModulusPredictOne= %g\tModulusPredictAll= %g\n",ModulusPredictOne_one, ModulusPredictAll_one);


  double Sy_one = -1.0; // no plasticity
  double Eh_one = 0.0; // no harderning
  if (narg > 5) {
    Sy_one = force->numeric(FLERR,arg[5]);
    if (narg == 7) Eh_one = force->numeric(FLERR,arg[6]);
    if (narg > 7) error->all(FLERR,"Incorrect args for mca pair coefficients for plasticity");
  }

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      cof[i][j] = cof_one;
      G[i][j] = G_one;
      K[i][j] = K_one;
      ModulusPredictOne[i][j] = ModulusPredictOne_one;
      ModulusPredictAll[i][j] = ModulusPredictAll_one;
      Sy[i][j] = Sy_one;
      Eh[i][j] = Eh_one;
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
    cof[i][j] = (cof[i][i] * cof[j][j]) / (cof[i][i] + cof[j][j]);
    G[i][j] = (G[i][i] * G[j][j]) / (G[i][i] + G[j][j]);
    K[i][j] = (K[i][i] * K[j][j]) / (K[i][i] + K[j][j]);
    Sy[i][j] = (Sy[i][i] * Sy[j][j]) / (Sy[i][i] + Sy[j][j]);
    Eh[i][j] = (Eh[i][i] * Eh[j][j]) / (Eh[i][i] + Eh[j][j]);
  }

  cof[j][i] = cof[i][j];
  G[j][i] = G[i][j];
  K[j][i] = K[i][j];
  ModulusPredictOne[j][i] = ModulusPredictOne[i][j];
  ModulusPredictAll[j][i] = ModulusPredictAll[i][j];

  Sy[j][i] = Sy[i][j];
  Eh[j][i] = Eh[i][j];

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
        fwrite(&cof[i][j],sizeof(double),1,fp);
        fwrite(&G[i][j],sizeof(double),1,fp);
        fwrite(&K[i][j],sizeof(double),1,fp);
        fwrite(&ModulusPredictOne[i][j],sizeof(double),1,fp);
        fwrite(&ModulusPredictAll[i][j],sizeof(double),1,fp);
        fwrite(&Sy[i][j],sizeof(double),1,fp);
        fwrite(&Eh[i][j],sizeof(double),1,fp);
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
          fread(&cof[i][j],sizeof(double),1,fp);
          fread(&G[i][j],sizeof(double),1,fp);
          fread(&K[i][j],sizeof(double),1,fp);
          fread(&ModulusPredictOne[i][j],sizeof(double),1,fp);
          fread(&ModulusPredictAll[i][j],sizeof(double),1,fp);
          fread(&Sy[i][j],sizeof(double),1,fp);
          fread(&Eh[i][j],sizeof(double),1,fp);
        }
        MPI_Bcast(&cof[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&G[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&K[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&ModulusPredictOne[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&ModulusPredictAll[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&Sy[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&Eh[i][j],1,MPI_DOUBLE,0,world);
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
    fprintf(fp,"%d %g %g %g %g %g %g %g\n",i,cof[i][i],G[i][i],K[i][i],ModulusPredictOne[i][i],ModulusPredictAll[i][i],Sy[i][i],Eh[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairMCA::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp,"%d %d %g %g %g %g %g %g %g\n",i,j,cof[i][i],G[i][j],K[i][j],ModulusPredictOne[i][j],ModulusPredictAll[i][j],Sy[i][j],Eh[i][j]);
}
