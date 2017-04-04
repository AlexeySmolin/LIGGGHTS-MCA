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
     BREAKSTYLE_NONE,
     BREAKSTYLE_EQUIV_STRAIN,
     BREAKSTYLE_EQUIV_STRESS,
     BREAKSTYLE_DRUCKER_PRAGER
    };

enum{
     BINDSTYLE_NONE,
     BINDSTYLE_PRESSURE,
     BINDSTYLE_PLASTIC_HEAT,
     BINDSTYLE_COMBINED,
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
    memory->destroy(breakmode);
    memory->destroy(bindmode);
    memory->destroy(crackVelo);
    memory->destroy(breakVal1);
    memory->destroy(breakVal2);
    memory->destroy(shapeDrPr);
    memory->destroy(volumeDrPr);
    memory->destroy(bindPressure);
    memory->destroy(bindPlastHeat);
  }
}

/* ---------------------------------------------------------------------- */

void BondMCA::allocate()
{
  allocated = 1;
  int n = atom->nbondtypes;

  memory->create(breakmode,(n+1),"bond:breakmode");
  memory->create(bindmode,(n+1),"bond:bindmode");
  memory->create(crackVelo,(n+1),"bond:crackVelo");
  memory->create(breakVal1,(n+1),"bond:breakVal1");
  memory->create(breakVal2,(n+1),"bond:breakVal2");
  memory->create(shapeDrPr,(n+1),"bond:shapeDrPr");
  memory->create(volumeDrPr,(n+1),"bond:volumeDrPr");
  memory->create(bindPressure,(n+1),"bond:bindPressure");
  memory->create(bindPlastHeat,(n+1),"bond:bindPlastHeat");

  memory->create(setflag,(n+1),"bond:setflag");
  for (int i = 1; i <= n; i++)
    setflag[i] = 0;
}

/* ---------------------------------------------------------------------- */

void  BondMCA::init_style()
{
/* AS TODO It seems we do not need this
    if(breakmode == BREAKSTYLE_STRESS_TEMP)
       fix_Temp = static_cast<FixPropertyAtom*>(modify->find_fix_property("Temp","property/atom","scalar",1,0,"mca bond"));
                                                                                                              "bond mca")); ???? */
}

/* ---------------------------------------------------------------------- */

void BondMCA::compute(int eflag, int vflag)
{
  int *num_bond = atom->num_bond;
  int **bond_atom = atom->bond_atom;
  int *tag = atom->tag; // tag of atom is their ID number
  double ***bond_hist = atom->bond_hist;
  double **x = atom->x;
  double *cont_distance = atom->cont_distance;

  int nbondlist = neighbor->nbondlist;
  int **bondlist = neighbor->bondlist;
  double cutoff=neighbor->skin;
  ///AS I do not whant to use it, because it requires to be copied every step/// double **bondhistlist = neighbor->bondhistlist;

//fprintf(logfile, "BondMCA::compute \n"); ///AS DEBUG

  comm->reverse_comm(); /// We copy only contact distances

/* TODO AS: It seems we do not need this
  if(breakmode == BREAKSTYLE_STRESS_TEMP) {
    if(!fix_Temp) error->all(FLERR,"Internal error in BondMCA::compute");
    Temp = fix_Temp->vector_atom;
  } */

//fprintf(logfile,"boxlo[0]=%g boxhi[0]=%g cutoff=%g \n",domain->boxlo[0],domain->boxhi[0],cutoff);
//fprintf(logfile,"boxlo[1]=%g boxhi[1]=%g cutoff=%g \n",domain->boxlo[1],domain->boxhi[1],cutoff);
//fprintf(logfile,"boxlo[2]=%g boxhi[2]=%g cutoff=%g \n",domain->boxlo[2],domain->boxhi[2],cutoff);

  for (int n = 0; n < nbondlist; n++) {

    int i1 = bondlist[n][0];
    int i2 = bondlist[n][1];
    int bond_state = bondlist[n][3];
    int b_type = bondlist[n][2];
    int brk_mode = breakmode[b_type];
    int bind_mode = bindmode[b_type];

    int n1, n2;

    for (n1 = 0; n1 < num_bond[i1]; n1++) {
      if (bond_atom[i1][n1]==tag[i2]) break;
//fprintf(logfile, "BondMCA::compute bond_atom[%d][%d]=%d \n", i1, n1, bond_atom[i1][n1]); ///AS DEBUG
    }
    if (n1 == num_bond[i1]) error->all(FLERR,"Internal error in BondMCA::compute(): n1 not found");
    double *bond_hist1 = bond_hist[i1][n1];
    for (n2 = 0; n2 < num_bond[i2]; n2++) {
      if (bond_atom[i2][n2]==tag[i1]) break;
//fprintf(logfile, "BondMCA::compute bond_atom[%d][%d]=%d \n", i1, n1, bond_atom[i1][n1]); ///AS DEBUG
    }
    if (n2 == num_bond[i2]) error->all(FLERR,"Internal error in BondMCA::compute(): n2 not found");
    double *bond_hist2 = bond_hist[i2][n2];
    double rIJ = bond_hist1[R];
    double rJI = bond_hist2[R];
    if (fabs(rIJ-rJI) > 5.0E-12*(atom->mca_radius)) fprintf(logfile,"BondMCA::compute(): bond %d(%d-%d) rIJ(%-1.16e) != rJI(%-1.16e)\n",n,i1,i2,rIJ,rJI);

    double cont_distance1 = cont_distance[i1];
    double cont_distance2 = cont_distance[i2];

    //1st check if the bond has been already broken,
    if(bond_state) { //AS TODO may be corrected in case of slow fracture using crackVelo
      if(bond_state==2) {
        if(rIJ <= (cont_distance1 + cont_distance2)) {
          bond_state = bondlist[n][3] = 1;
          fprintf(logfile,"BondMCA::compute(): bond %d is contacting\n",n);
        }
        else continue;
      } else {
        if(rIJ > (cont_distance1 + cont_distance2)) {
          bond_state = bondlist[n][3] = 2;
          fprintf(logfile,"BondMCA::compute(): bond %d is not interacting\n",n);
          continue;
        }
        if(bind_mode==BINDSTYLE_NONE) continue;
        else {
          fprintf(logfile,"BondMCA::compute(): binding has not been implemented so far\n");
          continue; // TODO
        }
      }
    }

    // breaking the bond if criterion met
    if(brk_mode == BREAKSTYLE_NONE) continue;
    else
    {
      const PairMCA * const mca_pair = (PairMCA*) force->pair;
      const double * const mean_stress = atom->mean_stress;
      const int * const a_type = atom->type;
      int type1 = a_type[i1];
      int type2 = a_type[i2];
      double rT1 = 0.0;
      double rT2 = 0.0;
      double criterion_mag = 0.0;
      bool broken = false;
      if(brk_mode == BREAKSTYLE_EQUIV_STRAIN) {
        if(bond_hist1[P] < 0.0) { // In compression we do not break the bond!
          rT1 = 0.0;
        } else {
          rT1 = bond_hist1[E] - mean_stress[i1] / (3.0 * mca_pair->K[type1][type2]);
        }
        if(bond_hist2[P] < 0.0) { // In compression we do not break the bond!
          rT2 = 0.0;
        } else {
          rT2 = bond_hist2[E] - mean_stress[i2] / (3.0 * mca_pair->K[type2][type1]);
        }
        double vY[3];
        const double * const vN1 = &(bond_hist1[NX]);       // normal unit vector
        const double * const vShear1 =  &(bond_hist1[SHX_PREV]); // shear strain
        vectorCross3D(vShear1, vN1, vY);        // shear
        rT1 = rT1*rT1 + vectorMag3DSquared(vY); // strain of shape change
        const double * const vN2 = &(bond_hist2[NX]);       // normal unit vector
        const double * const vShear2 =  &(bond_hist2[SHX_PREV]); // shear strain
        vectorCross3D(vShear2, vN2, vY);        // shear
        rT2 = rT2*rT2 + vectorMag3DSquared(vY); // strain of shape change
        double mult = 4.0 / 3.0;
        rT1 = sqrt(mult*rT1);
        rT2 = sqrt(mult*rT2);
//fprintf(stderr,"I eps=%g dVVo/=%g rT1= %g\n",apzNbr[iNbrx_I].rE,arMS[iAx_I]/(3.0*azSL[si].rK), rT1);
//fprintf(stderr,"J eps=%g dVVo/=%g rT2= %g\n",apzNbr[iNbrx_J].rE,arMS[iAx_J]/(3.0*azSL[sj].rK), rT2);
        criterion_mag = rT1 > rT2 ? rT1 : rT2; //0.5*fabs(rT1 + rT2);
        broken = breakVal1[b_type] < criterion_mag;
//fprintf(logfile,"bond# %d EQUIV_STRAIN: breakVal1[%d]=%g  rT1=%g rT2=%g\n", n,  b_type, breakVal1[b_type], rT1, rT2);
        if(broken) {
          bondlist[n][3] = 1;
          fprintf(logfile,"broken bond %d at step %d\n",n,update->ntimestep);
          fprintf(logfile,"   it was EQUIV_STRAIN: breakVal1[%d]=%g < criterion_mag=%g\n", b_type, breakVal1[b_type], criterion_mag);
        }
      }
      if(brk_mode == BREAKSTYLE_EQUIV_STRESS) {
        if(bond_hist1[P] < 0.0) { // In compression we do not break the bond!
          rT1 = 0.0;
        } else {
          rT1 = bond_hist1[P] - mean_stress[i1];
        }
        if(bond_hist2[P] < 0.0) { // In compression we do not break the bond!
          rT2 = 0.0;
        } else {
          rT2 = bond_hist2[P] - mean_stress[i2];
        }
        const double * const vShF1 =  &(bond_hist1[SX]); // shear force
        rT1 = rT1*rT1 + vectorMag3DSquared(vShF1); // analogue of equivalent stress
        const double * const vShF2 =  &(bond_hist2[SX]); // shear force
        rT2 = rT2*rT2 + vectorMag3DSquared(vShF2); // analogue of equivalent stress
        double mult = 3.0;
        rT1 = sqrt(mult*rT1);
        rT2 = sqrt(mult*rT2);
        criterion_mag = rT1 > rT2 ? rT1 : rT2;
        broken = breakVal1[b_type] < criterion_mag;
        if(broken) {
          bondlist[n][3] = 1;
          fprintf(logfile,"broken bond %d at step %d\n",n,update->ntimestep);
          fprintf(logfile,"   it was EQUIV_STRESS: breakVal1[%d]=%g < criterion_mag=%g\n", b_type, breakVal1[b_type], criterion_mag);
        }
      }
      if(brk_mode == BREAKSTYLE_DRUCKER_PRAGER) {
        rT1 = bond_hist1[P] - mean_stress[i1];
        rT2 = bond_hist2[P] - mean_stress[i2];
        const double * const vShF1 =  &(bond_hist1[SX]); // shear force
        rT1 = rT1*rT1 + vectorMag3DSquared(vShF1); // analogue of equivalent stress
        const double * const vShF2 =  &(bond_hist2[SX]); // shear force
        rT2 = rT2*rT2 + vectorMag3DSquared(vShF2); // analogue of equivalent stress
        double mult = 3.0;
        rT1 = sqrt(mult*rT1);
        rT2 = sqrt(mult*rT2);
        double q1 = 1.0 + bond_hist1[E];
        double q2 = 1.0 + bond_hist2[E];
        rT1 = (rT1 * q1 + rT2 * q2) * 0.5; // Equivalent Stress
        rT2 = (mean_stress[i1] * q1 + mean_stress[i2] * q2) * 0.5; // Mean Stress
        criterion_mag = shapeDrPr[b_type] * rT1 + volumeDrPr[b_type] * rT2;
        broken = breakVal2[b_type] < criterion_mag;
        if(broken) {
          bondlist[n][3] = 1;
          fprintf(logfile,"broken bond %d at step %d\n",n,update->ntimestep);
          fprintf(logfile,"   it was DRUCKER_PRAGER: breakVal1[%d]=%g breakVal2[%d]=%g < criterion_mag=%g\n", b_type, breakVal1[b_type], breakVal2[b_type], criterion_mag);
        }
      }
      if((broken)  && (rIJ > (cont_distance1 + cont_distance2))) {
          bondlist[n][3] = 2;
          fprintf(logfile,"BondMCA::compute(): bond %d is broken and not interacting\n",n);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   only called if nflag != 0 in nflag = neighbor->decide();
------------------------------------------------------------------------- */

void BondMCA::build_bond_index()
{
  const int nbondlist = neighbor->nbondlist;
fprintf(logfile, "BondMCA::build_bond_index \n");///AS DEBUG

#if defined (_OPENMP)
#pragma omp parallel for default(shared) schedule(static)
#endif
  for (int n = 0; n < nbondlist; n++) {
    const int * const tag = atom->tag; // tag of atom is their ID number
    int ** const bondlist = neighbor->bondlist;
    const int * const num_bond = atom->num_bond;
    int ** const bond_atom = atom->bond_atom;
    int ** bond_index = atom->bond_index;
    const int nlocal = atom->nlocal;
//    const int newton_bond = force->newton_bond;

    int i1,i2,n1,n2;

    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    for (n1 = 0; n1 < num_bond[i1]; n1++) {
      const int ib1 = bond_atom[i1][n1];
      if (ib1==tag[i2]) break;
    }
    if (n1 == num_bond[i1]) error->all(FLERR,"Internal error in BondMCA::build_bond_index: n1 not found");

    for (n2 = 0; n2 < num_bond[i2]; n2++) {
      const int ib2 = bond_atom[i2][n2];
      if (ib2==tag[i1]) break;
    }
    if (n2 == num_bond[i2]) error->all(FLERR,"Internal error in BondMCA::build_bond_index: n2 not found");

    if (i1 < nlocal && i2 < nlocal) { ///??????
      bond_index[i1][n1] = n;
      bond_index[i2][n2] = n;
//fprintf(logfile, "BondMCA::build_bond_index bond_index[%d][%d]=%d\n",i1,i2,n);///AS DEBUG
    } else {
//fprintf(logfile, "BondMCA::build_bond_index bond_index[%d][%d] NOT FOUND nlocal=%d\n",i1,i2,nlocal);///AS DEBUG
    }
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more types
------------------------------------------------------------------------- */

void BondMCA::coeff(int narg, char **arg)
{
// bond_coeff  N ${COF} ${CrackVelo} ${FRACT_CRITERION} ${FRACT_PARAM} ${BIND_CRITERION} ${BIND_PARAM}

  if(narg < 1)  error->all(FLERR,"Small number of args for MCA bond coefficients (<1)");

  int breakmode_one;
  int bindmode_one;

  double crackVelo_one = -1.0;
  double breakVal1_one = 0.0;
  double breakVal2_one = 0.0;
  double shapeDrPr_one  = 0.0;
  double volumeDrPr_one = 0.0;
  double bindPressure_one = 0.0;
  double bindPlastHeat_one = 0.0;

  if(narg <= 2) {
    breakmode_one = BREAKSTYLE_NONE;
    bindmode_one = BINDSTYLE_NONE;
    if(narg == 2) crackVelo_one = force->numeric(FLERR,arg[1]);
    if (!allocated) allocate();
  } else {

    crackVelo_one = force->numeric(FLERR,arg[1]);
    int i_arg = 2;
    int style = force->numeric(FLERR,arg[i_arg]);

    if(style == 0) {
      breakmode_one = BREAKSTYLE_NONE;
    }
    else if(style == 1) {
      breakmode_one = BREAKSTYLE_EQUIV_STRAIN;
      if (narg < (i_arg+2)) error->all(FLERR,"Incorrect 'Equivalent Strain' arg for MCA bond coefficients");
      i_arg++;
      breakVal1_one = force->numeric(FLERR,arg[i_arg]);
//fprintf(logfile,"BondMCA::coeff():BREAKSTYLE_EQUIV_STRAIN breakmode_one=%d breakVal1_one=%g\n",breakmode_one,breakVal1_one);
    }
    else if(style == 2) {
      breakmode_one = BREAKSTYLE_EQUIV_STRESS;
      if (narg < (i_arg+2)) error->all(FLERR,"Incorrect 'Equivalent Stress' arg for MCA bond coefficients");
      i_arg++;
      breakVal1_one = force->numeric(FLERR,arg[i_arg]);
//fprintf(logfile,"BondMCA::coeff():BREAKSTYLE_EQUIV_STRESS breakmode_one=%d breakVal1_one=%g\n",breakmode_one,breakVal1_one);
    }
    else if(style == 3) {
      breakmode_one = BREAKSTYLE_DRUCKER_PRAGER;
      if (narg < (i_arg+3)) error->all(FLERR,"Incorrect Drucker-Prager args for MCA bond coefficients");
      i_arg++;
      breakVal1_one = force->numeric(FLERR,arg[i_arg]); // tension strength
      i_arg++;
      breakVal2_one = force->numeric(FLERR,arg[i_arg]); // compression strength
      double rVariable = breakVal2_one / breakVal1_one;
      shapeDrPr_one  = 0.5 * (rVariable + 1.0);
      volumeDrPr_one = 1.5 * (rVariable - 1.0);
//fprintf(logfile,"BondMCA::coeff():BREAKSTYLE_DRUCKER_PRAGER breakmode_one=%d tension_strength=%g compression_strength=%g\n",breakmode_one,breakVal1_one,breakVal2_one);
//fprintf(logfile,"BondMCA::coeff():BREAKSTYLE_DRUCKER_PRAGER                  shapeDrPr_one=%g volumeDrPr_one=%g\n",shapeDrPr_one,volumeDrPr_one);
    }

    i_arg++;
    if (narg < (i_arg+1))
       style = BINDSTYLE_NONE;
    else
      style = force->numeric(FLERR,arg[i_arg]);
//fprintf(logfile,"BondMCA::coeff(): expecting binding params narg=%d  i_arg=%d style=%d\n",narg,i_arg,style);
    if(style == 0) {
      bindmode_one = BINDSTYLE_NONE;
    }
    else if(style == 1) {
      bindmode_one = BINDSTYLE_PRESSURE;
      if (narg < (i_arg+2)) error->all(FLERR,"Incorrect 'Pressure' arg for MCA bond coefficients");
      i_arg++;
      bindPressure_one = force->numeric(FLERR,arg[i_arg]);
//fprintf(logfile,"BondMCA::coeff():BINDSTYLE_PRESSURE bindmode_one=%d bindPressure_one=%g\n",bindmode_one,bindPressure_one);
    }
    else if(style == 2) {
      bindmode_one = BINDSTYLE_PLASTIC_HEAT;
      if (narg < (i_arg+2)) error->all(FLERR,"Incorrect 'Plastic Heat' arg for MCA bond coefficients");
      i_arg++;
      bindPlastHeat_one = force->numeric(FLERR,arg[i_arg]);
//fprintf(logfile,"BondMCA::coeff():BINDSTYLE_PLASTIC_HEAT bindmode_one=%d bindPlastHeat_one=%g\n",bindmode_one,bindPlastHeat_one);
    }
    else if(style == 3) {
      bindmode_one = BINDSTYLE_COMBINED;
      if (narg < (i_arg+3)) error->all(FLERR,"Incorrect combined binding args for MCA bond coefficients");
      i_arg++;
      bindPressure_one = force->numeric(FLERR,arg[i_arg]);
      i_arg++;
      bindPlastHeat_one = force->numeric(FLERR,arg[i_arg]);
//fprintf(logfile,"BondMCA::coeff():BINDSTYLE_COMBINED bindmode_one=%d bindPressure_one=%g bindPlastHeat_one=%g\n",bindmode_one,bindPressure_one,bindPlastHeat_one);
    }
    else  error->all(FLERR,"Incorrect args for MCA bond coefficients");

    if (!allocated) allocate();

  }

  int ilo,ihi;
  force->bounds(arg[0],atom->nbondtypes,ilo,ihi);
fprintf(logfile,"BondMCA::coeff(): ilo=%d ihi=%d \n",ilo,ihi);
  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    breakmode[i] = breakmode_one;
    bindmode[i] = bindmode_one;
    crackVelo[i] = crackVelo_one;
    breakVal1[i] = breakVal1_one;
    breakVal2[i] = breakVal2_one;
    shapeDrPr[i]  = shapeDrPr_one;
    volumeDrPr[i] = volumeDrPr_one;
    bindPressure[i] = bindPressure_one;
    bindPlastHeat[i] = bindPlastHeat_one;
fprintf(logfile,"BondMCA::coeff():i=%d crackVelo=%g\n\tbreakmode=%d breakVal1=%g breakVal2=%g shapeDrPr=%g volumeDrPr=%g\n\tbindmode=%d bindPressure=%g bindPlastHeat=%g\n",
i,crackVelo[i],breakmode[i],breakVal1[i],breakVal2[i],shapeDrPr[i],volumeDrPr[i],bindmode[i],bindPressure[i],bindPlastHeat[i]);
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
  fwrite(&breakmode[1],sizeof(int),atom->nbondtypes,fp);
  fwrite(&bindmode[1],sizeof(int),atom->nbondtypes,fp);
  fwrite(&crackVelo[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&breakVal1[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&breakVal2[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&shapeDrPr[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&volumeDrPr[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&bindPressure[1],sizeof(double),atom->nbondtypes,fp);
  fwrite(&bindPlastHeat[1],sizeof(double),atom->nbondtypes,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them
------------------------------------------------------------------------- */

void BondMCA::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
    fread(&breakmode[1],sizeof(int),atom->nbondtypes,fp);
    fread(&bindmode[1],sizeof(int),atom->nbondtypes,fp);
    fread(&crackVelo[1],sizeof(double),atom->nbondtypes,fp);
    fread(&breakVal1[1],sizeof(double),atom->nbondtypes,fp);
    fread(&breakVal2[1],sizeof(double),atom->nbondtypes,fp);
    fread(&shapeDrPr[1],sizeof(double),atom->nbondtypes,fp);
    fread(&volumeDrPr[1],sizeof(double),atom->nbondtypes,fp);
    fread(&bindPressure[1],sizeof(double),atom->nbondtypes,fp);
    fread(&bindPlastHeat[1],sizeof(double),atom->nbondtypes,fp);
  }
  MPI_Bcast(&breakmode[1],atom->nbondtypes,MPI_INT,0,world);
  MPI_Bcast(&bindmode[1],atom->nbondtypes,MPI_INT,0,world);
  MPI_Bcast(&crackVelo[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&breakVal1[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&breakVal2[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&shapeDrPr[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&volumeDrPr[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&bindPressure[1],atom->nbondtypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&bindPlastHeat[1],atom->nbondtypes,MPI_DOUBLE,0,world);

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
