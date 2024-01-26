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
#include "mpi.h"
#include "string.h"
#include "stdlib.h"
#include "fix_bond_create_mca.h"
#include "update.h"
#include "domain.h"
#include "respa.h"
#include "atom.h"
#include "force.h"
#include "pair.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "random_mars.h"
#include "memory.h"
#include "error.h"
#include "modify.h"
#include <iostream>
#include "atom_vec_mca.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MCAAtomConst;

#define BIG 1.0e20

/* ---------------------------------------------------------------------- */

FixBondCreateMCA::FixBondCreateMCA(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 9) error->all(FLERR,"Illegal fix bond/create/mca command");

  MPI_Comm_rank(world,&me);

  nevery = atoi(arg[3]);
  if (nevery <= 0) error->all(FLERR,"Illegal fix bond/create/mca command");

  force_reneighbor = 1;
  next_reneighbor = -1;
  vector_flag = 1;
  size_vector = 2;
  global_freq = 1;
  extvector = 0;

  iatomtype = atoi(arg[4]);
  jatomtype = atoi(arg[5]);
  double cutoff = atof(arg[6]);
  btype = atoi(arg[7]);
  maxbondsperatom = atoi(arg[8]);

  if (iatomtype < 1 || iatomtype > atom->ntypes ||
      jatomtype < 1 || jatomtype > atom->ntypes)
    error->all(FLERR,"Invalid atom type in fix bond/create/mca command");
  if (cutoff < 0.0) error->all(FLERR,"Illegal fix bond/create/mca command");
  if (btype < 1 || btype > atom->nbondtypes)
    error->all(FLERR,"Invalid bond type in fix bond/create/mca command");

  cutsq = cutoff*cutoff;

  // optional keywords

  imaxbond = 0;
  inewtype = iatomtype;
  jmaxbond = 0;
  jnewtype = jatomtype;
  init_state = BONDED;
  fraction = 1.0;
  char * seed = "11939";// "12345";

  int iarg = 9;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"iparam") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix bond/create/mca command");
      imaxbond = atoi(arg[iarg+1]);
      inewtype = atoi(arg[iarg+2]);
      if (imaxbond < 0) error->all(FLERR,"Illegal fix bond/create/mca command");
      if (inewtype < 1 || inewtype > atom->ntypes)
        error->all(FLERR,"Invalid atom type in fix bond/create/mca command");
      iarg += 3;
    } else if (strcmp(arg[iarg],"jparam") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix bond/create/mca command");
      jmaxbond = atoi(arg[iarg+1]);
      jnewtype = atoi(arg[iarg+2]);
      if (jmaxbond < 0) error->all(FLERR,"Illegal fix bond/create/mca command");
      if (jnewtype < 1 || jnewtype > atom->ntypes)
        error->all(FLERR,"Invalid atom type in fix bond/create/mca command");
      iarg += 3;
    } else if (strcmp(arg[iarg],"state") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix bond/create/mca command");
      init_state = atoi(arg[iarg+1]);
      if (init_state < BONDED || init_state > NOT_INTERACT)
        error->all(FLERR,"Illegal state in fix bond/create/mca command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"prob") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix bond/create/mca command");
      fraction = atof(arg[iarg+1]);
      seed = arg[iarg+2];
      if (fraction < 0.0 || fraction > 1.0)
        error->all(FLERR,"Illegal fix bond/create/mca command");
      //if (seed <= 0) error->all(FLERR,"Illegal fix bond/create/mca command");
      iarg += 3;
    } else error->all(FLERR,"Illegal fix bond/create/mca command");
  }

  // error check

  if (atom->molecular == 0)
    error->all(FLERR,"Cannot use fix bond/create/mca with non-molecular systems");
  if (iatomtype == jatomtype &&
      ((imaxbond != jmaxbond) || (inewtype != jnewtype)))
    error->all(FLERR,"Inconsistent iparam/jparam values in fix bond/create/mca command");

  // initialize Marsaglia RNG with processor-unique seed

    random = new RanMars(lmp, seed, true); // TODO? seed + me);

  // perform initial allocation of atom-based arrays
  // register with Atom class
  // bondcount values will be initialized in setup()

  bondcount = NULL;
  grow_arrays(atom->nmax);
  atom->add_callback(0);
  countflag = 0;

  // set comm sizes needed by this fix

  comm_forward = maxbondsperatom + 2;
  comm_reverse = maxbondsperatom + 1;

  // allocate arrays local to this fix

  nmax = 0;
  npartner = NULL;
  partner = NULL;
  probability = NULL;
  // distsq = NULL;

  // zero out stats

  createcount = 0;
  createcounttotal = 0;
}

/* ---------------------------------------------------------------------- */

FixBondCreateMCA::~FixBondCreateMCA()
{
  // unregister callbacks to this fix from Atom class

  atom->delete_callback(id,0);

  delete random;

  // delete locally stored arrays

  memory->sfree(bondcount);
  memory->sfree(npartner);
  memory->destroy(partner);
  memory->sfree(probability);

  //NP do _not_  delete this fix here - should stay active
  //NP modify->delete_fix("exchange_bonds_mca");
}

/* ---------------------------------------------------------------------- *
  // registernig of these fixes are moved to AtomVecMCA::init() because they should be registered once

void FixBondCreateMCA::post_create()
{
   / register a fix to call mca/meanstress
    char* fixarg[4];

    fixarg[0] = strdup("mca_meanstress");
    fixarg[1] = strdup("all");
    fixarg[2] = strdup("mca/meanstress");
    modify->add_fix(3,fixarg);
    free(fixarg[0]);
    free(fixarg[1]);
    free(fixarg[2]);

/   // register a fix to excange mca bonds across processors
    fixarg[0] = strdup("exchange_bonds_mca");
    fixarg[1] = strdup("all");
    fixarg[2] = strdup("bond/exchange/mca");
    modify->add_fix(3,fixarg);
    free(fixarg[0]);
    free(fixarg[1]);
    free(fixarg[2]);
/
}*/


/* ---------------------------------------------------------------------- */

int FixBondCreateMCA::setmask()
{
  int mask = 0;
//  mask |= PRE_FORCE; - moved it to BondMCA::build_bond_index() caled from Neighbor::bond_all()
  mask |= POST_INTEGRATE;
  mask |= POST_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

int FixBondCreateMCA::modify_param(int narg, char **arg)
{
    if(narg != 2) error->all(FLERR,"Illegal fix_modify command");
    if(strcmp(arg[0],"every") == 0) nevery = atoi(arg[1]);
    else error->all(FLERR,"Illegal fix_modify command");
    return 2;
}

/* ---------------------------------------------------------------------- */

void FixBondCreateMCA::init()
{
  if(force->pair == NULL) error->all(FLERR,"Fix bond/create/mca force->pair is NULL");

  if(!(force->bond_match("mca")))
     error->all(FLERR,"Fix bond/create can only be used together with dedicated 'mca' bond styles");

  // check cutoff for iatomtype,jatomtype - cutneighsq is used here
  double cutsq_limit = sqrt(force->pair->cutsq[iatomtype][jatomtype]) + neighbor->skin;
  cutsq_limit *= cutsq_limit;
  if (force->pair == NULL || cutsq > cutsq_limit)
    error->all(FLERR,"Fix bond/create/mca cutoff is longer than pairwise cutoff");
/*
  // require special bonds = 0,1,1

  int flag = 0;
  if (force->special_lj[1] != 0.0 || force->special_lj[2] != 1.0 ||
      force->special_lj[3] != 1.0) flag = 1;
  if (force->special_coul[1] != 0.0 || force->special_coul[2] != 1.0 ||
      force->special_coul[3] != 1.0) flag = 1;
  if (flag) error->all(FLERR,"Fix bond/create requires special_bonds = 0,1,1");

  // warn if angles, dihedrals, impropers are being used

  if (force->angle || force->dihedral || force->improper) {
    if (me == 0)
      error->warning(FLERR,"Created bonds will not create angles, "
                     "dihedrals, or impropers");
  }
*/
  // need a half neighbor list, built when ever re-neighboring occurs

  int irequest = neighbor->request((void *) this);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->fix = 1;

  if (strcmp(update->integrate_style,"respa") == 0)
    nlevels_respa = ((Respa *) update->integrate)->nlevels;
}

/* ---------------------------------------------------------------------- */

void FixBondCreateMCA::init_list(int id, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void FixBondCreateMCA::setup(int vflag)
{
  int i,j,m;

  // compute initial bondcount if this is first run
  // can't do this earlier, like in constructor or init, b/c need ghost info

  if (countflag) return;
  countflag = 1;

  // count bonds stored with each bond I own
  // if newton bond is not set, just increment count on atom I
  // if newton bond is set, also increment count on atom J even if ghost
  // bondcount is long enough to tally ghost atom counts

  int *num_bond = atom->num_bond;
  int **bond_type = atom->bond_type;
  int **bond_atom = atom->bond_atom;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;
  int newton_bond = force->newton_bond;

//if (logfile) fprintf(logfile, "FixBondCreateMCA::setup newton_bond=%d\n",newton_bond);///AS DEBUG
  for (i = 0; i < nall; i++) bondcount[i] = 0;

  for (i = 0; i < nlocal; i++)
    for (j = 0; j < num_bond[i]; j++) {
      if (bond_type[i][j] == btype) {
        bondcount[i]++;
        if (newton_bond) {
          m = atom->map(bond_atom[i][j]);
          if (m < 0)
            error->one(FLERR,"Could not count initial bonds in fix bond/create/mca");
          bondcount[m]++;
        }
      }
    }

  // if newton_bond is set, need to sum bondcount

  commflag = 0;
  if (newton_bond) comm->reverse_comm_fix(this);
}

/*
void FixBondCreateMCA::pre_force(int vflag) - moved it to BondMCA::build_bond_index() caled from Neighbor::bond_all()
{
if (logfile) fprintf(logfile, "FixBondCreateMCA::pre_force \n");///AS DEBUG
  build_bond_index();
}*/

/* ---------------------------------------------------------------------- */

void FixBondCreateMCA::post_integrate()
{
  int i,j,k,ii,jj,inum,jnum,itype,jtype,possible;
  double xtmp,ytmp,ztmp,delx,dely,delz;
  int *ilist,*jlist,*numneigh,**firstneigh;
  int flag;
  double *cont_distance = atom->cont_distance;

  if (nevery == 0 || update->ntimestep % nevery ) {
    return;
  }
//  if (logfile) fprintf(logfile, "FixBondCreateMCA::post_integrate iatomtype=%d jatomtype=%d btype=%d cutsq=%g\n",iatomtype, jatomtype, btype, cutsq);///AS DEBUG

  // need updated ghost atom positions

  comm->forward_comm();

  // forward comm of bondcount, so ghosts have it

  commflag = 0;
  comm->forward_comm_fix(this);

  // resize bond partner list and initialize it
  // probability array overlays distsq array
  // needs to be atom->nmax in length

  if (atom->nmax > nmax) {
    nmax = atom->nmax;
    memory->grow(partner,nmax,maxbondsperatom,"bond/create/mca:partner");
    npartner = (int*) memory->srealloc(npartner,nmax*sizeof(int),"bond/create/mca:npartner");
    probability = (double *)memory->srealloc(probability,nmax*sizeof(double),"bond/create/mca:probability");
  }

  int nlocal = atom->nlocal;
  int nall = atom->nlocal + atom->nghost;

#if defined (_OPENMP)
#pragma omp parallel for private(i,j) default(shared) schedule(static)
#endif
  for (i = 0; i < nall; i++) {
    for(j = 0; j< maxbondsperatom; j++) partner[i][j] = 0;
    npartner[i] = 0;
    probability[i] = 1.;
  }

  // loop over neighbors of my atoms
  // each atom sets one closest eligible partner atom ID to bond with

  double **x = atom->x;
  int *tag = atom->tag;
  int *mask = atom->mask;
  int *type = atom->type;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  flag = 0;

//if (logfile) fprintf(logfile,"FixBondCreateMCA::post_integrate: nall (%d) = atom->nlocal (%d) + atom->nghost (%d) inum=%d\n",nall,atom->nlocal,atom->nghost,inum);
///#if defined (_OPENMP)
///#pragma omp parallel for private(ii,jj,i,j,xtmp,ytmp,ztmp,delx,dely,delz) default(shared) schedule(static)
///#endif
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    if (!(mask[i] & groupbit)) continue;
    itype = type[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    jlist = firstneigh[i];
    jnum = numneigh[i];

//if(tag[i]==10) if (logfile) fprintf(logfile,"FixBondCreateMCA::post_integrate: list[%d]= %d tag= %d jnum= %d\n",ii,i,tag[i],jnum);

    for (jj = 0; jj < jnum; jj++) {

      j = jlist[jj];
//if(tag[i]==10) if (logfile) fprintf(logfile,"\t\tjlist[%d] = %d (j &= NEIGHMASK)=",jj,j);
      j &= NEIGHMASK;
//if(tag[i]==10) if (logfile) fprintf(logfile,"%d tag= %d\n",j,tag[j]);
      if (!(mask[j] & groupbit)) continue;
      jtype = type[j];

      possible = 0;
      if (itype == iatomtype && jtype == jatomtype) {
        if ((imaxbond == 0 || bondcount[i] < imaxbond) &&
            (jmaxbond == 0 || bondcount[j] < jmaxbond))
          possible = 1;
      } else if (itype == jatomtype && jtype == iatomtype) {
        if ((jmaxbond == 0 || bondcount[i] < jmaxbond) &&
            (imaxbond == 0 || bondcount[j] < imaxbond))
          possible = 1;
      }
      if (!possible) continue;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      double rsq = delx*delx + dely*dely + delz*delz;
      if (rsq >= cutsq) {
//if(tag[i]==10) if (logfile) fprintf(logfile,"FixBondCreateMCA::post_integrate\trsq (%g) >= cutsq (%g) btw atoms %d and %d \n",rsq,cutsq,i,j);
        continue;
      }

      if (init_state == UNBONDED) {
        double contsq = cont_distance[i] + cont_distance[j];
        contsq *= contsq;
        if (rsq > contsq) {
//          if (logfile) fprintf(logfile,"FixBondCreateMCA::post_integrate\trsq (%g) >= contsq (%g) btw atoms %d and %d \n",rsq,contsq,i,j);
          continue;
        }
      }

      if(already_bonded(i,j)) {
//        if (logfile) fprintf(logfile,"FixBondCreateMCA::post_integrate\tExisting bond btw atoms %d and %d \n",i,j);
        continue;
      }

      if(npartner[i]==maxbondsperatom || npartner[j]==maxbondsperatom) {
        if (logfile) fprintf(logfile,"FixBondCreateMCA::post_integrate\tnpartner[%d]=%d and npartner[%d]=%d Rij=%g\n",i,npartner[i],j,npartner[j],sqrt(rsq));
          flag = 1;
          continue;
      }

//if(tag[i]==10) if (logfile) fprintf(logfile,"\t\tPARTNER!!!\n");
//#pragma omp critical
//      {
      partner[i][npartner[i]] = tag[j];
      partner[j][npartner[j]] = tag[i];
//      }
//#pragma omp atomic
      npartner[i]++;
//#pragma omp atomic
      npartner[j]++;

    }
  }

  if(flag) error->warning(FLERR,"FixBondCreateMCA::post_integrate\tCould not generate all possible bonds");

  // reverse comm of distsq and partner
  // not needed if newton_pair off since I,J pair was seen by both procs

  commflag = 1;
  if (force->newton_pair) comm->reverse_comm_fix(this);

  // each atom now knows its partners
  // for prob check, generate random value for each atom with a bond partner
  // forward comm of partner and random value, so ghosts have it

  if (fraction < 1.0) {
#if defined (_OPENMP)
#pragma omp parallel for private(i) default(shared) schedule(static)
#endif
    for (i = 0; i < nlocal; i++)
      if (npartner[i]) probability[i] = random->uniform();
  }

  commflag = 1;
  comm->forward_comm_fix(this);

  // create bonds for atoms I own
  // if other atom is owned by another proc, it should create same bond
  // if both atoms list each other as bond partner
  // and probability constraint is satisfied

  int **bond_type = atom->bond_type;
  int **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;
//  int **nspecial = atom->nspecial;
//  int **special = atom->special;
  int newton_bond = force->newton_bond;
  int n_bondhist = atom->n_bondhist;
  double ***bond_hist = atom->bond_hist;
  double r,rinv;

  int ncreate = 0;
#if defined (_OPENMP)
#pragma omp parallel for private(i,j,k,xtmp,ytmp,ztmp,delx,dely,delz) default(shared) reduction(+:ncreate) schedule(static)
#endif
  for (i = 0; i < nlocal; i++) {
    if (npartner[i] == 0) {
      //if (logfile) fprintf(logfile,"FixBondCreateMCA::post_integrate: npartner[%d] == 0\n",i,npartner[i]);
      continue;
    }

    double min,max;
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    for(k = 0; k < npartner[i]; k++)
    {
        j = atom->map(partner[i][k]);
        if (j == -1) {
          char str[512];
          sprintf(str,
                "Bond atoms %d %d missing on proc %d at step " BIGINT_FORMAT,
                tag[i],bond_atom[i][k],me,update->ntimestep);
          error->one(FLERR,str);
        }
        j = domain->closest_image(i,j);
        if(tag[j] != partner[i][k]) {
            if (logfile) fprintf(logfile,"FixBondCreateMCA::post_integrate: for i=%d partner k=%d j=%d (tag=%d) != partner[i][k]=%d at step %ld\n",i,k,j,tag[j],partner[i][k],update->ntimestep);
            error->one(FLERR,"Error: tag[j] != partner in fix bond/create/mca");
        }

        int found = 0;
        for(int jp = 0; jp < npartner[j]; jp++)
            if(partner[j][jp] == tag[i]) found = 1;
        if (!found) {
           if (logfile) fprintf(logfile,"FixBondCreateMCA::post_integrate: i=%d(tag=%d) not found as a partner from %d for partner k=%d j=%d (tag=%d) at step %ld\n",i,tag[i],npartner[j],k,j,tag[j],update->ntimestep);
           for(int jp = 0; jp < npartner[j]; jp++) if (logfile) fprintf(logfile,"\t\tpartner[j][%d]= %d\n",jp,partner[j][jp]);
           error->all(FLERR,"Error: tag[i] not found in partners of j in fix bond/create/mca");
        }

        if(already_bonded(i,j)) {
          if (logfile) fprintf(logfile,"FixBondCreateMCA::post_integrate\tExisting bond btw atoms %d and %d \n",i,j);
          continue;
        }

        // apply probability constraint
        // MIN,MAX insures values are added in same order on different procs

        if (fraction < 1.0) {
          min = MIN(probability[i],probability[j]);
          max = MAX(probability[i],probability[j]);
          if (0.5*(min+max) >= fraction) {if (logfile) fprintf(logfile,"FixBondCreateMCA::post_integrate: for i=%d partner k=%d j=%d (>= fraction)\n",i,npartner[i],j); continue;}
        }

        // if newton_bond is set, only store with I or J
        // if not newton_bond, store bond with both I and J

        if (!newton_bond /*|| tag[i] < tag[j]*/)
        {
///if(tag[i]==10) 
///if (logfile) fprintf(logfile,"FixBondCreateMCA::post_integrate: creating bond btw atoms i=%d and j=%d (tag=%d) (i has now %d bonds) at step %ld\n",i,j,tag[j],num_bond[i]+1,update->ntimestep);

          if (num_bond[i] == atom->bond_per_atom) {
	    if (logfile) fprintf(logfile,"FixBondCreateMCA::post_integrate: for i=%d num_bond[i]=%d atom->bond_per_atom=%d (==)!!\n",i,num_bond[i],atom->bond_per_atom);
	    fprintf(stderr,"FixBondCreateMCA::post_integrate: for i=%d num_bond[i]=%d atom->bond_per_atom=%d (==)!!\n",i,num_bond[i],atom->bond_per_atom);
            error->one(FLERR,"New bond exceeded bonds per atom in fix bond/create/mca");
	  }
          bond_type[i][num_bond[i]] = btype;
          bond_atom[i][num_bond[i]] = tag[j];
          /*  print these lines
          std::cout << "if (!newton_bond || tag["<<i<<"] (="<<tag[i]<<") < tag["<<j<<"] (="<<tag[j]<<")) "<<std::endl; // NP P.F. correct this okt-29
          std::cout << "if (num_bond["<<i<<"] (="<<num_bond[i]<<")== atom->bond_per_atom(="<<atom->bond_per_atom<<"))  "<<std::endl; 
          std::cout << "bond_type["<<i<<"]["<<num_bond[i]<<"] = btype (="<<btype<<");"<<std::endl; 
          std::cout << "bond_atom["<<i<<"]["<<num_bond[i]<<"] = tag["<<j<<"] (="<<tag[j]<<");"<<std::endl; 
          */

          //reset history
          double *tmp = bond_hist[tag[i]-1][num_bond[i]];
          for (int ih = 0; ih < n_bondhist; ih++) {
              tmp[ih] = 0.;
          }

          delx = xtmp - x[j][0];
          dely = ytmp - x[j][1];
          delz = ztmp - x[j][2];
          r = sqrt(delx*delx + dely*dely + delz*delz);
//if(tag[i]==10) if (logfile) fprintf(logfile,"FixBondCreateMCA::post_integrate: distance btw atoms i=%d and j=%d = %g  (%g %g %g)\n",i,j,r,x[j][0],x[j][1],x[j][2]);
          rinv = -1. / r; // "-" means that unit vector is from i1 to i2
          tmp[R] = tmp[R_PREV] = r;
          tmp[NX_PREV] = tmp[NX] = delx * rinv;
          tmp[NY_PREV] = tmp[NY] = dely * rinv;
          tmp[NZ_PREV] = tmp[NZ] = delz * rinv;

          tmp[TAG] = double(tag[j]);
          tmp[STATE] = double(init_state);
          num_bond[i]++;
        }
        // increment bondcount, convert atom to new type if limit reached

        bondcount[i]++;
        if (type[i] == iatomtype) {
          if (bondcount[i] == imaxbond) type[i] = inewtype; // bondtype defined by user in input script = iatomtype
        } else {
          if (bondcount[i] == jmaxbond) type[i] = jnewtype;
        }

        // count the created bond once

        if (tag[i] < tag[j]) ncreate++;
    }
  }

  // need updated bonds AS

  comm->forward_comm();

  // tally stats

  MPI_Allreduce(&ncreate,&createcount,1,MPI_INT,MPI_SUM,world);
  createcounttotal += createcount;
  atom->nbonds += createcount;

  if(createcount && comm->me == 0) if (logfile) fprintf(logfile,"Created %d unique bonds at timestep " BIGINT_FORMAT "\n",createcount,update->ntimestep);

  // trigger reneighboring if any bonds were formed

  if (createcount) next_reneighbor = update->ntimestep;

}

inline bool FixBondCreateMCA::already_bonded(int i,int j)
{
    for(int k = 0; k < atom->num_bond[i]; k++)
      if(atom->bond_atom[i][k] == atom->tag[j]) return true;
    return false;
}

/* ---------------------------------------------------------------------- */

void FixBondCreateMCA::post_integrate_respa(int ilevel, int iloop)
{
  if (ilevel == nlevels_respa-1) post_integrate();
}

/* ---------------------------------------------------------------------- */

int FixBondCreateMCA::pack_comm(int n, int *list, double *buf,
                             int pbc_flag, int *pbc)
{
  int i,j,m;

  m = 0;

  if (commflag == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = ubuf(bondcount[j]).d; ///static_cast<int>(bondcount[j]);
    }
//if (stderr) fprintf(stderr,"FixBondCreateMCA::pack_comm m=%d n=%d [%d - %d]\n",m,n,list[0],list[n-1]);
    return 1;
  } else {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = ubuf(npartner[j]).d; ///static_cast<int>(npartner[j]);
      for(int k=0; k<maxbondsperatom; k++) //NP communicate all slots, also the empty ones
        buf[m++] = ubuf(partner[j][k]).d; ///static_cast<int>(partner[j][k]);
      buf[m++] = probability[j];
    }
//if (stderr) fprintf(stderr,"FixBondCreateMCA::pack_comm m=%d n=%d [%d - %d]\n",m,n,list[0],list[n-1]);
    return comm_forward;//maxbondsperatom + 2; //m; for last versions of lammps !!!
  }
}

/* ---------------------------------------------------------------------- */

void FixBondCreateMCA::unpack_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;

  if (commflag == 0) {
    for (i = first; i < last; i++)
      bondcount[i] = (int) ubuf(buf[m++]).i;///static_cast<int> (buf[m++]);

  } else {
    for (i = first; i < last; i++) {
      npartner[i] = (int) ubuf(buf[m++]).i;///static_cast<int> (buf[m++]);
      for(int k=0; k<maxbondsperatom; k++)
        partner[i][k] = (int) ubuf(buf[m++]).i;///static_cast<int>(buf[m++]);
      probability[i] = buf[m++];
    }
  }
//if (stderr) fprintf(stderr,"FixBondCreateMCA::unpack_comm m=%d n=%d [%d - %d]\n",m,n,first,last);
}

/* ---------------------------------------------------------------------- */

int FixBondCreateMCA::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;

  if (commflag == 0) {
    for (i = first; i < last; i++)
      buf[m++] = ubuf(bondcount[i]).d; ///bondcount[i];
///if (logfile) fprintf(logfile,"FixBondCreateMCA::pack_reverse_comm m=%d n=%d [%d - %d]\n",m,n,first,last);
    return 1;
  } else {
    for (i = first; i < last; i++) {
      buf[m++] = ubuf(npartner[i]).d; ///static_cast<int>(npartner[i]);
      for(int k=0; k<maxbondsperatom; k++)
        buf[m++] = ubuf(partner[i][k]).d; ///static_cast<int>(partner[i][k]);
    }
///if (logfile) fprintf(logfile,"FixBondCreateMCA::pack_reverse_comm m=%d n=%d [%d - %d]\n",m,n,first,last);
    return comm_reverse; //maxbondsperatom + 1;  //m; for last versions of lammps !!!
  }
}

/* ---------------------------------------------------------------------- */

void FixBondCreateMCA::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  int flag = 0;
  int nnew;

  if (commflag == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      bondcount[j] += (int) ubuf(buf[m++]).i;///static_cast<int> (buf[m++]);
    }
  } else {
    //NP add new bonds coming from other proc
    for (i = 0; i < n; i++) {
      j = list[i];
      nnew = (int) ubuf(buf[m++]).i;///static_cast<int> (buf[m++]);
      if(nnew+npartner[j] > maxbondsperatom)
      {
          flag = 1;
          nnew = maxbondsperatom - npartner[j];
      }
      for(int k = npartner[j]; k < npartner[j]+maxbondsperatom; k++)
      {
          if(k >= npartner[j]+nnew) m++; //NP do not do anything if
          else partner[j][k] = (int) ubuf(buf[m++]).i;///static_cast<int> (buf[m++]);
      }
      npartner[i] += nnew;
    }
  }
  if(flag) error->warning(FLERR,"Could not generate all possible bonds");
///if (logfile) fprintf(logfile,"FixBondCreateMCA::unpack_reverse_comm m=%d n=%d [%d - %d]\n",m,n,list[0],list[n-1]);
}

/* ----------------------------------------------------------------------
   allocate local atom-based arrays
------------------------------------------------------------------------- */

void FixBondCreateMCA::grow_arrays(int nmax)
{
  bondcount = (int *)
    memory->srealloc(bondcount,nmax*sizeof(int),"bond/create/mca:bondcount");
}

/* ----------------------------------------------------------------------
   copy values within local atom-based arrays
------------------------------------------------------------------------- */

void FixBondCreateMCA::copy_arrays(int i, int j)
{
  bondcount[j] = bondcount[i];
}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for exchange with another proc
------------------------------------------------------------------------- */

int FixBondCreateMCA::pack_exchange(int i, double *buf)
{
  buf[0] = bondcount[i];
  return 1;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based arrays from exchange with another proc
------------------------------------------------------------------------- */

int FixBondCreateMCA::unpack_exchange(int nlocal, double *buf)
{
  bondcount[nlocal] = (int) ubuf(buf[0]).i;///static_cast<int> (buf[0]);
  return 1;
}

/* ---------------------------------------------------------------------- */

double FixBondCreateMCA::compute_vector(int n)
{
  if (n == 1) return (double) createcount;
  return (double) createcounttotal;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixBondCreateMCA::memory_usage()
{
  int nmax = atom->nmax;
  double bytes = nmax*2 * sizeof(int);
  bytes += maxbondsperatom*nmax * sizeof(int);
  bytes += nmax * sizeof(double);
  return bytes;
}
