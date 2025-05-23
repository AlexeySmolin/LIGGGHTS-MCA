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
#include "fix_bond_exchange_mca.h"
#include "update.h"
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
#include <list>
#include "atom_vec_mca.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MCAAtomConst;

#define BIG 1.0e20

/* ---------------------------------------------------------------------- */

FixBondExchangeMCA::FixBondExchangeMCA(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
//    if (logfile) fprintf(logfile,"constructor FixBondExchangeMCA ###########\n");
    restart_global = 1;
    laststep=-1;
}

/* ---------------------------------------------------------------------- */

FixBondExchangeMCA::~FixBondExchangeMCA()
{
}

/* ---------------------------------------------------------------------- */

int FixBondExchangeMCA::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  return mask;
}

void FixBondExchangeMCA::pre_exchange()
{
  int i1,i2,n;
  bool found;
  int **bondlist = neighbor->bondlist;
  double **bondhistlist = neighbor->bondhistlist;
  int nbondlist = neighbor->nbondlist;

  int **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;
  double ***bond_hist = atom->bond_hist;
  int n_bondhist = atom->n_bondhist;
  int nlocal = atom->nlocal;
  int *tag = atom->tag;

  int newton_bond = force->newton_bond;

///if (logfile) fprintf(logfile,"FixBondExchangeMCA::pre_exchange\n");
///fprintf(stderr,"FixBondExchangeMCA::pre_exchange\n");

  /* task 1: propagate bond contact history  AS - we do not need this!!!
  if (!n_bondhist) return;

  for (n = 0; n < nbondlist; n++) {
    if(bondlist[n][3] == NOT_INTERACT) continue; //do not copy broken bonds

    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    if (newton_bond || i1 < nlocal)
    {
        found = false;
        for(int k = 0; k < num_bond[i1]; k++)
           if(bond_atom[i1][k] == tag[i2])
           {
              found = true;
              for(int q = 0; q < n_bondhist; q++) bond_hist[i1][k][q] = bondhistlist[n][q];
              break;
           }
        if(!found) error->one(FLERR,"Failed to operate on MCA bond history during copy i1");
    }

    if (newton_bond || i2 < nlocal)
    {
        found = false;
        for(int k = 0; k < num_bond[i2]; k++)
           if(bond_atom[i2][k] == tag[i1])
           {
              found = true;
              for(int q = 0; q < n_bondhist; q++) bond_hist[i2][k][q] = -bondhistlist[n][q];
              break;
           }
        if(!found) error->one(FLERR,"Failed to operate on MCA bond history during copy i2");

    }
  }
*/
  // task 2: remove broken bonds
  // should be done equally on all processors

  // create list for all BOND-IDs, erase all the broken bonds compress bondlist with new value list
  std::vector<unsigned int> list_bond_id;
  std::vector<unsigned int>::iterator it1;

  for (n=0; n<nbondlist; n++) 
      if(bondlist[n][3] == NOT_INTERACT) list_bond_id.push_back(n);// load all broken bond ids into the list

  //DEBUG
  if (list_bond_id.size()>0) if (logfile) fprintf(logfile,"FixBondExchangeMCA::pre_exchange will delete %ld broken bonds at step %ld\n",list_bond_id.size(),update->ntimestep);

  for (it1=list_bond_id.begin(); it1!=list_bond_id.end(); ++it1)
  {
    n = *it1;
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];

    if(bondlist[n][3] < NOT_INTERACT) continue;

    if (logfile) fprintf(logfile,"FixBondExchangeMCA::pre_exchange detected bond %d:%d(tag=%d)<->%d(tag=%d) as broken at step %ld\n",n,i1,atom->tag[i1],i2,atom->tag[i2],update->ntimestep);
    // if the bond is broken, we remove it from both atom data

    // delete bond from atom I if I stores it
    // atom J will also do this

    if (newton_bond || i1 < nlocal) {
        found = false;
        for(int k = 0; k < num_bond[i1]; k++)
           if(bond_atom[i1][k] == tag[i2]) {
             found = true;
             remove_bond(i1,k,n);
             break;
           }
        if(!found) error->one(FLERR,"Failed to operate on MCA bond history during deletion1");
    }
    if (newton_bond || i2 < nlocal) {
        found = false;
        for(int k = 0; k < num_bond[i2]; k++)
           if(bond_atom[i2][k] == tag[i1]) {
             found = true;
             remove_bond(i2,k,n);
             int nbondlist = neighbor->nbondlist;
             if(n<nbondlist) {
               neighbor->nbondlist = (nbondlist-1);                               // delete one bond -> reduce nbondlist by one
               for(int i = 0; i <= 3; i++)
                  neighbor->bondlist[n][i] = neighbor->bondlist[nbondlist-1][i];  // also change in neighbor bondlist
               break;
             } else if(n == nbondlist) {
               neighbor->nbondlist = (nbondlist-1);
               break;
             }
             break;
           }
        if(!found) error->one(FLERR,"Failed to operate on MCA bond history during deletion2");
    }
  }
}

inline void FixBondExchangeMCA::remove_bond(int ilocal, int ibond, int bondnumber)
{
    const int * const tag = atom->tag;
    int *bond_atom = atom->bond_atom[ilocal];
    int *bond_type = atom->bond_type[ilocal];
    int *bond_mca = atom->bond_mca[ilocal];
    double **bond_hist = atom->bond_hist[tag[ilocal] - 1];
    if (logfile) fprintf(logfile,"FixBondExchangeMCA::remove_bond #%d : ilocal=%d ibond=%d global=%d\n",bondnumber,ilocal,ibond,atom->map(bond_atom[ibond]));
    ///error->one(FLERR,"romoving bond");
    int nbond = atom->num_bond[ilocal] - 1;
    bond_atom[ibond] = bond_atom[nbond];
    bond_mca[ibond] = bond_mca[nbond];
    bond_type[ibond] = bond_type[nbond];
    for(int k = 0; k < atom->n_bondhist; k++) bond_hist[ibond][k] = bond_hist[nbond][k];
    atom->num_bond[ilocal]--;
}

/* ----------------------------------------------------------------------
   pack entire state of Fix into one write
------------------------------------------------------------------------- */

void FixBondExchangeMCA::write_restart(FILE *fp)
{
  //error->warning(FLERR,"Restart functionality not yet tested for MCA bonds...");

  //NP write a dummy value
  int n = 0;
  double list[1];
  list[n++] = 1.;

  if (comm->me == 0) {
    int size = n * sizeof(double);
    fwrite(&size,sizeof(int),1,fp);
    fwrite(list,sizeof(double),n,fp);
  }

  //NP write data to atom arrays where it can then be stored
  //NP can be done this way bc modify writes before avec
  pre_exchange();
}

/* ----------------------------------------------------------------------
   use state info from restart file to restart the Fix
------------------------------------------------------------------------- */

void FixBondExchangeMCA::restart(char *buf)
{
  int n = 0;
  double *list = (double *) buf;

  double dummy = static_cast<int> (list[n++]);

  //error->warning(FLERR,"Restart functionality not yet tested for MCA bonds...");
}
