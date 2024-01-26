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
    This file is from LAMMPS
    LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
    http://lammps.sandia.gov, Sandia National Laboratories
    Steve Plimpton, sjplimp@sandia.gov

    Copyright (2003) Sandia Corporation.  Under the terms of Contract
    DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
    certain rights in this software.  This software is distributed under
    the GNU General Public License.
------------------------------------------------------------------------- */

#include "string.h"
#include "compute_property_mca.h"
#include "force.h"
#include "pair_mca.h"
#include "atom.h"
#include "update.h"
//#include "domain.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputePropertyMCA::
ComputePropertyMCA(LAMMPS *lmp, int &iarg, int narg, char **arg) :
  Compute(lmp, iarg, narg, arg)
{
/*  
<PRE>compute ID group-ID property/mca input1 input2 ... 
</PRE>
<UL>
    <LI>ID, group-ID are documented in <A HREF = "compute.html">compute</A> command 
    <LI>property/mca = style name of this compute command 
    <LI>input = one or more attributes 
  <PRE>  possible attributes = mean_stress eq_stress mean_strain eq_strain
    mean_stress = mean (volume) stress
    eq_stress = equivalent (von Mises) stress
    mean_strain = mean (volume) strian
    eq_strain = equivalent (von Mises) strain
  </PRE>
</UL>
*/
  if (narg < iarg+1) error->all(FLERR,"Illegal compute property/mca command");

  if ((atom->molecular == 0) || (atom->bond_mca == 0))
    error->all(FLERR,"Compute property/mca requires mca atom style");

  peratom_flag = 1;
  nvalues = narg - iarg;
  if (nvalues == 1) size_peratom_cols = 0;
  else size_peratom_cols = nvalues;

  pack_choice = new FnPtrPack[nvalues];

  // parse input values
  // customize a new keyword by adding to if statement
  int i;
  const int arg_offset = iarg;
  for (; iarg < narg; iarg++) {
    i = iarg-arg_offset;

    if (strcmp(arg[iarg],"mean_stress") == 0)
      pack_choice[i] = &ComputePropertyMCA::pack_meanstress;
    else if (strcmp(arg[iarg],"eq_stress") == 0)
      pack_choice[i] = &ComputePropertyMCA::pack_eqstress;
    else if (strcmp(arg[iarg],"mean_strain") == 0)
      pack_choice[i] = &ComputePropertyMCA::pack_meanstrain;
    else if (strcmp(arg[iarg],"eq_strain") == 0)
      pack_choice[i] = &ComputePropertyMCA::pack_eqstrain;
    else error->all(FLERR,"Invalid keyword in compute property/mca command");
  }

  nmax = 0;
  vector = NULL;
  array = NULL;
}

/* ---------------------------------------------------------------------- */

ComputePropertyMCA::~ComputePropertyMCA()
{
  delete [] pack_choice;
  memory->destroy(vector);
  memory->destroy(array);
}

/* ---------------------------------------------------------------------- */

void ComputePropertyMCA::init()
{
}

/* ---------------------------------------------------------------------- */

void ComputePropertyMCA::compute_peratom()
{
  invoked_peratom = update->ntimestep;

  // grow vector or array if necessary

  if (atom->nlocal > nmax) {
    nmax = atom->nmax;
    if (nvalues == 1) {
      memory->destroy(vector);
      memory->create(vector,nmax,"property/mca:vector");
      vector_atom = vector;
    } else {
      memory->destroy(array);
      memory->create(array,nmax,nvalues,"property/mca:array");
      array_atom = array;
    }
  }

  // fill vector or array with per-atom values

  if (nvalues == 1) {
    buf = vector;
    (this->*pack_choice[0])(0);
  } else {
    if (nmax) buf = &array[0][0];
    else buf = NULL;
    for (int n = 0; n < nvalues; n++)
      (this->*pack_choice[n])(n);
  }
}

/* ----------------------------------------------------------------------
   memory usage of local data
------------------------------------------------------------------------- */

double ComputePropertyMCA::memory_usage()
{
  double bytes = nmax*nvalues * sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   one method for every keyword compute property/molecule can output
   the atom property is packed into buf starting at n with stride nvalues
   customize a new keyword by adding a method
------------------------------------------------------------------------- */

void ComputePropertyMCA::pack_meanstress(int n)
{
  double *mean_stress = atom->mean_stress;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = mean_stress[i];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyMCA::pack_eqstress(int n)
{
  double *equiv_stress = atom->equiv_stress;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = equiv_stress[i];
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyMCA::pack_meanstrain(int n)
{
  double *mean_stress = atom->mean_stress;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  const int * const type = atom->type;
  const PairMCA * const mca_pair = (PairMCA*) force->pair;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      int itype = type[i];
      double r3Ki = 3.0 * mca_pair->K[itype][itype];
      buf[n] = mean_stress[i] / r3Ki;
    }
    else buf[n] = 0.0;
    n += nvalues;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePropertyMCA::pack_eqstrain(int n)
{
  double *equiv_strain = atom->equiv_strain;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) buf[n] = equiv_strain[i];
    else buf[n] = 0.0;
    n += nvalues;
  }
}
