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
#include "string.h"
#include "fix_nve_mca.h"
#include "atom.h"
#include "atom_vec.h"
#include "update.h"
#include "respa.h"
#include "force.h"
#include "error.h"
#include "domain.h" 

using namespace LAMMPS_NS;
using namespace FixConst;

#define INERTIA 0.4          // moment of inertia prefactor for sphere

enum{NONE,DIPOLE};

/* ---------------------------------------------------------------------- */

FixNVEMCA::FixNVEMCA(LAMMPS *lmp, int narg, char **arg) :
  FixNVE(lmp, narg, arg),
  useAM_(false),
  CAddRhoFluid_(0.0),
  onePlusCAddRhoFluid_(1.0)
{
  if (narg < 3) error->all(FLERR,"Illegal fix nve/mca command");

  time_integrate = 1;

  // process extra keywords

  extra = NONE;

  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"update") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix nve/mca command");
      if (strcmp(arg[iarg+1],"dipole") == 0) extra = DIPOLE;
      else if (strcmp(arg[iarg+1],"CAddRhoFluid") == 0)
      {
            if(narg < iarg+2)
                error->fix_error(FLERR,this,"not enough arguments for 'CAddRhoFluid'");
            iarg+=2;
            useAM_ = true;
            CAddRhoFluid_        = atof(arg[iarg]);
            onePlusCAddRhoFluid_ = 1.0 + CAddRhoFluid_;
            fprintf(screen,"cfd_coupling_force_implicit will consider added mass with CAddRhoFluid = %f\n",
                    CAddRhoFluid_);
      }
      else error->all(FLERR,"Illegal fix nve/mca command");
      iarg += 2;
    } else error->all(FLERR,"Illegal fix nve/mca command");
  }

  // error checks

  if (!atom->mca_flag)
    error->all(FLERR,"Fix nve/mca requires atom style mca");
  /*!! //AS
  if (extra == DIPOLE && !atom->mu_flag) 
    error->all(FLERR,"Fix nve/mca requires atom attribute mu"); */
}

/* ---------------------------------------------------------------------- */

void FixNVEMCA::init()
{
  FixNVE::init();

  // check that all particles are atom_vec_mca

  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  if (!(atom->mca_flag))
    error->one(FLERR,"Fix nve/mca requires atom_vec_mca particles");

/*  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit)
      if (!mca_flag[i])
        error->one(FLERR,"Fix nve/mca requires atom_vec_mca particles");*/
}

/* ---------------------------------------------------------------------- */

void FixNVEMCA::initial_integrate(int vflag)
{
  double dtfm,dtirotate,msq,scale;

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double **omega = atom->omega;
  double **torque = atom->torque;
  double *rmass = atom->rmass;
  double *inertia = atom->q;
  double **theta = atom->mu;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  // set timestep here since dt may have changed or come via rRESPA

  if (domain->dimension != 3) {
   error->one(FLERR,"Fix nve/mca is impllemented only for 3D");
  }

  // update 1/2 step for v and omega, and full step for  x for all particles
  // d_omega/dt = torque / inertia
//fprintf(stderr, "FixNVEMCA::initial_integrate dtv= %g dtf= %g\n",dtv,dtf);      

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {

      // velocity update for 1/2 step
      dtfm = dtf / (rmass[i]*onePlusCAddRhoFluid_);
//fprintf(stderr, "rmass[%d]= %20.12e onePlusCAddRhoFluid_= %20.12e f= %20.12e %20.12e %20.12e \n",i,rmass[i],onePlusCAddRhoFluid_,f[i][0],f[i][1],f[i][2]);      
      v[i][0] += dtfm * f[i][0];
      v[i][1] += dtfm * f[i][1];
      v[i][2] += dtfm * f[i][2];

//fprintf(stderr, "coord[%d]= %20.12e %20.12e %20.12e velo= %20.12e %20.12e %20.12e \n",i,x[i][0],x[i][1],x[i][2],v[i][0],v[i][1],v[i][2]);      
      // position update
      x[i][0] += dtv * v[i][0];
      x[i][1] += dtv * v[i][1];
      x[i][2] += dtv * v[i][2];
//fprintf(stderr, "coord[%d]= %20.12e %20.12e %20.12e\n",i,x[i][0],x[i][1],x[i][2]);      
      // rotation update
      dtirotate = dtf / (inertia[i]);
      omega[i][0] += dtirotate * torque[i][0];
      omega[i][1] += dtirotate * torque[i][1];
      omega[i][2] += dtirotate * torque[i][2];

      // update rotation 
      ///AS TODO Here it is for small roations as vector, Must be done for quaternions
      theta[i][0] += dtv * omega[i][0];
      theta[i][1] += dtv * omega[i][1];
      theta[i][2] += dtv * omega[i][2];
    }
  }

}

/* ---------------------------------------------------------------------- */

void FixNVEMCA::final_integrate()
{
  double dtfm,dtirotate;

  double **v = atom->v;
  double **f = atom->f;
  double **omega = atom->omega;
  double **torque = atom->torque;
  double *rmass = atom->rmass;
  double *inertia = atom->q;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  // update 1/2 step for v,omega for all particles
  // d_omega/dt = torque / inertia
//fprintf(stderr, "FixNVEMCA::final_integrate \n");      

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {

      // velocity update for 1/2 step
      dtfm = dtf / (rmass[i]*onePlusCAddRhoFluid_);
      v[i][0] += dtfm * f[i][0];
      v[i][1] += dtfm * f[i][1];
      v[i][2] += dtfm * f[i][2];
//fprintf(stderr, "velo[%d]= %20.12e %20.12e %20.12e\n",i,v[i][0],v[i][1],v[i][2]);      

      // rotation update
      dtirotate = dtf / (inertia[i]);
      omega[i][0] += dtirotate * torque[i][0];
      omega[i][1] += dtirotate * torque[i][1];
      omega[i][2] += dtirotate * torque[i][2];
   }
}
