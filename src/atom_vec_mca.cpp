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
#include "stdlib.h"
#include "string.h"
#include "atom_vec_mca.h"
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "modify.h"
#include "force.h"
#include "fix.h"
#include "fix_adapt.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include "domain_wedge.h"
#include "update.h"

using namespace LAMMPS_NS;
using namespace MathConst;

#define DELTA 10000
#define MAX_BONDS 30

enum{NONE,SC,BCC,FCC,HCP,DIAMOND,SQ,SQ2,HEX,CUSTOM};///AS taken from 'lattice.c'

using namespace MCAAtomConst;

/* ---------------------------------------------------------------------- */

AtomVecMCA::AtomVecMCA(LAMMPS *lmp) : AtomVec(lmp)
{
  molecular = 1;  //!! This allows storing 'bonds,angles,dihedrals,improper'. Do we need them all? in AtomVecEllipsoid==0  
  bonds_allow = 1;
  mass_type = 1; // per-type masses

  comm_x_only = 0;   // 1 if only exchange x in forward comm - we need rotation and mean stress
  comm_f_only = 0;   // 1 if only exchange f in reverse comm - we need torque

///AS TODO What is border communications? What is forward and reverse communications? What we need for them?
  size_forward = 9;  ///AS TODO # of values per atom in comm !! Later choose what to pass via MPI
  size_reverse = 6;  // # in reverse comm
  size_border = 22+MAX_BONDS*(2+BOND_HIST_LEN);  // # in border comm (periodic boundary ?)
  size_velocity = 6; ///AS # of velocity based quantities
  size_data_atom = 7;///AS TODO number of values in Atom line
  size_data_vel = 7; ///AS TODO number of values in Velocity line
  xcol_data = 5;     ///AS TODO column (1-N) where x is in Atom line

  atom->molecule_flag = 1; // this allows compute bonds

  atom->mca_flag = 1; //!! mca uses this flag
  atom->radius_flag = 0; ///AS mca uses the same radius for all particles
  atom->density_flag = 0; ///AS if 1 then in set density we get 'rmass=density'
  atom->rmass_flag = atom->omega_flag = atom->torque_flag = 1;
  atom->implicit_factor = IMPLFACTOR;
  atom->n_bondhist = BOND_HIST_LEN;

  fbe = NULL; //!! delete in destructor
}

AtomVecMCA::~AtomVecMCA()
{
  //if(fbe != NULL) {
  //  delete fbe; //!! It is created in AtomVecMCA::init() by calling modify->add_fix()
  //}
}

/* ---------------------------------------------------------------------- */

void AtomVecMCA::settings(int narg, char **arg)
{
// atom_style mca radius 0.0001 packing fcc n_bondtypes 1 bonds_per_atom 6 implicit_factor 0.5

  if (narg == 0) return;	//in case of restart no arguments are given, instead they are defined by read_restart_settings
  if ((narg < 8) || (narg > 10)) error->all(FLERR,"Invalid atom_style mca command, expecting exactly 8 arguments");

  if(strcmp(arg[0],"radius")) // 
    error->all(FLERR,"Illegal atom_style mca command, expecting 'radius'");

  mca_radius = atom->mca_radius = atof(arg[1]);
fprintf(logfile, "atom->mca_radius= %g  arg[1] '%s' \n", atom->mca_radius, arg[1]);  ///AS DEBUG

  if(strcmp(arg[2],"packing")) // 
    error->all(FLERR,"Illegal atom_style mca command, expecting 'packing'");

  if(!strcmp(arg[3],"sc")) {
    packing = atom->packing = SC;
    coord_num = atom->coord_num = 6;
  } else if(!strcmp(arg[3],"fcc")){
    packing = atom->packing = FCC;
    coord_num = atom->coord_num = 12;
  } else if(!strcmp(arg[3],"hcp")) {
    packing = atom->packing = HCP;
    coord_num = atom->coord_num = 12;
  } else {
    error->all(FLERR,"Illegal atom_style mca command, first parameter (packing) should be 'sc' or 'fcc' or 'hcp'");
  }
  contact_area = atom->contact_area = get_contact_area();

  if(strcmp(arg[4],"n_bondtypes")) // The number of bond types (different materials of the system)
    error->all(FLERR,"Illegal atom_style mca command, expecting 'n_bondtypes'");

  atom->nbondtypes = atoi(arg[5]); // we need types as the number of different materials of the system

  if(strcmp(arg[6],"bonds_per_atom")) // The maximum number of bonds that each atom can have (== packing: 6 - cubic, 12 - fcc)
    error->all(FLERR,"Illegal atom_style mca command, expecting 'bonds_per_atom'");

  atom->bond_per_atom = atoi(arg[7]);
//fprintf(logfile, "atom->bond_per_atom= %d < atom->coord_num %d atom->packing= %d arg[3] '%s' \n", atom->bond_per_atom, atom->coord_num, atom->packing, arg[3]);
  if (atom->bond_per_atom < atom->coord_num) 
    error->all(FLERR,"Illegal atom_style mca command, 'bonds_per_atom' must be >= coordination number for packing");
  if (atom->bond_per_atom > MAX_BONDS)
    error->all(FLERR,"Illegal atom_style mca command, 'bonds_per_atom' > MAX_BONDS");

  if (narg > 8) {
    if(strcmp(arg[8],"implicit_factor")) //  Implicit factor used to make integration scheme stable for larger time steps
      error->all(FLERR,"Illegal atom_style mca command, expecting 'implicit_factor'");
    if(narg == 10) atom->implicit_factor = atof(arg[9]);
  }
}

void AtomVecMCA::write_restart_settings(FILE *fp)
{
  fwrite(&atom->mca_radius,sizeof(double),1,fp);//!?? In other types this function is used only for neighbours  
  fwrite(&atom->packing,sizeof(int),1,fp);//!??
  fwrite(&atom->coord_num,sizeof(int),1,fp);//!??
  fwrite(&atom->nbondtypes,sizeof(int),1,fp);
  fwrite(&atom->bond_per_atom,sizeof(int),1,fp);
  fwrite(&atom->implicit_factor,sizeof(double),1,fp);//!?? In other types this function is used only for neighbours  
}

void AtomVecMCA::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    fread(&atom->mca_radius,sizeof(double),1,fp);//!??
    fread(&atom->packing,sizeof(int),1,fp);//!??
    fread(&atom->coord_num,sizeof(int),1,fp);//!??
    fread(&atom->nbondtypes,sizeof(int),1,fp);
    fread(&atom->bond_per_atom,sizeof(int),1,fp);
    fread(&atom->implicit_factor,sizeof(double),1,fp);
  }
  MPI_Bcast(&atom->mca_radius,1,MPI_DOUBLE,0,world);//!??
  MPI_Bcast(&atom->packing,1,MPI_INT,0,world);//!??
  MPI_Bcast(&atom->coord_num,1,MPI_INT,0,world);//!??
  MPI_Bcast(&atom->nbondtypes,1,MPI_INT,0,world);
  MPI_Bcast(&atom->bond_per_atom,1,MPI_INT,0,world);
  MPI_Bcast(&atom->implicit_factor,1,MPI_DOUBLE,0,world);//!??
  mca_radius = atom->mca_radius;
  packing = atom->packing;
  coord_num = atom->coord_num;
  contact_area = atom->contact_area = get_contact_area();
  implicit_factor = atom->implicit_factor;
}

/* -----!!!!!!!--------------------------------------------------------- */

int AtomVecMCA::pack_comm_vel_wedge(int n, int *list, double *buf,
			       int pbc_flag, int *pbc)
{
  return 0;
}

/* ----------------!!!!!!------------------------------------------------ */

int AtomVecMCA::pack_border_vel_wedge(int n, int *list, double *buf,
				 int pbc_flag, int *pbc)
{
  return 0;
}

void AtomVecMCA::init()
{
  AtomVec::init();

//  comm_x_only = 0; ///TODO !! See in constructor
//  size_forward = 9; ///TODO !! See in constructor

  // set radvary if particle diameters are time-varying due to fix adapt
  //radvary = 0;
  /* ///AS We do not need it, at least for now
  // set radvary if particle diameters are time-varying due to some fix
  for (int i = 0; i < modify->nfix; i++)
  {
      if (modify->fix[i]->rad_mass_vary_flag) {
        radvary = 1;
        size_forward = 7; 
        comm_x_only = 1;
      }
  }

  if(radvary) atom->radvary_flag = 1;
  */

  if(fbe == NULL)
  {
      char **fixarg = new char*[3];
      fixarg[0] = strdup("exchange_bonds_mca");
      fixarg[1] = strdup("all");
      fixarg[2] = strdup("bond/exchange/mca");
      modify->add_fix(3,fixarg);
      fbe = (FixBondExchangeMCA*) modify->find_fix_id(fixarg[0]);
      free(fixarg[0]);
      free(fixarg[1]);
      free(fixarg[2]);
      delete [] fixarg;
  }
}

/* ----------------------------------------------------------------------
   grow atom arrays
   n = 0 grows arrays by DELTA
   n > 0 allocates arrays to size n
------------------------------------------------------------------------- */

void AtomVecMCA::grow(int n)
{
  if (n == 0) nmax += DELTA;
  else nmax = n;
  atom->nmax = nmax;
  if (nmax < 0 || nmax > MAXSMALLINT)
    error->one(FLERR,"Per-processor system is too big");

  tag = memory->grow(atom->tag,nmax,"atom:tag");
  type = memory->grow(atom->type,nmax,"atom:type");
  mask = memory->grow(atom->mask,nmax,"atom:mask");
  image = memory->grow(atom->image,nmax,"atom:image");
  x = memory->grow(atom->x,nmax,3,"atom:x");
  v = memory->grow(atom->v,nmax,3,"atom:v");
  f = memory->grow(atom->f,nmax*comm->nthreads,3,"atom:f");

  density = memory->grow(atom->density,nmax,"atom:density");
  rmass = memory->grow(atom->rmass,nmax,"atom:rmass");
  omega = memory->grow(atom->omega,nmax,3,"atom:omega");
  torque = memory->grow(atom->torque,nmax*comm->nthreads,3,"atom:torque");

  mca_inertia = memory->grow(atom->mca_inertia,nmax,"atom:mca_inertia");
  theta = memory->grow(atom->theta,nmax,3,"atom:theta");
  theta_prev = memory->grow(atom->theta_prev,nmax,3,"atom:theta_prev");

  mean_stress = memory->grow(atom->mean_stress,nmax*comm->nthreads,"atom:mean_stress");
  mean_stress_prev = memory->grow(atom->mean_stress_prev,nmax*comm->nthreads,"atom:mean_stress_prev");
  equiv_stress = memory->grow(atom->equiv_stress,nmax*comm->nthreads,"atom:equiv_stress");
  equiv_stress_prev = memory->grow(atom->equiv_stress_prev,nmax*comm->nthreads,"atom:equiv_stress_prev");
  equiv_strain = memory->grow(atom->equiv_strain,nmax*comm->nthreads,"atom:equiv_strain");
  cont_distance = memory->grow(atom->cont_distance,nmax*comm->nthreads,"atom:cont_distance");

  molecule = memory->grow(atom->molecule,nmax,"atom:molecule");
  nspecial = memory->grow(atom->nspecial,nmax,3,"atom:nspecial");
  special = memory->grow(atom->special,nmax,atom->maxspecial,"atom:special");
  num_bond = memory->grow(atom->num_bond,nmax,"atom:num_bond");

  if(0 == atom->bond_per_atom)
    error->all(FLERR,"mca atoms need 'bond_per_atom' > 0");

  bond_type = memory->grow(atom->bond_type,nmax,atom->bond_per_atom,"atom:bond_type");
  bond_atom = memory->grow(atom->bond_atom,nmax,atom->bond_per_atom,"atom:bond_atom");
fprintf(logfile, "AtomVecMCA::grow atom->bond_index= %d \n", atom->bond_index);  ///AS DEBUG
  bond_index = memory->grow(atom->bond_index,nmax,atom->bond_per_atom,"atom:bond_index");

  if(atom->n_bondhist < 0)
    error->all(FLERR,"atom->n_bondhist < 0 suggests that 'bond_style mca' has not been called before 'read_restart' command! Please check that.");

fprintf(logfile, "AtomVecMCA::grow atom->n_bondhist= %d \n", atom->n_bondhist);  ///AS DEBUG
  if(atom->n_bondhist)
  {
     bond_hist = atom->bond_hist =
        memory->grow(atom->bond_hist,nmax,atom->bond_per_atom,atom->n_bondhist,"atom:bond_hist");
  }

fprintf(logfile, "AtomVecMCA::grow atom->nextra_grow= %d \n", atom->nextra_grow);  ///AS DEBUG
  if (atom->nextra_grow)
    for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
      modify->fix[atom->extra_grow[iextra]]->grow_arrays(nmax);
}

/* ----------------------------------------------------------------------
   reset local array ptrs
------------------------------------------------------------------------- */

void AtomVecMCA::grow_reset()
{
  tag = atom->tag; type = atom->type;
  mask = atom->mask; image = atom->image;
  x = atom->x; v = atom->v; f = atom->f;
  density = atom->density; rmass = atom->rmass;
  omega = atom->omega; torque = atom->torque;

  mca_inertia = atom->mca_inertia;
  theta = atom->theta;
  theta_prev = atom->theta_prev;
  mean_stress = atom->mean_stress;
  mean_stress_prev = atom->mean_stress_prev;
  equiv_stress = atom->equiv_stress;
  equiv_stress_prev = atom->equiv_stress_prev;
  equiv_strain = atom->equiv_strain;
  cont_distance = atom->cont_distance;

  molecule = atom->molecule;
  nspecial = atom->nspecial; special = atom->special;
  num_bond = atom->num_bond; bond_type = atom->bond_type;
  bond_atom = atom->bond_atom;
  bond_index = atom->bond_index;
  n_bondhist = atom->n_bondhist; bond_hist = atom->bond_hist;
}

/* ----------------------------------------------------------------------
   copy atom I info to atom J
------------------------------------------------------------------------- */

void AtomVecMCA::copy(int i, int j, int delflag)
{
  int k,l;

  tag[j] = tag[i];
  type[j] = type[i];
  mask[j] = mask[i];
  image[j] = image[i];
  x[j][0] = x[i][0];
  x[j][1] = x[i][1];
  x[j][2] = x[i][2];
  v[j][0] = v[i][0];
  v[j][1] = v[i][1];
  v[j][2] = v[i][2];

  rmass[j] = rmass[i];
  density[j] = density[i];
  omega[j][0] = omega[i][0];
  omega[j][1] = omega[i][1];
  omega[j][2] = omega[i][2];

  mca_inertia[j] = mca_inertia[i];
  theta[j][0] = theta[i][0];
  theta[j][1] = theta[i][1];
  theta[j][2] = theta[i][2];
  theta_prev[j][0] = theta_prev[i][0];
  theta_prev[j][1] = theta_prev[i][1];
  theta_prev[j][2] = theta_prev[i][2];
  mean_stress[j] = mean_stress[i];
  mean_stress_prev[j] = mean_stress_prev[i];
  equiv_stress[j] = equiv_stress[i];
  equiv_stress_prev[j] = equiv_stress_prev[i];
  equiv_strain[j] = equiv_strain[i];
  cont_distance[j] = cont_distance[i];

  molecule[j] = molecule[i];

  num_bond[j] = num_bond[i];
  for (k = 0; k < num_bond[j]; k++) {
    bond_type[j][k] = bond_type[i][k];
    bond_atom[j][k] = bond_atom[i][k];
    bond_index[j][k] = bond_index[i][k];
  }

  if(atom->n_bondhist)
  {
      for (k = 0; k < num_bond[j]; k++)
         for (l =0; l < atom->n_bondhist; l++)
            bond_hist[j][k][l] = bond_hist[i][k][l];
  }

  nspecial[j][0] = nspecial[i][0];
  nspecial[j][1] = nspecial[i][1];
  nspecial[j][2] = nspecial[i][2];
  for (k = 0; k < nspecial[j][2]; k++) special[j][k] = special[i][k];

  if (atom->nextra_grow)
    for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
      modify->fix[atom->extra_grow[iextra]]->copy_arrays(i,j,delflag);
}

/* ---------------------------------------------------------------------- */

int AtomVecMCA::pack_comm(int n, int *list, double *buf,
			   int pbc_flag, int *pbc)
{
  int i,j,m;
  int k,l;
  double dx,dy,dz;

///AS  if (radvary == 0) {
  m = 0;
  if (pbc_flag == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j][0];
      buf[m++] = x[j][1];
      buf[m++] = x[j][2];
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
      buf[m++] = cont_distance[j];
      buf[m++] = ubuf(num_bond[j]).d;
      for (k = 0; k < num_bond[j]; k++) {
        buf[m++] = ubuf(bond_type[j][k]).d;
        buf[m++] = ubuf(bond_atom[j][k]).d;
        buf[m++] = ubuf(bond_index[j][k]).d;
      }
      if(atom->n_bondhist)
      {
          for (k = 0; k < num_bond[j]; k++)
             for (l = 0; l < atom->n_bondhist; l++)
                buf[m++] = bond_hist[j][k][l];
      }
    }
  } else {
    if (domain->triclinic == 0) {
      dx = pbc[0]*domain->xprd;
      dy = pbc[1]*domain->yprd;
      dz = pbc[2]*domain->zprd;
    } else {
      dx = pbc[0]*domain->xprd + pbc[5]*domain->xy + pbc[4]*domain->xz;
      dy = pbc[1]*domain->yprd + pbc[3]*domain->yz;
      dz = pbc[2]*domain->zprd;
    }
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j][0] + dx;
      buf[m++] = x[j][1] + dy;
      buf[m++] = x[j][2] + dz;
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
      buf[m++] = cont_distance[j];
      buf[m++] = ubuf(num_bond[j]).d;
      for (k = 0; k < num_bond[j]; k++) {
        buf[m++] = ubuf(bond_type[j][k]).d;
        buf[m++] = ubuf(bond_atom[j][k]).d;
        buf[m++] = ubuf(bond_index[j][k]).d;
      }
      if(atom->n_bondhist)
      {
          for (k = 0; k < num_bond[j]; k++)
             for (l = 0; l < atom->n_bondhist; l++)
                buf[m++] = bond_hist[j][k][l];
      }
    }
  }
/*
  } else {
    m = 0;
    if (pbc_flag == 0) {
      for (i = 0; i < n; i++) {
        j = list[i];
        buf[m++] = x[j][0];
        buf[m++] = x[j][1];
        buf[m++] = x[j][2];
        buf[m++] = ubuf(type[j]).d;
        buf[m++] = rmass[j];
        buf[m++] = density[j];
        buf[m++] = mca_inertia[j];
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
        buf[m++] = cont_distance[j]
      }
    } else {
      if (domain->triclinic == 0) {
        dx = pbc[0]*domain->xprd;
        dy = pbc[1]*domain->yprd;
        dz = pbc[2]*domain->zprd;
      } else {
        dx = pbc[0]*domain->xprd + pbc[5]*domain->xy + pbc[4]*domain->xz;
        dy = pbc[1]*domain->yprd + pbc[3]*domain->yz;
        dz = pbc[2]*domain->zprd;
      }
      for (i = 0; i < n; i++) {
        j = list[i];
        buf[m++] = x[j][0] + dx;
        buf[m++] = x[j][1] + dy;
        buf[m++] = x[j][2] + dz;
        buf[m++] = ubuf(type[j]).d;
        buf[m++] = rmass[j];
        buf[m++] = density[j];
        buf[m++] = mca_inertia[j];
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
        buf[m++] = cont_distance[j]
      }
    }
  } */
fprintf(logfile,"AtomVecMCA::pack_comm m=%d n=%d \n",m,n);

  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecMCA::pack_comm_vel(int n, int *list, double *buf,
			       int pbc_flag, int *pbc)
{
  if(dynamic_cast<DomainWedge*>(domain))
    return pack_comm_vel_wedge(n,list,buf,pbc_flag,pbc);

  int i,j,m;
  int k,l;
  double dx,dy,dz,dvx,dvy,dvz;

///AS if (radvary == 0) {
  m = 0;
  if (pbc_flag == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j][0];
      buf[m++] = x[j][1];
      buf[m++] = x[j][2];
      buf[m++] = v[j][0];
      buf[m++] = v[j][1];
      buf[m++] = v[j][2];
/*
      buf[m++] = omega[j][0];
      buf[m++] = omega[j][1];
      buf[m++] = omega[j][2];
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
      buf[m++] = cont_distance[j];
      buf[m++] = ubuf(num_bond[j]).d;
      for (k = 0; k < num_bond[j]; k++) {
        buf[m++] = ubuf(bond_type[j][k]).d;
        buf[m++] = ubuf(bond_atom[j][k]).d;
        buf[m++] = ubuf(bond_index[j][k]).d;
      }
      if(atom->n_bondhist)
      {
          for (k = 0; k < num_bond[j]; k++)
             for (l = 0; l < atom->n_bondhist; l++)
                buf[m++] = bond_hist[j][k][l];
      }*/
    }
  } else {
    if (domain->triclinic == 0) {
      dx = pbc[0]*domain->xprd;
      dy = pbc[1]*domain->yprd;
      dz = pbc[2]*domain->zprd;
    } else {
      dx = pbc[0]*domain->xprd + pbc[5]*domain->xy + pbc[4]*domain->xz;
      dy = pbc[1]*domain->yprd + pbc[3]*domain->yz;
      dz = pbc[2]*domain->zprd;
    }
    if (!deform_vremap) {
      for (i = 0; i < n; i++) {
        j = list[i];
        buf[m++] = x[j][0] + dx;
        buf[m++] = x[j][1] + dy;
        buf[m++] = x[j][2] + dz;
        buf[m++] = v[j][0];
        buf[m++] = v[j][1];
        buf[m++] = v[j][2];
/*
        buf[m++] = omega[j][0];
        buf[m++] = omega[j][1];
        buf[m++] = omega[j][2];
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
        buf[m++] = cont_distance[j];
        buf[m++] = ubuf(num_bond[j]).d;
//if(tag[j]==10) fprintf(logfile,"pack_comm_vel num_bond[%d(tag=%d)]=%d\n",j,tag[j],num_bond[j]);
        for (k = 0; k < num_bond[j]; k++) {
          buf[m++] = ubuf(bond_type[j][k]).d;
          buf[m++] = ubuf(bond_atom[j][k]).d;
          buf[m++] = ubuf(bond_index[j][k]).d;
//if(tag[j]==10) fprintf(logfile,"\tbond_atom[j][%d]=%d m=%d buf[m]=%+20.14e\n",k,bond_atom[j][k],m-1,buf[m-1]);
        }
        if(atom->n_bondhist) {
          for (k = 0; k < num_bond[j]; k++)
            for (l = 0; l < atom->n_bondhist; l++)
              buf[m++] = bond_hist[j][k][l];
        }*/
      }
    } else {
      dvx = pbc[0]*h_rate[0] + pbc[5]*h_rate[5] + pbc[4]*h_rate[4];
      dvy = pbc[1]*h_rate[1] + pbc[3]*h_rate[3];
      dvz = pbc[2]*h_rate[2];
      for (i = 0; i < n; i++) {
        j = list[i];
        buf[m++] = x[j][0] + dx;
        buf[m++] = x[j][1] + dy;
        buf[m++] = x[j][2] + dz;
        if (mask[i] & deform_groupbit) {
          buf[m++] = v[j][0] + dvx;
          buf[m++] = v[j][1] + dvy;
          buf[m++] = v[j][2] + dvz;
        } else {
          buf[m++] = v[j][0];
          buf[m++] = v[j][1];
          buf[m++] = v[j][2];
        }
/*
        buf[m++] = omega[j][0];
        buf[m++] = omega[j][1];
        buf[m++] = omega[j][2];
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
        buf[m++] = cont_distance[j];
        buf[m++] = ubuf(num_bond[j]).d;
        for (k = 0; k < num_bond[j]; k++) {
          buf[m++] = ubuf(bond_type[j][k]).d;
          buf[m++] = ubuf(bond_atom[j][k]).d;
          buf[m++] = ubuf(bond_index[j][k]).d;
        }
        if(atom->n_bondhist) {
          for (k = 0; k < num_bond[j]; k++)
            for (l = 0; l < atom->n_bondhist; l++)
              buf[m++] = bond_hist[j][k][l];
        }*/
      }
    }
  }
/* AS
  } else {
    m = 0;
    if (pbc_flag == 0) {
      for (i = 0; i < n; i++) {
        j = list[i];
        buf[m++] = x[j][0];
        buf[m++] = x[j][1];
        buf[m++] = x[j][2];
        buf[m++] = ubuf(type[j]).d;
        buf[m++] = rmass[j];
        buf[m++] = density[j];
        buf[m++] = v[j][0];
        buf[m++] = v[j][1];
        buf[m++] = v[j][2];
        buf[m++] = omega[j][0];
        buf[m++] = omega[j][1];
        buf[m++] = omega[j][2];

        buf[m++] = mca_inertia[j];
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
        buf[m++] = cont_distance[j]
      }
    } else {
      if (domain->triclinic == 0) {
        dx = pbc[0]*domain->xprd;
        dy = pbc[1]*domain->yprd;
        dz = pbc[2]*domain->zprd;
      } else {
        dx = pbc[0]*domain->xprd + pbc[5]*domain->xy + pbc[4]*domain->xz;
        dy = pbc[1]*domain->yprd + pbc[3]*domain->yz;
        dz = pbc[2]*domain->zprd;
      }
      if (!deform_vremap) {
        for (i = 0; i < n; i++) {
          j = list[i];
          buf[m++] = x[j][0] + dx;
          buf[m++] = x[j][1] + dy;
          buf[m++] = x[j][2] + dz;
          buf[m++] = ubuf(type[j]).d;
          buf[m++] = rmass[j];
          buf[m++] = density[j];
          buf[m++] = v[j][0];
          buf[m++] = v[j][1];
          buf[m++] = v[j][2];
          buf[m++] = omega[j][0];
          buf[m++] = omega[j][1];
          buf[m++] = omega[j][2];

          buf[m++] = mca_inertia[j];
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
          buf[m++] = cont_distance[j]
        }
      } else {
        dvx = pbc[0]*h_rate[0] + pbc[5]*h_rate[5] + pbc[4]*h_rate[4];
        dvy = pbc[1]*h_rate[1] + pbc[3]*h_rate[3];
        dvz = pbc[2]*h_rate[2];
        for (i = 0; i < n; i++) {
          j = list[i];
          buf[m++] = x[j][0] + dx;
          buf[m++] = x[j][1] + dy;
          buf[m++] = x[j][2] + dz;
          buf[m++] = ubuf(type[j]).d;
          buf[m++] = rmass[j];
          buf[m++] = density[j];
          if (mask[i] & deform_groupbit) {
            buf[m++] = v[j][0] + dvx;
            buf[m++] = v[j][1] + dvy;
            buf[m++] = v[j][2] + dvz;
          } else {
            buf[m++] = v[j][0];
            buf[m++] = v[j][1];
            buf[m++] = v[j][2];
          }
          buf[m++] = omega[j][0];
          buf[m++] = omega[j][1];
          buf[m++] = omega[j][2];

          buf[m++] = mca_inertia[j];
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
          buf[m++] = cont_distance[j]
        }
      }
    }
  } */

//fprintf(logfile,"AtomVecMCA::pack_comm_vel m=%d n=%d [%d - %d]\n",m,n,list[0],list[n-1]);
  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecMCA::pack_comm_hybrid(int n, int *list, double *buf)
{
  int i,j,m;
  int k,l;

  m = 0;
///AS  if (radvary == 0) {
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
      buf[m++] = cont_distance[j];
      buf[m++] = ubuf(num_bond[j]).d;
      for (k = 0; k < num_bond[j]; k++) {
        buf[m++] = ubuf(bond_type[j][k]).d;
        buf[m++] = ubuf(bond_atom[j][k]).d;
        buf[m++] = ubuf(bond_index[j][k]).d;
      }
      if(atom->n_bondhist)
      {
          for (k = 0; k < num_bond[j]; k++)
             for (l = 0; l < atom->n_bondhist; l++)
                buf[m++] = bond_hist[j][k][l];
      }
    }
/* AS
  } else {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = ubuf(type[j]).d;
      buf[m++] = rmass[j];
      buf[m++] = density[j];
      buf[m++] = mca_inertia[j];
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
      buf[m++] = cont_distance[j]
    }
 } */
fprintf(logfile,"AtomVecMCA::pack_comm_hybrid m=%d n=%d \n",m,n);
  return m;
}

/* ---------------------------------------------------------------------- */

void AtomVecMCA::unpack_comm(int n, int first, double *buf)
{
  int i,m,last;
  int k,l;

  m = 0;
///AS if (radvary == 0) {
  last = first + n;
  for (i = first; i < last; i++) {
    x[i][0] = buf[m++];
    x[i][1] = buf[m++];
    x[i][2] = buf[m++];
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
    cont_distance[i] = buf[m++];
    num_bond[i] = (int) ubuf(buf[m++]).i;
    for (k = 0; k < num_bond[i]; k++) {
      bond_type[i][k] = (int) ubuf(buf[m++]).i;
      bond_atom[i][k] = (int) ubuf(buf[m++]).i;
      bond_index[i][k] = (int) ubuf(buf[m++]).i;
    }
    if(atom->n_bondhist)
    {
        for (k = 0; k < num_bond[i]; k++)
          for (l = 0; l < atom->n_bondhist; l++)
            bond_hist[i][k][l] = buf[m++];
    }
   }
/*AS  } else {
    m = 0;
    last = first + n;
    for (i = first; i < last; i++) {
      x[i][0] = buf[m++];
      x[i][1] = buf[m++];
      x[i][2] = buf[m++];
      type[i] = (int) ubuf(buf[m++]).i;
      rmass[i] = buf[m++];
      density[i] = buf[m++];
      mca_inertia[i] = buf[m++];
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
      cont_distance[i] = buf[m++];
    }
  } */
fprintf(logfile,"AtomVecMCA::unpack_comm m=%d n=%d \n",m,n);
}

/* ---------------------------------------------------------------------- */

void AtomVecMCA::unpack_comm_vel(int n, int first, double *buf)
{
  int i,m,last;
  int k,l;

///AS if (radvary == 0) {
  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    x[i][0] = buf[m++];
    x[i][1] = buf[m++];
    x[i][2] = buf[m++];
    v[i][0] = buf[m++];
    v[i][1] = buf[m++];
    v[i][2] = buf[m++];
/*
    omega[i][0] = buf[m++];
    omega[i][1] = buf[m++];
    omega[i][2] = buf[m++];
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
    cont_distance[i] = buf[m++];
    num_bond[i] = (int) ubuf(buf[m++]).i;
//if(tag[i]==10) fprintf(logfile,"unpack_comm_vel num_bond[%d(tag=%d)]=%d\n",i,tag[i],num_bond[i]);
    for (k = 0; k < num_bond[i]; k++) {
      bond_type[i][k] = (int) ubuf(buf[m++]).i;
      bond_atom[i][k] = (int) ubuf(buf[m++]).i;
      bond_index[i][k] = (int) ubuf(buf[m++]).i;
//if(tag[i]==10) fprintf(logfile,"\tbond_atom[i][%d]=%d m=%d buf[m]=%+20.14e\n",k,bond_atom[i][k],m-1,buf[m-1]);
    }
    if(atom->n_bondhist)
    {
        for (k = 0; k < num_bond[i]; k++)
          for (l = 0; l < atom->n_bondhist; l++)
            bond_hist[i][k][l] = buf[m++];
    }*/
  }
/*AS
  } else {
    m = 0;
    last = first + n;
    for (i = first; i < last; i++) {
      x[i][0] = buf[m++];
      x[i][1] = buf[m++];
      x[i][2] = buf[m++];
      type[i] = (int) ubuf(buf[m++]).i;
      rmass[i] = buf[m++];
      density[i] = buf[m++];
      v[i][0] = buf[m++];
      v[i][1] = buf[m++];
      v[i][2] = buf[m++];
      omega[i][0] = buf[m++];
      omega[i][1] = buf[m++];
      omega[i][2] = buf[m++];

      mca_inertia[i] = buf[m++];
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
      cont_distance[i] = buf[m++];
    }
  } */
//fprintf(logfile,"AtomVecMCA::unpack_comm_vel m=%d n=%d [%d - %d]\n",m,n,first,last);
}

/* ---------------------------------------------------------------------- */

int AtomVecMCA::unpack_comm_hybrid(int n, int first, double *buf)
{
  int i,m,last;
  int k,l;

  m = 0;
  last = first + n;
///AS if (radvary == 0) {
  for (i = first; i < last; i++) {
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
    cont_distance[i] = buf[m++];
    num_bond[i] = (int) ubuf(buf[m++]).i;
    for (k = 0; k < num_bond[i]; k++) {
      bond_type[i][k] = (int) ubuf(buf[m++]).i;
      bond_atom[i][k] = (int) ubuf(buf[m++]).i;
      bond_index[i][k] = (int) ubuf(buf[m++]).i;
    }
    if(atom->n_bondhist)
    {
        for (k = 0; k < num_bond[i]; k++)
          for (l = 0; l < atom->n_bondhist; l++)
            bond_hist[i][k][l] = buf[m++];
    }
   }
/*  } else {
  for (i = first; i < last; i++) {
    type[i] = (int) ubuf(buf[m++]).i;
    rmass[i] = buf[m++];
    density[i] = buf[m++];
    mca_inertia[i] = buf[m++];
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
    cont_distance[i] = buf[m++];
  }
  }*/
fprintf(logfile,"AtomVecMCA::unpack_comm_hybrid m=%d n=%d \n",m,n);
  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecMCA::pack_reverse(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
/*
    buf[m++] = f[i][0];
    buf[m++] = f[i][1];
    buf[m++] = f[i][2];
    buf[m++] = torque[i][0];
    buf[m++] = torque[i][1];
    buf[m++] = torque[i][2];
*/
    buf[m++] = cont_distance[i];
  }
//fprintf(logfile,"AtomVecMCA::pack_reverse m=%d n=%d first=%d\n",m,n,first);
  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecMCA::pack_reverse_hybrid(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
/*    buf[m++] = torque[i][0];
    buf[m++] = torque[i][1];
    buf[m++] = torque[i][2];*/
    buf[m++] = cont_distance[i];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void AtomVecMCA::unpack_reverse(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
/*
    f[j][0] += buf[m++];/// += buf[m++];
    f[j][1] += buf[m++];/// += buf[m++];
    f[j][2] += buf[m++];/// += buf[m++];
    torque[j][0] += buf[m++];/// += buf[m++];
    torque[j][1] += buf[m++];/// += buf[m++];
    torque[j][2] += buf[m++];/// += buf[m++];
*/
    cont_distance[i] = buf[m++];
  }
//fprintf(logfile,"AtomVecMCA::unpack_reverse m=%d n=%d first=%d\n",m,n,list[0]);
}

/* ---------------------------------------------------------------------- */

int AtomVecMCA::unpack_reverse_hybrid(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
/*    torque[j][0] = buf[m++];/// += buf[m++];
    torque[j][1] = buf[m++];/// += buf[m++];
    torque[j][2] = buf[m++];/// += buf[m++];*/
    cont_distance[i] = buf[m++];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecMCA::pack_border(int n, int *list, double *buf,
			     int pbc_flag, int *pbc)
{
  int i,j,m;
  double dx,dy,dz;
  int k,l;

  m = 0;
  if (pbc_flag == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j][0];
      buf[m++] = x[j][1];
      buf[m++] = x[j][2];
      buf[m++] = ubuf(tag[j]).d;
      buf[m++] = ubuf(type[j]).d;
      buf[m++] = ubuf(mask[j]).d;
      buf[m++] = rmass[j];
      buf[m++] = density[j];

      buf[m++] = mca_inertia[j];
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
      buf[m++] = cont_distance[j];

      buf[m++] = ubuf(molecule[j]).d;
      buf[m++] = ubuf(num_bond[j]).d;
      for (k = 0; k < num_bond[j]; k++) {
        buf[m++] = ubuf(bond_type[j][k]).d;
        buf[m++] = ubuf(bond_atom[j][k]).d;
        buf[m++] = ubuf(bond_index[j][k]).d;
      }
      if(atom->n_bondhist) {
        for (k = 0; k < num_bond[j]; k++)
          for (l = 0; l < atom->n_bondhist; l++)
            buf[m++] = bond_hist[j][k][l];
      }
    }
  } else {
    if (domain->triclinic == 0) {
      dx = pbc[0]*domain->xprd;
      dy = pbc[1]*domain->yprd;
      dz = pbc[2]*domain->zprd;
    } else {
      dx = pbc[0];
      dy = pbc[1];
      dz = pbc[2];
    }
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j][0] + dx;
      buf[m++] = x[j][1] + dy;
      buf[m++] = x[j][2] + dz;
      buf[m++] = ubuf(tag[j]).d;
      buf[m++] = ubuf(type[j]).d;
      buf[m++] = ubuf(mask[j]).d;
      buf[m++] = rmass[j];
      buf[m++] = density[j];

      buf[m++] = mca_inertia[j];
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
      buf[m++] = cont_distance[j];

      buf[m++] = ubuf(molecule[j]).d;
      buf[m++] = ubuf(num_bond[j]).d;
      for (k = 0; k < num_bond[j]; k++) {
        buf[m++] = ubuf(bond_type[j][k]).d;
        buf[m++] = ubuf(bond_atom[j][k]).d;
        buf[m++] = ubuf(bond_index[j][k]).d;
      }
      if(atom->n_bondhist) {
        for (k = 0; k < num_bond[j]; k++)
          for (l = 0; l < atom->n_bondhist; l++)
            buf[m++] = bond_hist[j][k][l];
      }
    }
  }

 if (atom->nextra_border)
    for (int iextra = 0; iextra < atom->nextra_border; iextra++)
      m += modify->fix[atom->extra_border[iextra]]->pack_border(n,list,&buf[m]);

fprintf(logfile,"AtomVecMCA::pack_border m=%d\n",m);
  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecMCA::pack_border_vel(int n, int *list, double *buf,
				 int pbc_flag, int *pbc)
{
  if(dynamic_cast<DomainWedge*>(domain))
    return pack_border_vel_wedge(n,list,buf,pbc_flag,pbc);

  int i,j,m;
  double dx,dy,dz,dvx,dvy,dvz;
  int k,l;

  m = 0;
  if (pbc_flag == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j][0];
      buf[m++] = x[j][1];
      buf[m++] = x[j][2];

      buf[m++] = ubuf(tag[j]).d;
      buf[m++] = ubuf(type[j]).d;
      buf[m++] = ubuf(mask[j]).d;
      buf[m++] = rmass[j];
      buf[m++] = density[j];

      buf[m++] = v[j][0];
      buf[m++] = v[j][1];
      buf[m++] = v[j][2];

      buf[m++] = omega[j][0];
      buf[m++] = omega[j][1];
      buf[m++] = omega[j][2];

      buf[m++] = mca_inertia[j];
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
      buf[m++] = cont_distance[j];

      buf[m++] = ubuf(molecule[j]).d;
      buf[m++] = ubuf(num_bond[j]).d;
      for (k = 0; k < num_bond[j]; k++) {
        buf[m++] = ubuf(bond_type[j][k]).d;
        buf[m++] = ubuf(bond_atom[j][k]).d;
        buf[m++] = ubuf(bond_index[j][k]).d;
      }
      if(atom->n_bondhist) {
        for (k = 0; k < num_bond[j]; k++)
          for (l = 0; l < atom->n_bondhist; l++)
            buf[m++] = bond_hist[j][k][l];
      }
    }
  } else {
    if (domain->triclinic == 0) {
      dx = pbc[0]*domain->xprd;
      dy = pbc[1]*domain->yprd;
      dz = pbc[2]*domain->zprd;
    } else {
      dx = pbc[0];
      dy = pbc[1];
      dz = pbc[2];
    }
    if (!deform_vremap) {
      for (i = 0; i < n; i++) {
        j = list[i];
        buf[m++] = x[j][0] + dx;
        buf[m++] = x[j][1] + dy;
        buf[m++] = x[j][2] + dz;
//if(tag[j]==10) fprintf(logfile,"pack_border_vel %d(tag=%d) X= %g %g %g\n",j,tag[j],x[j][0],x[j][1],x[j][2]);

        buf[m++] = ubuf(tag[j]).d;
        buf[m++] = ubuf(type[j]).d;
        buf[m++] = ubuf(mask[j]).d;
        buf[m++] = rmass[j];
        buf[m++] = density[j];

        buf[m++] = v[j][0];
        buf[m++] = v[j][1];
        buf[m++] = v[j][2];

        buf[m++] = omega[j][0];
        buf[m++] = omega[j][1];
        buf[m++] = omega[j][2];

        buf[m++] = mca_inertia[j];
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
        buf[m++] = cont_distance[j];

        buf[m++] = ubuf(molecule[j]).d;
        buf[m++] = ubuf(num_bond[j]).d;
//if(tag[j]==10) fprintf(logfile,"pack_border_vel num_bond[%d(tag=%d)]=%d\n",j,tag[j],num_bond[j]);
        for (k = 0; k < num_bond[j]; k++) {
          buf[m++] = ubuf(bond_type[j][k]).d;
          buf[m++] = ubuf(bond_atom[j][k]).d;
          buf[m++] = ubuf(bond_index[j][k]).d;
//if(tag[j]==10) fprintf(logfile,"\tbond_atom[j][%d]=%d m=%d buf[m]=%+20.14e\n",k,bond_atom[j][k],m-1,buf[m-1]);
        }
        if(atom->n_bondhist) {
          for (k = 0; k < num_bond[j]; k++)
            for (l = 0; l < atom->n_bondhist; l++)
              buf[m++] = bond_hist[j][k][l];
        }
      }
    } else {
      dvx = pbc[0]*h_rate[0] + pbc[5]*h_rate[5] + pbc[4]*h_rate[4];
      dvy = pbc[1]*h_rate[1] + pbc[3]*h_rate[3];
      dvz = pbc[2]*h_rate[2];
      for (i = 0; i < n; i++) {
        j = list[i];
        buf[m++] = x[j][0] + dx;
        buf[m++] = x[j][1] + dy;
        buf[m++] = x[j][2] + dz;

        buf[m++] = ubuf(tag[j]).d;
        buf[m++] = ubuf(type[j]).d;
        buf[m++] = ubuf(mask[j]).d;
        buf[m++] = rmass[j];
        buf[m++] = density[j];

        if (mask[i] & deform_groupbit) {
          buf[m++] = v[j][0] + dvx;
          buf[m++] = v[j][1] + dvy;
          buf[m++] = v[j][2] + dvz;
        } else {
          buf[m++] = v[j][0];
          buf[m++] = v[j][1];
          buf[m++] = v[j][2];
        }
        buf[m++] = omega[j][0];
        buf[m++] = omega[j][1];
        buf[m++] = omega[j][2];

        buf[m++] = mca_inertia[j];
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
        buf[m++] = cont_distance[j];

        buf[m++] = ubuf(molecule[j]).d;
        buf[m++] = ubuf(num_bond[j]).d;
        for (k = 0; k < num_bond[j]; k++) {
          buf[m++] = ubuf(bond_type[j][k]).d;
          buf[m++] = ubuf(bond_atom[j][k]).d;
          buf[m++] = ubuf(bond_index[j][k]).d;
        }
        if(atom->n_bondhist) {
          for (k = 0; k < num_bond[j]; k++)
            for (l = 0; l < atom->n_bondhist; l++)
              buf[m++] = bond_hist[j][k][l];
        }
      }
    }
  }

   if (atom->nextra_border)
    for (int iextra = 0; iextra < atom->nextra_border; iextra++)
      m += modify->fix[atom->extra_border[iextra]]->pack_border(n,list,&buf[m]);

//fprintf(logfile,"AtomVecMCA::pack_border_vel m=%d n=%d [%d - %d] deform_vremap=%d\n",m,n,list[0],list[n-1],deform_vremap);
  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecMCA::pack_border_hybrid(int n, int *list, double *buf)
{
  int i,j,m;
  int k,l;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = rmass[j];
    buf[m++] = density[j];

    buf[m++] = mca_inertia[j];
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
    buf[m++] = cont_distance[j];

    buf[m++] = ubuf(molecule[j]).d;
    buf[m++] = ubuf(num_bond[j]).d;
    for (k = 0; k < num_bond[j]; k++) {
      buf[m++] = ubuf(bond_type[j][k]).d;
      buf[m++] = ubuf(bond_atom[j][k]).d;
      buf[m++] = ubuf(bond_index[j][k]).d;
    }
    if(atom->n_bondhist) {
      for (k = 0; k < num_bond[j]; k++)
        for (l = 0; l < atom->n_bondhist; l++)
          buf[m++] = bond_hist[j][k][l];
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void AtomVecMCA::unpack_border(int n, int first, double *buf)
{
  int i,m,last;
  int k,l;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    if (i == nmax) grow(0);
    x[i][0] = buf[m++];
    x[i][1] = buf[m++];
    x[i][2] = buf[m++];
    tag[i] = (int) ubuf(buf[m++]).i;
    type[i] = (int) ubuf(buf[m++]).i;
    mask[i] = (int) ubuf(buf[m++]).i;
    rmass[i] = buf[m++];
    density[i] = buf[m++];

    mca_inertia[i] = buf[m++];
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
    cont_distance[i] = buf[m++];

    molecule[i] = (int) ubuf(buf[m++]).i; // remove?
    num_bond[i] = (int) ubuf(buf[m++]).i;
    for (k = 0; k < num_bond[i]; k++) {
      bond_type[i][k] = (int) ubuf(buf[m++]).i;
      bond_atom[i][k] = (int) ubuf(buf[m++]).i;
      bond_index[i][k] = (int) ubuf(buf[m++]).i;
    }
    if(atom->n_bondhist) {
      for (k = 0; k < num_bond[i]; k++)
        for (l = 0; l < atom->n_bondhist; l++)
          bond_hist[i][k][l] = buf[m++];
    }
  }

  if (atom->nextra_border)
    for (int iextra = 0; iextra < atom->nextra_border; iextra++)
      m += modify->fix[atom->extra_border[iextra]]->
        unpack_border(n,first,&buf[m]);
fprintf(logfile,"AtomVecMCA::unpack_border m=%d n=%d \n",m,n);
}

/* ---------------------------------------------------------------------- */

void AtomVecMCA::unpack_border_vel(int n, int first, double *buf)
{
  int i,m,last;
  int k,l;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    if (i == nmax) grow(0);
    x[i][0] = buf[m++];
    x[i][1] = buf[m++];
    x[i][2] = buf[m++];

    tag[i] = (int) ubuf(buf[m++]).i;
//if(tag[i]==10) fprintf(logfile,"unpack_border_vel %d(tag=%d) X= %g %g %g\n",i,tag[i],x[i][0],x[i][1],x[i][2]);
    type[i] = (int) ubuf(buf[m++]).i;
    mask[i] = (int) ubuf(buf[m++]).i;
    rmass[i] = buf[m++];
    density[i] = buf[m++];
    v[i][0] = buf[m++];
    v[i][1] = buf[m++];
    v[i][2] = buf[m++];
    omega[i][0] = buf[m++];
    omega[i][1] = buf[m++];
    omega[i][2] = buf[m++];

    mca_inertia[i] = buf[m++];
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
    cont_distance[i] = buf[m++];

    molecule[i] = (int) ubuf(buf[m++]).i;  // remove?
    num_bond[i] = (int) ubuf(buf[m++]).i;
//if(tag[i]==10) fprintf(logfile,"unpack_border_vel num_bond[%d(tag=%d)]=%d\n",i,tag[i],num_bond[i]);
    for (k = 0; k < num_bond[i]; k++) {
      bond_type[i][k] = (int) ubuf(buf[m++]).i;
      bond_atom[i][k] = (int) ubuf(buf[m++]).i;
      bond_index[i][k] = (int) ubuf(buf[m++]).i;
//if(tag[i]==10) fprintf(logfile,"\tbond_atom[i][%d]=%d m=%d buf[m]=%+20.14e\n",k,bond_atom[i][k],m-1,buf[m-1]);
    }
    if(atom->n_bondhist) {
      for (k = 0; k < num_bond[i]; k++)
        for (l = 0; l < atom->n_bondhist; l++)
          bond_hist[i][k][l] = buf[m++];
    }
  }

if (atom->nextra_border)
    for (int iextra = 0; iextra < atom->nextra_border; iextra++)
      m += modify->fix[atom->extra_border[iextra]]->
        unpack_border(n,first,&buf[m]);
//fprintf(logfile,"AtomVecMCA::unpack_border_vel m=%d n=%d [%d - %d]\n",m,n,first,last);
}

/* ---------------------------------------------------------------------- */

int AtomVecMCA::unpack_border_hybrid(int n, int first, double *buf)
{
  int i,m,last;
  int k,l;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    rmass[i] = buf[m++];
    density[i] = buf[m++];
    mca_inertia[i] = buf[m++];
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
    cont_distance[i] = buf[m++];
    molecule[i] = (int) ubuf(buf[m++]).i; //remove?
    num_bond[i] = (int) ubuf(buf[m++]).i;
    for (k = 0; k < num_bond[i]; k++) {
      bond_type[i][k] = (int) ubuf(buf[m++]).i;
      bond_atom[i][k] = (int) ubuf(buf[m++]).i;
      bond_index[i][k] = (int) ubuf(buf[m++]).i;
    }
    if(atom->n_bondhist) {
      for (k = 0; k < num_bond[i]; k++)
        for (l = 0; l < atom->n_bondhist; l++)
          bond_hist[i][k][l] = buf[m++];
    }
  }
  return m;
}

/* ----------------------------------------------------------------------
   pack data for atom I for sending to another proc
   xyz must be 1st 3 values, so comm::exchange() can test on them
------------------------------------------------------------------------- */

int AtomVecMCA::pack_exchange(int i, double *buf)
{
  int k,l;

  int m = 1;
  buf[m++] = x[i][0];
  buf[m++] = x[i][1];
  buf[m++] = x[i][2];
  buf[m++] = v[i][0];
  buf[m++] = v[i][1];
  buf[m++] = v[i][2];

  buf[m++] = ubuf(tag[i]).d;
  buf[m++] = ubuf(type[i]).d;
  buf[m++] = ubuf(mask[i]).d;
  buf[m++] = ubuf(image[i]).d;

  buf[m++] = rmass[i];
  buf[m++] = density[i];
  buf[m++] = omega[i][0];
  buf[m++] = omega[i][1];
  buf[m++] = omega[i][2];

  buf[m++] = mca_inertia[i];
  buf[m++] = theta[i][0];
  buf[m++] = theta[i][1];
  buf[m++] = theta[i][2];
  buf[m++] = theta_prev[i][0];
  buf[m++] = theta_prev[i][1];
  buf[m++] = theta_prev[i][2];
  buf[m++] = mean_stress[i];
  buf[m++] = mean_stress_prev[i];
  buf[m++] = equiv_stress[i];
  buf[m++] = equiv_stress_prev[i];
  buf[m++] = equiv_strain[i];
  buf[m++] = cont_distance[i];

  buf[m++] = ubuf(molecule[i]).d;
  buf[m++] = ubuf(num_bond[i]).d;
//if(tag[i]==10) fprintf(logfile,"pack_exchange num_bond[%d(tag=%d)]=%d\n",i,tag[i],num_bond[i]);
  for (k = 0; k < num_bond[i]; k++) {
    buf[m++] = ubuf(bond_type[i][k]).d;
    buf[m++] = ubuf(bond_atom[i][k]).d;
    buf[m++] = ubuf(bond_index[i][k]).d;
//if(tag[i]==10) fprintf(logfile,"\tbond_atom[i][%d]=%d m=%d buf[m]=%+20.14e\n",k,bond_atom[i][k],m-1,buf[m-1]);
  }
  if(atom->n_bondhist) {
    for (k = 0; k < num_bond[i]; k++)
      for (l = 0; l < atom->n_bondhist; l++)
        buf[m++] = bond_hist[i][k][l];
  }

  buf[m++] = ubuf(nspecial[i][0]).d;
  buf[m++] = ubuf(nspecial[i][1]).d;
  buf[m++] = ubuf(nspecial[i][2]).d;
  for (k = 0; k < nspecial[i][2]; k++) buf[m++] = ubuf(special[i][k]).d;

  if (atom->nextra_grow)
    for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
      m += modify->fix[atom->extra_grow[iextra]]->pack_exchange(i,&buf[m]);

  buf[0] = m;
fprintf(logfile,"AtomVecMCA::pack_exchange m=%d i=%d\n",m,i);
  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecMCA::unpack_exchange(double *buf)
{
  int k,l;

  int nlocal = atom->nlocal;
  if (nlocal == nmax) grow(0);

  int m = 1;
  x[nlocal][0] = buf[m++];
  x[nlocal][1] = buf[m++];
  x[nlocal][2] = buf[m++];
  v[nlocal][0] = buf[m++];
  v[nlocal][1] = buf[m++];
  v[nlocal][2] = buf[m++];

  tag[nlocal] = (int) ubuf(buf[m++]).i;
  type[nlocal] = (int) ubuf(buf[m++]).i;
  mask[nlocal] = (int) ubuf(buf[m++]).i;
  image[nlocal] = (tagint) ubuf(buf[m++]).i;

  rmass[nlocal] = buf[m++];
  density[nlocal] = buf[m++];
  omega[nlocal][0] = buf[m++];
  omega[nlocal][1] = buf[m++];
  omega[nlocal][2] = buf[m++];

  mca_inertia[nlocal] = buf[m++];
  theta[nlocal][0] = buf[m++];
  theta[nlocal][1] = buf[m++];
  theta[nlocal][2] = buf[m++];
  theta_prev[nlocal][0] = buf[m++];
  theta_prev[nlocal][1] = buf[m++];
  theta_prev[nlocal][2] = buf[m++];
  mean_stress[nlocal] = buf[m++];
  mean_stress_prev[nlocal] = buf[m++];
  equiv_stress[nlocal] = buf[m++];
  equiv_stress_prev[nlocal] = buf[m++];
  equiv_strain[nlocal] = buf[m++];
  cont_distance[nlocal] = buf[m++];

  molecule[nlocal] = (int) ubuf(buf[m++]).i;
  num_bond[nlocal] = (int) ubuf(buf[m++]).i;
//if(tag[nlocal]==10) fprintf(logfile,"unpack_exchange num_bond[%d(tag=%d)]=%d\n",nlocal,tag[nlocal],num_bond[nlocal]);
  for (k = 0; k < num_bond[nlocal]; k++) {
    bond_type[nlocal][k] = (int) ubuf(buf[m++]).i;
    bond_atom[nlocal][k] = (int) ubuf(buf[m++]).i;
    bond_index[nlocal][k] = (int) ubuf(buf[m++]).i;
//if(tag[nlocal]==10) fprintf(logfile,"\tbond_atom[nlocal][%d]=%d m=%d buf[m]=%+20.14e\n",k,bond_atom[nlocal][k],m-1,buf[m-1]);
  }
  if(atom->n_bondhist) {
    for (k = 0; k < num_bond[nlocal]; k++)
      for (l = 0; l < atom->n_bondhist; l++)
        bond_hist[nlocal][k][l] = buf[m++];
  }

  nspecial[nlocal][0] = (int) ubuf(buf[m++]).i;
  nspecial[nlocal][1] = (int) ubuf(buf[m++]).i;
  nspecial[nlocal][2] = (int) ubuf(buf[m++]).i;
  for (k = 0; k < nspecial[nlocal][2]; k++)
    special[nlocal][k] = (int) ubuf(buf[m++]).i;

  if (atom->nextra_grow)
    for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
      m += modify->fix[atom->extra_grow[iextra]]->
	unpack_exchange(nlocal,&buf[m]);

  atom->nlocal++;
fprintf(logfile,"AtomVecMCA::unpack_exchange m=%d nlocal=%d\n",m,nlocal);
  return m;
}

/* ----------------------------------------------------------------------
   size of restart data for all atoms owned by this proc
   include extra data stored by fixes
------------------------------------------------------------------------- */

int AtomVecMCA::size_restart()
{
  int i;

  int nlocal = atom->nlocal;
  int n = 0; ///AS We also have packing; coord_num; mca_radius; contact_area but...
             ///AS It was in AtomSphere  int n = 17 * nlocal;  ///////////////

  for (i = 0; i < nlocal; i++)
  {
    n += 31 + 3*num_bond[i];
///AS it was 13, we added 18 (rmass[i];density[i];omega[i][3];mca_inertia[i];theta[i][3];theta_prev[i][3];mean_stress[i];mean_stress_prev[i];equiv_stress[i];equiv_stress_prev[i];equiv_strain[i];cont_distance[i]) private variables, so total # is 30

    if(atom->n_bondhist) n += 1/*num_bondhist*/ + num_bond[i] * atom->n_bondhist/*bond_hist*/; //CR 26.01.2015
  }

  if (atom->nextra_restart)
    for (int iextra = 0; iextra < atom->nextra_restart; iextra++)
      for (i = 0; i < nlocal; i++)
	n += modify->fix[atom->extra_restart[iextra]]->size_restart(i);

  return n;
}

/* ----------------------------------------------------------------------
   pack atom I's data for restart file including extra quantities
   xyz must be 1st 3 values, so that read_restart can test on them
   molecular types may be negative, but write as positive
------------------------------------------------------------------------- */

int AtomVecMCA::pack_restart(int i, double *buf)
{
  int k,l;

  int m = 1;
  buf[m++] = x[i][0];
  buf[m++] = x[i][1];
  buf[m++] = x[i][2];
  buf[m++] = ubuf(tag[i]).d;
  buf[m++] = ubuf(type[i]).d;
  buf[m++] = ubuf(mask[i]).d;
  buf[m++] = ubuf(image[i]).d;

  buf[m++] = v[i][0];
  buf[m++] = v[i][1];
  buf[m++] = v[i][2];

  buf[m++] = rmass[i];
  buf[m++] = density[i];
  buf[m++] = omega[i][0];
  buf[m++] = omega[i][1];
  buf[m++] = omega[i][2];

  buf[m++] = mca_inertia[i];
  buf[m++] = theta[i][0];
  buf[m++] = theta[i][1];
  buf[m++] = theta[i][2]; 
  buf[m++] = theta_prev[i][0];
  buf[m++] = theta_prev[i][1];
  buf[m++] = theta_prev[i][2]; 
  buf[m++] = mean_stress[i];
  buf[m++] = mean_stress_prev[i];
  buf[m++] = equiv_stress[i];
  buf[m++] = equiv_stress_prev[i];
  buf[m++] = equiv_strain[i];
  buf[m++] = cont_distance[i];


  buf[m++] = ubuf(molecule[i]).d;
  buf[m++] = ubuf(num_bond[i]).d;
  for (k = 0; k < num_bond[i]; k++) {
    buf[m++] = bond_type[i][k];
    buf[m++] = bond_atom[i][k];
    buf[m++] = bond_index[i][k];
  }

  if(atom->n_bondhist) {
    buf[m++] = ubuf(atom->n_bondhist).d;
    for (k = 0; k < num_bond[i]; k++)
      for (l = 0; l < atom->n_bondhist; l++)
        buf[m++] = bond_hist[i][k][l];
  }

  if (atom->nextra_restart)
    for (int iextra = 0; iextra < atom->nextra_restart; iextra++)
      m += modify->fix[atom->extra_restart[iextra]]->pack_restart(i,&buf[m]);

  buf[0] = m;
  return m;
}

/* ----------------------------------------------------------------------
   unpack data for one atom from restart file including extra quantities
------------------------------------------------------------------------- */

int AtomVecMCA::unpack_restart(double *buf)
{
  int k,l;

  int nlocal = atom->nlocal;
  if (nlocal == nmax) {
    grow(0);
    if (atom->nextra_store)
      ///AS atom->extra = 
      memory->grow(atom->extra,nmax,atom->nextra_store,"atom:extra");
  }

  int m = 1;
  x[nlocal][0] = buf[m++];
  x[nlocal][1] = buf[m++];
  x[nlocal][2] = buf[m++];
  tag[nlocal] = (int) ubuf(buf[m++]).i;
  type[nlocal] = (int) ubuf(buf[m++]).i;
  mask[nlocal] = (int) ubuf(buf[m++]).i;
  image[nlocal] = (tagint) ubuf(buf[m++]).i;

  v[nlocal][0] = buf[m++];
  v[nlocal][1] = buf[m++];
  v[nlocal][2] = buf[m++];

  rmass[nlocal] = buf[m++];
  density[nlocal] = buf[m++];
  omega[nlocal][0] = buf[m++];
  omega[nlocal][1] = buf[m++];
  omega[nlocal][2] = buf[m++];

  mca_inertia[nlocal] = buf[m++];
  theta[nlocal][0] = buf[m++];
  theta[nlocal][1] = buf[m++];
  theta[nlocal][2] = buf[m++];
  theta_prev[nlocal][0] = buf[m++];
  theta_prev[nlocal][1] = buf[m++];
  theta_prev[nlocal][2] = buf[m++];
  mean_stress[nlocal] = buf[m++];
  mean_stress_prev[nlocal] = buf[m++];
  equiv_stress[nlocal] = buf[m++];
  equiv_stress_prev[nlocal] = buf[m++];
  equiv_strain[nlocal] = buf[m++];
  cont_distance[nlocal] = buf[m++];

  molecule[nlocal] = (int) ubuf(buf[m++]).i; //remove?
  num_bond[nlocal] = (int) ubuf(buf[m++]).i;
  for (k = 0; k < num_bond[nlocal]; k++) {
    bond_type[nlocal][k] = (int) ubuf(buf[m++]).i;
    bond_atom[nlocal][k] = (int) ubuf(buf[m++]).i;
    bond_index[nlocal][k] = (int) ubuf(buf[m++]).i;
  }

  if(atom->n_bondhist) {
    if(atom->n_bondhist != (int) ubuf(buf[m++]).i)
          error->all(FLERR,"Incompatible restart file: file was created using a bond model with a different number of history values");
    for (k = 0; k < num_bond[nlocal]; k++)
      for (l = 0; l < atom->n_bondhist; l++)
        atom->bond_hist[nlocal][k][l] = buf[m++];
  }

  double **extra = atom->extra;
  if (atom->nextra_store) {
    int size = (int) ubuf(buf[0]).i - m;

    for (int i = 0; i < size; i++) extra[nlocal][i] = buf[m++];
  }

  atom->nlocal++;
  return m;
}

/* ----------------------------------------------------------------------
   create one atom of itype at coord
   set other values to defaults
------------------------------------------------------------------------- */

void AtomVecMCA::create_atom(int itype, double *coord)
{
  int nlocal = atom->nlocal;
  if (nlocal == nmax) grow(0);

  tag[nlocal] = 0;
  type[nlocal] = itype;
  x[nlocal][0] = coord[0];
  x[nlocal][1] = coord[1];
  x[nlocal][2] = coord[2];
  mask[nlocal] = 1;
  image[nlocal] = ((tagint) IMGMAX << IMG2BITS) |
    ((tagint) IMGMAX << IMGBITS) | IMGMAX;
  v[nlocal][0] = 0.0;
  v[nlocal][1] = 0.0;
  v[nlocal][2] = 0.0;

  molecule[nlocal] = 0;
  num_bond[nlocal] = 0;
  nspecial[nlocal][0] = nspecial[nlocal][1] = nspecial[nlocal][2] = 0;

  density[nlocal] = 1.0;

  // To compute mass of mca particle we assume that it is a cube for
  // cubic packing and a rhombic dodecahedron for fcc or hcp packing.
  rmass[nlocal] = get_init_volume() * density[nlocal];
  omega[nlocal][0] = 0.0;
  omega[nlocal][1] = 0.0;
  omega[nlocal][2] = 0.0;

  // To simplify computation for rotation we assume that mca is a ball,
  // and its inertia can be described by one parameter (scalar).
  // The radius of this ball is not the same as mca_radius, but
  // is calculated from the initial volume of the particle.
  mca_inertia[nlocal] = 0.4*pow(3.0*get_init_volume()/(4.0*MY_PI), 2.0/3.0) * rmass[nlocal];
///fprintf(logfile, "AtomVecMCA::create_atom # %d at %20.12e %20.12e %20.12e rmass= %20.12e mca_radius= %g contact_area= %g mca_inertia=%g\n",nlocal,coord[0],coord[1],coord[2],rmass[nlocal],atom->mca_radius,atom->contact_area,mca_inertia[nlocal]); ///AS DEBUG

  theta[nlocal][0] = 0.0;
  theta[nlocal][1] = 0.0;
  theta[nlocal][2] = 0.0;
  theta_prev[nlocal][0] = 0.0;
  theta_prev[nlocal][1] = 0.0;
  theta_prev[nlocal][2] = 0.0;
  mean_stress[nlocal] = 0.0;
  mean_stress_prev[nlocal] = 0.0;
  equiv_stress[nlocal] = 0.0;
  equiv_stress_prev[nlocal] = 0.0;
  equiv_strain[nlocal] = 0.0;
  cont_distance[nlocal] = mca_radius;

  for(int k = 0; k < atom->bond_per_atom; k++) { ///num_bond[nlocal]; k++) {
      bond_index[nlocal][k] = -1;
      for (int l = 0; l < atom->n_bondhist; l++)
        atom->bond_hist[nlocal][k][l] = 0.0;
  }

  atom->nlocal++;
}

/* ----------------------------------------------------------------------
   unpack one line from Atoms section of data file
   initialize other atom quantities
------------------------------------------------------------------------- */

void AtomVecMCA::data_atom(double *coord, tagint imagetmp, char **values)
{
fprintf(logfile,"AtomVecMCA::data_atom\n"); ///AS DEBUG
  int nlocal = atom->nlocal;
  if (nlocal == nmax) grow(0);

  tag[nlocal] = atoi(values[0]);
  if (tag[nlocal] <= 0)
    error->one(FLERR,"Invalid atom ID in Atoms section of data file");

  type[nlocal] = atoi(values[1]);
  if (type[nlocal] <= 0 || type[nlocal] > atom->ntypes)
    error->one(FLERR,"Invalid atom type in Atoms section of data file");

  density[nlocal] = atof(values[2]);
  if (density[nlocal] <= 0.0)
    error->one(FLERR,"Invalid density in Atoms section of data file");

  rmass[nlocal] = get_init_volume() * density[nlocal];

  molecule[nlocal] = atoi(values[3]); ///AS We moved molecule parameter after all granular parameters

  x[nlocal][0] = coord[0];
  x[nlocal][1] = coord[1];
  x[nlocal][2] = coord[2];

  image[nlocal] = imagetmp;

  mask[nlocal] = 1;
  v[nlocal][0] = 0.0;
  v[nlocal][1] = 0.0;
  v[nlocal][2] = 0.0;
  omega[nlocal][0] = 0.0;
  omega[nlocal][1] = 0.0;
  omega[nlocal][2] = 0.0;

  mca_inertia[nlocal] = 0.4*pow(3.0*get_init_volume()/(4.0*MY_PI), 2.0/3.0) * rmass[nlocal];
  theta[nlocal][0] = 0.0;
  theta[nlocal][1] = 0.0;
  theta[nlocal][2] = 0.0;
  theta_prev[nlocal][0] = 0.0;
  theta_prev[nlocal][1] = 0.0;
  theta_prev[nlocal][2] = 0.0;
  mean_stress[nlocal] = 0.0;
  mean_stress_prev[nlocal] = 0.0;
  equiv_stress[nlocal] = 0.0;
  equiv_stress_prev[nlocal] = 0.0;
  equiv_strain[nlocal] = 0.0;
  cont_distance[nlocal] = mca_radius;

  num_bond[nlocal] = 0;

  atom->nlocal++;
}

/* ----------------------------------------------------------------------
   unpack hybrid quantities from one line in Atoms section of data file
   initialize other atom quantities for this sub-style
------------------------------------------------------------------------- */

int AtomVecMCA::data_atom_hybrid(int nlocal, char **values)
{
  molecule[nlocal] = atoi(values[0]);

  num_bond[nlocal] = 0;

  density[nlocal] = atof(values[1]);
  if (density[nlocal] <= 0.0)
    error->one(FLERR,"Invalid density in Atoms section of data file");

  rmass[nlocal] = get_init_volume() * density[nlocal];
  mca_inertia[nlocal] = 0.4*pow(3.0*get_init_volume()/(4.0*MY_PI), 2.0/3.0) * rmass[nlocal];

  return 2;
}


/* ----------------------------------------------------------------------
   unpack one line from Velocities section of data file
------------------------------------------------------------------------- */

void AtomVecMCA::data_vel(int m, char **values)
{
  v[m][0] = atof(values[0]);
  v[m][1] = atof(values[1]);
  v[m][2] = atof(values[2]);
  omega[m][0] = atof(values[3]);
  omega[m][1] = atof(values[4]);
  omega[m][2] = atof(values[5]);
}

/* ----------------------------------------------------------------------
   unpack hybrid quantities from one line in Velocities section of data file
------------------------------------------------------------------------- */

int AtomVecMCA::data_vel_hybrid(int m, char **values)
{
  omega[m][0] = atof(values[0]);
  omega[m][1] = atof(values[1]);
  omega[m][2] = atof(values[2]);
  return 3;
}


/* ----------------------------------------------------------------------
   pack atom info for data file including 3 image flags
------------------------------------------------------------------------- */

void AtomVecMCA::pack_data(double **buf)
{
  int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) {
    buf[i][0] = ubuf(tag[i]).d;
    buf[i][1] = ubuf(type[i]).d;
    buf[i][2] = rmass[i] / get_init_volume();

    buf[i][3] = x[i][0];
    buf[i][4] = x[i][1];
    buf[i][5] = x[i][2];
    buf[i][6] = ubuf((image[i] & IMGMASK) - IMGMAX).d;
    buf[i][7] = ubuf((image[i] >> IMGBITS & IMGMASK) - IMGMAX).d;
    buf[i][8] = ubuf((image[i] >> IMG2BITS) - IMGMAX).d;

///AS inertia can be restored using mca_radius and density    buf[i][9] = mca_inertia[i][0];
///AS ????????????
    buf[i][9] = theta[i][0]; // like coordinates
    buf[i][10] = theta[i][1];
    buf[i][11] = theta[i][2];
    buf[i][12] = theta_prev[i][0]; // like coordinates
    buf[i][13] = theta_prev[i][1];
    buf[i][14] = theta_prev[i][2];
    buf[i][15] = mean_stress[i];
    buf[i][16] = mean_stress_prev[i];
    buf[i][17] = equiv_stress[i];
    buf[i][18] = equiv_stress_prev[i];
    buf[i][19] = equiv_strain[i];
    buf[i][20] = cont_distance[i];
  }
}

/* ----------------------------------------------------------------------
   pack atom info for data file including 3 image flags
------------------------------------------------------------------------- */

void AtomVecMCA::pack_data(double **buf,int tag_offset)
{
  int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) {
    buf[i][0] = ubuf(tag[i]+tag_offset).d;
    buf[i][1] = ubuf(type[i]).d;
    buf[i][2] = rmass[i] / get_init_volume();

    buf[i][3] = x[i][0];
    buf[i][4] = x[i][1];
    buf[i][5] = x[i][2];
    buf[i][6] = ubuf((image[i] & IMGMASK) - IMGMAX).d;
    buf[i][7] = ubuf((image[i] >> IMGBITS & IMGMASK) - IMGMAX).d;
    buf[i][8] = ubuf((image[i] >> IMG2BITS) - IMGMAX).d;

///AS inertia can be restored using mca_radius and density        buf[i][9] = mca_inertia[i][0];
///AS ????????????
    buf[i][9] = theta[i][0]; // like coordinates
    buf[i][10] = theta[i][1];
    buf[i][11] = theta[i][2];
    buf[i][12] = theta_prev[i][0]; // like coordinates
    buf[i][13] = theta_prev[i][1];
    buf[i][14] = theta_prev[i][2];
    buf[i][15] = mean_stress[i];
    buf[i][16] = mean_stress_prev[i];
    buf[i][17] = equiv_stress[i];
    buf[i][18] = equiv_stress_prev[i];
    buf[i][19] = equiv_strain[i];
    buf[i][20] = cont_distance[i];
  }
}

/* ----------------------------------------------------------------------
   pack hybrid atom info for data file
------------------------------------------------------------------------- */

int AtomVecMCA::pack_data_hybrid(int i, double *buf)
{
  buf[0] = rmass[i] / get_init_volume();
///AS TODO ??? what we need to save?
  return 1;
}

/* ----------------------------------------------------------------------
   write atom info to data file including 3 image flags
------------------------------------------------------------------------- */

void AtomVecMCA::write_data(FILE *fp, int n, double **buf)
{
///AS TODO ??? what we need to save?
  for (int i = 0; i < n; i++)
    fprintf(fp,"%d %d %-1.16e %-1.16e %-1.16e %-1.16e %d %d %d %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e\n",
            (int) ubuf(buf[i][0]).i,(int) ubuf(buf[i][1]).i, // tag[i]+tag_offset  type[i]
            buf[i][2],				// rmass[i] / get_init_volume();
            buf[i][3],buf[i][4],buf[i][5],	// x[i][0] x[i][1] x[i][2]
            (int) ubuf(buf[i][6]).i,(int) ubuf(buf[i][7]).i, // ((image[i] & IMGMASK) - IMGMAX) ((image[i] >> IMGBITS & IMGMASK) - IMGMAX)
            (int) ubuf(buf[i][8]).i,		// ((image[i] >> IMG2BITS) - IMGMAX)
            buf[i][9],buf[i][10],buf[i][11],	// theta[i][1] theta[i][2] theta[i][2]
            buf[i][12],buf[i][13],buf[i][14],	// theta_prev[i][1] theta_prev[i][2] theta_prev[i][2]
            buf[i][15],buf[i][16],buf[i][17],buf[i][18],buf[i][19],buf[i][20]);	// mean_stress[i] mean_stress_prev[i] equiv_stress[i] equiv_stress_prev[i] equiv_strain[i] cont_distance[i]
}

/* ----------------------------------------------------------------------
   write hybrid atom info to data file
------------------------------------------------------------------------- */

int AtomVecMCA::write_data_hybrid(FILE *fp, double *buf)
{
///AS TODO ??? what we need to save?
  fprintf(fp," %-1.16e %-1.16e",buf[0],buf[1]);
  return 2;
}

/* ----------------------------------------------------------------------
   pack velocity info for data file
------------------------------------------------------------------------- */

void AtomVecMCA::pack_vel(double **buf)
{
  int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) {
    buf[i][0] = ubuf(tag[i]).d;
    buf[i][1] = v[i][0];
    buf[i][2] = v[i][1];
    buf[i][3] = v[i][2];
    buf[i][4] = omega[i][0];
    buf[i][5] = omega[i][1];
    buf[i][6] = omega[i][2];
  }
}

/* ----------------------------------------------------------------------
   pack velocity info for data file
------------------------------------------------------------------------- */

void AtomVecMCA::pack_vel(double **buf,int tag_offset) 
{
  int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) {
    buf[i][0] = ubuf(tag[i]+tag_offset).d;
    buf[i][1] = v[i][0];
    buf[i][2] = v[i][1];
    buf[i][3] = v[i][2];
    buf[i][4] = omega[i][0];
    buf[i][5] = omega[i][1];
    buf[i][6] = omega[i][2];
  }
}

/* ----------------------------------------------------------------------
   pack hybrid velocity info for data file
------------------------------------------------------------------------- */

int AtomVecMCA::pack_vel_hybrid(int i, double *buf)
{
  buf[0] = omega[i][0];
  buf[1] = omega[i][1];
  buf[2] = omega[i][2];
  return 3;
}

/* ----------------------------------------------------------------------
   write velocity info to data file
------------------------------------------------------------------------- */

void AtomVecMCA::write_vel(FILE *fp, int n, double **buf)
{
  for (int i = 0; i < n; i++)
    fprintf(fp,"%d %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e\n",
            (int) ubuf(buf[i][0]).i,buf[i][1],buf[i][2],buf[i][3],
            buf[i][4],buf[i][5],buf[i][6]);
}

/* ----------------------------------------------------------------------
   write hybrid velocity info to data file
------------------------------------------------------------------------- */

int AtomVecMCA::write_vel_hybrid(FILE *fp, double *buf)
{
  fprintf(fp," %-1.16e %-1.16e %-1.16e",buf[0],buf[1],buf[2]);
  return 3;
}

/* ----------------------------------------------------------------------
   return # of bytes of allocated memory
------------------------------------------------------------------------- */

bigint AtomVecMCA::memory_usage()
{
  bigint bytes = 0;

  if (atom->memcheck("tag")) bytes += memory->usage(tag,nmax);
  if (atom->memcheck("type")) bytes += memory->usage(type,nmax);
  if (atom->memcheck("mask")) bytes += memory->usage(mask,nmax);
  if (atom->memcheck("image")) bytes += memory->usage(image,nmax);
  if (atom->memcheck("x")) bytes += memory->usage(x,nmax,3);
  if (atom->memcheck("v")) bytes += memory->usage(v,nmax,3);
  if (atom->memcheck("f")) bytes += memory->usage(f,nmax*comm->nthreads,3);

  if (atom->memcheck("density")) bytes += memory->usage(density,nmax);
  if (atom->memcheck("rmass")) bytes += memory->usage(rmass,nmax);
  if (atom->memcheck("omega")) bytes += memory->usage(omega,nmax,3);
  if (atom->memcheck("torque")) bytes += memory->usage(torque,nmax*comm->nthreads,3);

  if (atom->memcheck("mca_inertia")) bytes += memory->usage(mca_inertia,nmax);
  if (atom->memcheck("theta")) bytes += memory->usage(theta,nmax,3);
  if (atom->memcheck("theta_prev")) bytes += memory->usage(theta_prev,nmax,3);
///AS TODO ??? comm->nthreads - do we need to pass them to other processors
  if (atom->memcheck("mean_stress")) bytes += memory->usage(mean_stress,nmax*comm->nthreads);
  if (atom->memcheck("mean_stress_prev")) bytes += memory->usage(mean_stress_prev,nmax*comm->nthreads);
  if (atom->memcheck("equiv_stress")) bytes += memory->usage(equiv_stress,nmax*comm->nthreads);
  if (atom->memcheck("equiv_stress_prev")) bytes += memory->usage(equiv_stress_prev,nmax*comm->nthreads);
  if (atom->memcheck("equiv_strain")) bytes += memory->usage(equiv_strain,nmax*comm->nthreads);
  if (atom->memcheck("cont_distance")) bytes += memory->usage(cont_distance,nmax*comm->nthreads);

  if (atom->memcheck("molecule")) bytes += memory->usage(molecule,nmax);
  if (atom->memcheck("nspecial")) bytes += memory->usage(nspecial,nmax,3);
  if (atom->memcheck("special"))  bytes += memory->usage(special,nmax,atom->maxspecial);
  if (atom->memcheck("num_bond")) bytes += memory->usage(num_bond,nmax);
  if (atom->memcheck("bond_type")) bytes += memory->usage(bond_type,nmax,atom->bond_per_atom);
  if (atom->memcheck("bond_atom")) bytes += memory->usage(bond_atom,nmax,atom->bond_per_atom);
  if (atom->memcheck("bond_index")) bytes += memory->usage(bond_index,nmax,atom->bond_per_atom);
  if (atom->n_bondhist) bytes += nmax*(atom->bond_per_atom)*BOND_HIST_LEN*sizeof(double);  //!! not sure about atom->n_bondhist

  return bytes;
}

/* ----------------------------------------------------------------------
  Compute initial volume of cellular automaton based on radius and packing.
  To compute initial volume of mca particle we assume that it is a cube for
  cubic packing and a rhombic dodecahedron for fcc or hcp packing.
------------------------------------------------------------------------- */

double AtomVecMCA::get_init_volume()
{
  if (domain->dimension == 2) {
    error->all(FLERR,"Illegal dimension in AtomVecMCA::get_init_volume()");
    return 1.0;
  }

  mca_radius = atom->mca_radius;
  if(atom->packing == SC) {
    return 8.0*mca_radius*mca_radius*mca_radius;
  } else if(atom->packing == FCC){
    return 4.0*sqrt(2.0)*mca_radius*mca_radius*mca_radius;
  } else if(atom->packing == HCP) {
    return 4.0*sqrt(2.0)*mca_radius*mca_radius*mca_radius;
  } else {
    error->all(FLERR,"Illegal packing in AtomVecMCA::get_init_volume()");
    return 1.0;
  }
}

/* ----------------------------------------------------------------------
  compute initial contact_area between cellular automata based on radius and packing
------------------------------------------------------------------------- */

double AtomVecMCA::get_contact_area()
{
  if (domain->dimension == 2) {
    error->all(FLERR,"Illegal dimension in AtomVecMCA::get_contact_area()");
    return 1.0;
  }

  mca_radius = atom->mca_radius;
  if(atom->packing == SC) {
    return 4.0*mca_radius*mca_radius;
  } else if(atom->packing == FCC){
    return sqrt(2.0)*mca_radius*mca_radius;
  } else if(atom->packing == HCP) {
    return sqrt(2.0)*mca_radius*mca_radius;
  } else {
    error->all(FLERR,"Illegal packing in AtomVecMCA::get_contact_area()");
    return 1.0;
  }
}
