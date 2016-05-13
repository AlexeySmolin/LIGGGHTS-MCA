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

enum{NONE,SC,BCC,FCC,HCP,DIAMOND,SQ,SQ2,HEX,CUSTOM};///AS taken from 'lattice.c'

/* ---------------------------------------------------------------------- */

AtomVecMCA::AtomVecMCA(LAMMPS *lmp) : AtomVec(lmp)
{
  molecular = 1;  //!! This allows storing 'bonds,angles,dihedrals,improper'. Do we need them all? in AtomVecEllipsoid==0  
  bonds_allow = 1;
  mass_type = 1; // per-type masses

  comm_x_only = 1;   // 1 if only exchange x in forward comm
  comm_f_only = 0;   ///AS TODO !! 1 if only exchange f in reverse comm - in Sphere = 0 ???????

  size_forward = 3;  ///AS TODO # of values per atom in comm !! Later choose what to pass via MPI
  size_reverse = 6;  ///AS TODO # in reverse comm !! Later choose what to pass via MPI
  size_border = 9;   ///AS TODO # in border comm
  size_velocity = 6; ///AS # of velocity based quantities
  size_data_atom = 7;///AS TODO number of values in Atom line
  size_data_vel = 7; ///AS TODO number of values in Velocity line
  xcol_data = 5;     ///AS TODO column (1-N) where x is in Atom line

  atom->molecule_flag = 1;

  atom->mca_flag = 1; //!! mca is not a sphere, similar but different
  atom->radius_flag = 0; ///AS ?????
  atom->rmass_flag = atom->omega_flag = atom->density_flag = atom->torque_flag = 1;

  fbe = NULL; //!! delete in destructor
}

AtomVecMCA::~AtomVecMCA()
{
  if(fbe != NULL) {
    delete fbe; //!! I do not see where it is created
  }
}

/* ---------------------------------------------------------------------- */

void AtomVecMCA::settings(int narg, char **arg)
{
// atom_style mca radius 0.0001 packing fcc n_bondtypes 1 bonds_per_atom 6  

  if (narg == 0) return;	//in case of restart no arguments are given, instead they are defined by read_restart_settings
  if (narg != 8) error->all(FLERR,"Invalid atom_style mca command, expecting exactly 8 arguments");

  if(strcmp(arg[0],"radius")) // 
    error->all(FLERR,"Illegal atom_style mca command, expecting 'radius'");

  mca_radius = atom->mca_radius = atof(arg[1]);
fprintf(stderr, "atom->mca_radius= %g  arg[1] '%s' \n", atom->mca_radius, arg[1]);  

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

  if(strcmp(arg[4],"n_bondtypes")) // The number of bond types (linked, unlinked)
    error->all(FLERR,"Illegal atom_style mca command, expecting 'n_bondtypes'");

  atom->nbondtypes = atoi(arg[5]); //!! Do we need types? It may be free,occupied,bonded,unbonded

  if(strcmp(arg[6],"bonds_per_atom")) // The maximum number of bonds that each atom can have (== packing: 6 - cubic, 12 - fcc)
    error->all(FLERR,"Illegal atom_style mca command, expecting 'bonds_per_atom'");

  atom->bond_per_atom = atoi(arg[7]);
//fprintf(stderr, "atom->bond_per_atom= %d < atom->coord_num %d atom->packing= %d arg[3] '%s' \n", atom->bond_per_atom, atom->coord_num, atom->packing, arg[3]);  
  if (atom->bond_per_atom < atom->coord_num) 
    error->all(FLERR,"Illegal atom_style mca command, 'bonds_per_atom' must be >= coordination number for packing");

}

void AtomVecMCA::write_restart_settings(FILE *fp)
{
  fwrite(&atom->mca_radius,sizeof(double),1,fp);//!?? In other types this function is used only for neighbours  
  fwrite(&atom->packing,sizeof(int),1,fp);//!??
  fwrite(&atom->coord_num,sizeof(int),1,fp);//!??
  fwrite(&atom->nbondtypes,sizeof(int),1,fp);
  fwrite(&atom->bond_per_atom,sizeof(int),1,fp);
}

void AtomVecMCA::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    fread(&atom->mca_radius,sizeof(double),1,fp);//!??
    fread(&atom->packing,sizeof(int),1,fp);//!??
    fread(&atom->coord_num,sizeof(int),1,fp);//!??
    fread(&atom->nbondtypes,sizeof(int),1,fp);
    fread(&atom->bond_per_atom,sizeof(int),1,fp);
  }
  MPI_Bcast(&atom->mca_radius,1,MPI_DOUBLE,0,world);//!??
  MPI_Bcast(&atom->packing,1,MPI_INT,0,world);//!??
  MPI_Bcast(&atom->coord_num,1,MPI_INT,0,world);//!??
  MPI_Bcast(&atom->nbondtypes,1,MPI_INT,0,world);
  MPI_Bcast(&atom->bond_per_atom,1,MPI_INT,0,world);
  mca_radius = atom->mca_radius;
  packing = atom->packing;
  coord_num = atom->coord_num;
  contact_area = atom->contact_area = get_contact_area();
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

  comm_x_only = 1; ///TODO !! See in constructor
  size_forward = 3; ///TODO !! See in constructor

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

 //!!AS - To check it in 'domain'!!!!  if (packing != domain->lattice->style)
 //!!AS    error->all(FLERR,"Packing in 'atom_style mca' must be the same as 'lattice style'");

  if(fbe == NULL)
  {
      char **fixarg = new char*[3];
      fixarg[0] = (char *) "MCA_BOND_EXCHANGE"; //!! It seems that it does not work.
      fixarg[1] = (char *) "all";
      fixarg[2] = (char *) "bond/exchange/mca";
      modify->add_fix(3,fixarg);
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

///AS  radius = memory->grow(atom->radius,nmax,"atom:radius");
  density = memory->grow(atom->density,nmax,"atom:density"); 
  rmass = memory->grow(atom->rmass,nmax,"atom:rmass");
  omega = memory->grow(atom->omega,nmax,3,"atom:omega");
  torque = memory->grow(atom->torque,nmax*comm->nthreads,3,"atom:torque");

  q = memory->grow(atom->q,nmax,"atom:q");
  mu = memory->grow(atom->mu,nmax,3,"atom:mu");

  p = memory->grow(atom->p,nmax*comm->nthreads,"atom:p");
  s0 = memory->grow(atom->s0,nmax*comm->nthreads,"atom:s0");
  e = memory->grow(atom->e,nmax*comm->nthreads,"atom:e");

  molecule = memory->grow(atom->molecule,nmax,"atom:molecule");
  nspecial = memory->grow(atom->nspecial,nmax,3,"atom:nspecial");
  special = memory->grow(atom->special,nmax,atom->maxspecial,"atom:special");
  num_bond = memory->grow(atom->num_bond,nmax,"atom:num_bond");
  bond_type = memory->grow(atom->bond_type,nmax,atom->bond_per_atom,"atom:bond_type");
  bond_atom = memory->grow(atom->bond_atom,nmax,atom->bond_per_atom,"atom:bond_atom");

  if(0 == atom->bond_per_atom)
    error->all(FLERR,"mca atoms need 'bond_per_atom' > 0");

  if(atom->n_bondhist < 0)
	  error->all(FLERR,"atom->n_bondhist < 0 suggests that 'bond_style mca' has not been called before 'read_restart' command! Please check that.");

  if(atom->n_bondhist)
  {
     bond_hist = atom->bond_hist =
        memory->grow(atom->bond_hist,nmax,atom->bond_per_atom,atom->n_bondhist,"atom:bond_hist");
  }

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
///AS  radius = atom->radius;
  density = atom->density; rmass = atom->rmass; 
  omega = atom->omega; torque = atom->torque;
//AS TODO
  q = atom->q;
  mu = atom->mu;
  p = atom->p;
  s0 = atom->s0;
  e = atom->e;


  molecule = atom->molecule;
  nspecial = atom->nspecial; special = atom->special;
  num_bond = atom->num_bond; bond_type = atom->bond_type;
  bond_atom = atom->bond_atom;
  bond_hist = atom->bond_hist;
}

/* ----------------------------------------------------------------------
   copy atom I info to atom J
------------------------------------------------------------------------- */

void AtomVecMCA::copy(int *i, int *j, int delflag)
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

///AS  radius[j] = radius[i];
  rmass[j] = rmass[i];
  density[j] = density[i]; 
  omega[j][0] = omega[i][0];
  omega[j][1] = omega[i][1];
  omega[j][2] = omega[i][2];

//AS TODO
  q[j] = q[i];

  mu[j][0] = mu[i][0];
  mu[j][1] = mu[i][1];
  mu[j][2] = mu[i][2];
  p[j][0] = p[i][0];
  p[j][1] = p[i][1];
  p[j][2] = p[i][2];
  s0[j][0] = s0[i][0];
  s0[j][1] = s0[i][1];
  s0[j][2] = s0[i][2];
  e[j][0] = e[i][0];
  e[j][1] = e[i][1];
  e[j][2] = e[i][2];
  
  
  molecule[j] = molecule[i];

  num_bond[j] = num_bond[i];
  for (k = 0; k < num_bond[j]; k++) {
    bond_type[j][k] = bond_type[i][k];
    bond_atom[j][k] = bond_atom[i][k];
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
  double dx,dy,dz;

///AS  if (radvary == 0) {
  m = 0;
  if (pbc_flag == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j][0];
      buf[m++] = x[j][1];
      buf[m++] = x[j][2];
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
///AS        buf[m++] = radius[j];
        buf[m++] = rmass[j];
        buf[m++] = density[j]; 
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
///AS        buf[m++] = radius[j];
        buf[m++] = rmass[j];
        buf[m++] = density[j]; 
      }
    }
  } */

  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecMCA::pack_comm_vel(int n, int *list, double *buf,
			       int pbc_flag, int *pbc)
{
  if(dynamic_cast<DomainWedge*>(domain))
    return pack_comm_vel_wedge(n,list,buf,pbc_flag,pbc);

  int i,j,m;
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
      buf[m++] = omega[j][0];
      buf[m++] = omega[j][1];
      buf[m++] = omega[j][2];

///AS TODO
      buf[m++] = q[j][0];
      buf[m++] = q[j][1];
      buf[m++] = q[j][2];
      buf[m++] = mu[j][0];
      buf[m++] = mu[j][1];
      buf[m++] = mu[j][2];
      buf[m++] = p[j][0];
      buf[m++] = p[j][1];
      buf[m++] = p[j][2];
      buf[m++] = s0[j][0];
      buf[m++] = s0[j][1];
      buf[m++] = s0[j][2];
      buf[m++] = e[j][0];
      buf[m++] = e[j][1];
      buf[m++] = e[j][2];
 

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
		  buf[m++] = omega[j][0];
		  buf[m++] = omega[j][1];
		  buf[m++] = omega[j][2];
//AS TODO 
		  buf[m++] = q[j][0];
		  buf[m++] = q[j][1];
		  buf[m++] = q[j][2];
		  buf[m++] = mu[j][0];
		  buf[m++] = mu[j][1];
		  buf[m++] = mu[j][2];
		  buf[m++] = p[j][0];
		  buf[m++] = p[j][1];
		  buf[m++] = p[j][2];
		  buf[m++] = s0[j][0];
		  buf[m++] = s0[j][1];
		  buf[m++] = s0[j][2];
		  buf[m++] = e[j][0];
		  buf[m++] = e[j][1];
		  buf[m++] = e[j][2];


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
		  buf[m++] = omega[j][0];
		  buf[m++] = omega[j][1];
		  buf[m++] = omega[j][2];
//AS TODO
		  buf[m++] = q[j][0];
		  buf[m++] = q[j][1];
		  buf[m++] = q[j][2];
		  buf[m++] = mu[j][0];
		  buf[m++] = mu[j][1];
		  buf[m++] = mu[j][2];
		  buf[m++] = p[j][0];
		  buf[m++] = p[j][1];
		  buf[m++] = p[j][2];
		  buf[m++] = s0[j][0];
		  buf[m++] = s0[j][1];
		  buf[m++] = s0[j][2];
		  buf[m++] = e[j][0];
		  buf[m++] = e[j][1];
		  buf[m++] = e[j][2];

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
///AS        buf[m++] = radius[j];
        buf[m++] = rmass[j];
        buf[m++] = density[j]; 
        buf[m++] = v[j][0];
        buf[m++] = v[j][1];
        buf[m++] = v[j][2];
        buf[m++] = omega[j][0];
        buf[m++] = omega[j][1];
        buf[m++] = omega[j][2];
///AS TODO
        buf[m++] = q[j][0];
        buf[m++] = q[j][1];
        buf[m++] = q[j][2];
        buf[m++] = mu[j][0];
        buf[m++] = mu[j][1];
        buf[m++] = mu[j][2];
        buf[m++] = p[j][0];
        buf[m++] = p[j][1];
        buf[m++] = p[j][2];
        buf[m++] = s0[j][0];
        buf[m++] = s0[j][1];
        buf[m++] = s0[j][2];
        buf[m++] = e[j][0];
        buf[m++] = e[j][1];
        buf[m++] = e[j][2];

/
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
///AS          buf[m++] = radius[j];
          buf[m++] = rmass[j];
          buf[m++] = density[j]; 
          buf[m++] = v[j][0];
          buf[m++] = v[j][1];
          buf[m++] = v[j][2];
          buf[m++] = omega[j][0];
          buf[m++] = omega[j][1];
          buf[m++] = omega[j][2];
////AS TODO
          buf[m++] = q[j][0];
          buf[m++] = q[j][1];
          buf[m++] = q[j][2];
          buf[m++] = mu[j][0];
          buf[m++] = mu[j][1];
          buf[m++] = mu[j][2];
          buf[m++] = p[j][0];
          buf[m++] = p[j][1];
          buf[m++] = p[j][2];
          buf[m++] = s0[j][0];
          buf[m++] = s0[j][1];
          buf[m++] = s0[j][2];
          buf[m++] = e[j][0];
          buf[m++] = e[j][1];
          buf[m++] = e[j][2];

/
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
///AS          buf[m++] = radius[j];
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
///AS TODO
            buf[m++] = q[j][0];
            buf[m++] = q[j][1];
            buf[m++] = q[j][2];
            buf[m++] = mu[j][0];
            buf[m++] = mu[j][1];
            buf[m++] = mu[j][2];
            buf[m++] = p[j][0];
            buf[m++] = p[j][1];
            buf[m++] = p[j][2];
            buf[m++] = s0[j][0];
            buf[m++] = s0[j][1];
            buf[m++] = s0[j][2];
            buf[m++] = e[j][0];
            buf[m++] = e[j][1];
            buf[m++] = e[j][2];

/
        }
      }
    }
  } */

  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecMCA::pack_comm_hybrid(int n, int *list, double *buf)
{
  int i,j,m;

///AS  if (radvary == 0)
    return 0;
/*
  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = ubuf(type[j]).d; 
///AS    buf[m++] = radius[j];
    buf[m++] = rmass[j];
    buf[m++] = density[j];
  }
  return m; */
}

/* ---------------------------------------------------------------------- */

void AtomVecMCA::unpack_comm(int n, int first, double *buf)
{
  int i,m,last;

///AS if (radvary == 0) {
  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    x[i][0] = buf[m++];
    x[i][1] = buf[m++];
    x[i][2] = buf[m++];
   }
/*  } else {
    m = 0;
    last = first + n;
    for (i = first; i < last; i++) {
      x[i][0] = buf[m++];
      x[i][1] = buf[m++];
      x[i][2] = buf[m++];
      type[i] = (int) ubuf(buf[m++]).i; 
///AS      radius[i] = buf[m++];
      rmass[i] = buf[m++];
      density[i] = buf[m++]; 
    }
  } */
}

/* ---------------------------------------------------------------------- */

void AtomVecMCA::unpack_comm_vel(int n, int first, double *buf)
{
  int i,m,last;

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
    omega[i][0] = buf[m++];
    omega[i][1] = buf[m++];
    omega[i][2] = buf[m++];

//AS TODO
    q[i][0] = buf[m++];
    q[i][1] = buf[m++];
    q[i][2] = buf[m++];
    mu[i][0] = buf[m++];
    mu[i][1] = buf[m++];
    mu[i][2] = buf[m++];
    p[i][0] = buf[m++];
    p[i][1] = buf[m++];
    p[i][2] = buf[m++];
    s0[i][0] = buf[m++];
    s0[i][1] = buf[m++];
    s0[i][2] = buf[m++];
    e[i][0] = buf[m++];
    e[i][1] = buf[m++];
    e[i][2] = buf[m++];


  }
/*
  } else {
    m = 0;
    last = first + n;
    for (i = first; i < last; i++) {
      x[i][0] = buf[m++];
      x[i][1] = buf[m++];
      x[i][2] = buf[m++];
      type[i] = (int) ubuf(buf[m++]).i; 
///AS      radius[i] = buf[m++];
      rmass[i] = buf[m++];
      density[i] = buf[m++]; 
      v[i][0] = buf[m++];
      v[i][1] = buf[m++];
      v[i][2] = buf[m++];
      omega[i][0] = buf[m++];
      omega[i][1] = buf[m++];
      omega[i][2] = buf[m++];
///AS TODO
      q[i][0] = buf[m++];
      q[i][1] = buf[m++];
      q[i][2] = buf[m++];
      mu[i][0] = buf[m++];
      mu[i][1] = buf[m++];
      mu[i][2] = buf[m++];
      p[i][0] = buf[m++];
      p[i][1] = buf[m++];
      p[i][2] = buf[m++];
      s0[i][0] = buf[m++];
      so[i][1] = buf[m++];
      so[i][2] = buf[m++];
      e[i][0] = buf[m++];
      e[i][1] = buf[m++];
      e[i][2] = buf[m++];
 
/
    }
  } */
}

/* ---------------------------------------------------------------------- */

int AtomVecMCA::unpack_comm_hybrid(int n, int first, double *buf)
{
  int i,m,last;

///AS  if (radvary == 0)
    return 0;
/*
  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    type[i] = (int) ubuf(buf[m++]).i; 
///AS    radius[i] = buf[m++];
    rmass[i] = buf[m++];
    density[i] = buf[m++]; 
  }
  return m; */
}

/* ---------------------------------------------------------------------- */

int AtomVecMCA::pack_reverse(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = f[i][0];
    buf[m++] = f[i][1];
    buf[m++] = f[i][2];
    buf[m++] = torque[i][0];
    buf[m++] = torque[i][1];
    buf[m++] = torque[i][2];
//AS TODO
    buf[m++] = p[i][0];
    buf[m++] = p[i][1];
    buf[m++] = p[i][2];
    buf[m++] = s0[i][0];
    buf[m++] = s0[i][1];
    buf[m++] = s0[i][2];
    buf[m++] = e[i][0];
    buf[m++] = e[i][1];
    buf[m++] = e[i][2];


  }
  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecMCA::pack_reverse_hybrid(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = torque[i][0];
    buf[m++] = torque[i][1];
    buf[m++] = torque[i][2];
//AS TODO
    buf[m++] = p[i][0];
    buf[m++] = p[i][1];
    buf[m++] = p[i][2];
    buf[m++] = s0[i][0];
    buf[m++] = s0[i][1];
    buf[m++] = s0[i][2];
    buf[m++] = e[i][0];
    buf[m++] = e[i][1];
    buf[m++] = e[i][2];

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
    f[j][0] += buf[m++];
    f[j][1] += buf[m++];
    f[j][2] += buf[m++];
    torque[j][0] += buf[m++];
    torque[j][1] += buf[m++];
    torque[j][2] += buf[m++];
//AS TODO
    p[j][0] += buf[m++];
    p[j][1] += buf[m++];
    p[j][2] += buf[m++];
    s0[j][0] += buf[m++];
    s0[j][1] += buf[m++];
    s0[j][2] += buf[m++];
    e[j][0] += buf[m++];
    e[j][1] += buf[m++];
    e[j][2] += buf[m++];

  }
}

/* ---------------------------------------------------------------------- */

int AtomVecMCA::unpack_reverse_hybrid(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    torque[j][0] += buf[m++];
    torque[j][1] += buf[m++];
    torque[j][2] += buf[m++];
//AS TODO
    p[j][0] += buf[m++];
    p[j][1] += buf[m++];
    p[j][2] += buf[m++];
    s0[j][0] += buf[m++];
    s0[j][1] += buf[m++];
    s0[j][2] += buf[m++];
    e[j][0] += buf[m++];
    e[j][1] += buf[m++];
    e[j][2] += buf[m++];

  }
  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecMCA::pack_border(int n, int *list, double *buf,
			     int pbc_flag, int *pbc)
{
  int i,j,m;
  double dx,dy,dz;

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
///AS      buf[m++] = radius[j];
      buf[m++] = rmass[j];
      buf[m++] = density[j];
      buf[m++] = ubuf(molecule[j]).d;
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
///AS      buf[m++] = radius[j];
      buf[m++] = rmass[j];
      buf[m++] = density[j];
      buf[m++] = ubuf(molecule[j]).d;
    }
  }
 
 if (atom->nextra_border)
    for (int iextra = 0; iextra < atom->nextra_border; iextra++)
      m += modify->fix[atom->extra_border[iextra]]->pack_border(n,list,&buf[m]);

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
///AS      buf[m++] = radius[j];
      buf[m++] = rmass[j];
      buf[m++] = density[j]; 

      buf[m++] = v[j][0];
      buf[m++] = v[j][1];
      buf[m++] = v[j][2];

      buf[m++] = omega[j][0];
      buf[m++] = omega[j][1];
      buf[m++] = omega[j][2];
//AS TODO
      buf[m++] = mu[j][0];
      buf[m++] = mu[j][1];
      buf[m++] = mu[j][2];
      buf[m++] = p[j][0];
      buf[m++] = p[j][1];
      buf[m++] = p[j][2];
      buf[m++] = s0[j][0];
      buf[m++] = s0[j][1];
      buf[m++] = s0[j][2];
      buf[m++] = e[j][0];
      buf[m++] = e[j][1];
      buf[m++] = e[j][2];  


      buf[m++] = ubuf(molecule[j]).d;
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

		  buf[m++] = ubuf(tag[j]).d;
		  buf[m++] = ubuf(type[j]).d;
		  buf[m++] = ubuf(mask[j]).d;
///AS		  buf[m++] = radius[j];
		  buf[m++] = rmass[j];
		  buf[m++] = density[j]; 

		  buf[m++] = v[j][0];
		  buf[m++] = v[j][1];
		  buf[m++] = v[j][2];

		  buf[m++] = omega[j][0];
		  buf[m++] = omega[j][1];
		  buf[m++] = omega[j][2];
//AS TODO
     		  buf[m++] = mu[j][0];
     		  buf[m++] = mu[j][1];
    		  buf[m++] = mu[j][2];
     		  buf[m++] = p[j][0];
   		  buf[m++] = p[j][1];
      		  buf[m++] = p[j][2];
      		  buf[m++] = s0[j][0];
      		  buf[m++] = s0[j][1];
      		  buf[m++] = s0[j][2];
      		  buf[m++] = e[j][0];
      		  buf[m++] = e[j][1];
      		  buf[m++] = e[j][2]; 



		  buf[m++] = ubuf(molecule[j]).d;
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
///AS		buf[m++] = radius[j];
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
//AS TODO

     		buf[m++] = mu[j][0];
     		buf[m++] = mu[j][1];
    		buf[m++] = mu[j][2];

     		buf[m++] = p[j][0];
   		buf[m++] = p[j][1];
      		buf[m++] = p[j][2];

      		buf[m++] = s0[j][0];
      		buf[m++] = s0[j][1];
      		buf[m++] = s0[j][2];

      		buf[m++] = e[j][0];
      		buf[m++] = e[j][1];
      		buf[m++] = e[j][2];
  

		buf[m++] = ubuf(molecule[j]).d;
      }
    }
  }

   if (atom->nextra_border)
    for (int iextra = 0; iextra < atom->nextra_border; iextra++)
      m += modify->fix[atom->extra_border[iextra]]->pack_border(n,list,&buf[m]);

  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecMCA::pack_border_hybrid(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
///AS    buf[m++] = radius[j];
    buf[m++] = rmass[j];
    buf[m++] = density[j];
    buf[m++] = ubuf(molecule[j]).d;
//AS TODO
    buf[m++] = q[j];

  }
  return m;
}

/* ---------------------------------------------------------------------- */

void AtomVecMCA::unpack_border(int n, int first, double *buf)
{
  int i,m,last;

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
///AS    radius[i] = buf[m++];
    rmass[i] = buf[m++];
    density[i] = buf[m++];
//AS TODO
    q[i] = buf[m++];

    molecule[i] = (int) ubuf(buf[m++]).i; // remove?
  }

  if (atom->nextra_border)
    for (int iextra = 0; iextra < atom->nextra_border; iextra++)
      m += modify->fix[atom->extra_border[iextra]]->
        unpack_border(n,first,&buf[m]);
}

/* ---------------------------------------------------------------------- */

void AtomVecMCA::unpack_border_vel(int n, int first, double *buf)
{
  int i,m,last;

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
///AS    radius[i] = buf[m++];
    rmass[i] = buf[m++];
    density[i] = buf[m++];
    v[i][0] = buf[m++];
    v[i][1] = buf[m++];
    v[i][2] = buf[m++];
    omega[i][0] = buf[m++];
    omega[i][1] = buf[m++];
    omega[i][2] = buf[m++];
//AS TODO
    q[i][0] = buf[m++];
    q[i][1] = buf[m++];
    q[i][2] = buf[m++];
    mu[i][0] = buf[m++];
    mu[i][1] = buf[m++];
    mu[i][2] = buf[m++];
    e[i][0] = buf[m++];
    e[i][1] = buf[m++];
    e[i][2] = buf[m++];

    
    molecule[i] = (int) ubuf(buf[m++]).i;  // remove?

  }

if (atom->nextra_border)
    for (int iextra = 0; iextra < atom->nextra_border; iextra++)
      m += modify->fix[atom->extra_border[iextra]]->
        unpack_border(n,first,&buf[m]);
}

/* ---------------------------------------------------------------------- */

int AtomVecMCA::unpack_border_hybrid(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
///AS    radius[i] = buf[m++];
    rmass[i] = buf[m++];
    density[i] = buf[m++];
//AS TODO
    q[i] = buf[m++];

    molecule[i] = (int) ubuf(buf[m++]).i; //remove?
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

///AS  buf[m++] = radius[i];
  buf[m++] = rmass[i];
  buf[m++] = density[i]; 
  buf[m++] = omega[i][0];
  buf[m++] = omega[i][1];
  buf[m++] = omega[i][2];
//AS TODO
  buf[m++] = q[i][0];
  buf[m++] = q[i][1];
  buf[m++] = q[i][2];  
  buf[m++] = mu[i][0];
  buf[m++] = mu[i][1];
  buf[m++] = mu[i][2];
  buf[m++] = p[i][0];
  buf[m++] = p[i][1];
  buf[m++] = p[i][2];
  buf[m++] = s0[i][0];
  buf[m++] = s0[i][1];
  buf[m++] = s0[i][2];
  buf[m++] = e[i][0];
  buf[m++] = e[i][1];
  buf[m++] = e[i][2]; 


  buf[m++] = ubuf(molecule[i]).d;

  buf[m++] = num_bond[i];
  for (k = 0; k < num_bond[i]; k++) {
    buf[m++] = bond_type[i][k];
    buf[m++] = bond_atom[i][k];
  }

  if(atom->n_bondhist)
  {
      for (k = 0; k < num_bond[i]; k++)
        for (l = 0; l < atom->n_bondhist; l++)
          buf[m++] = bond_hist[i][k][l];
  }

  buf[m++] = nspecial[i][0];
  buf[m++] = nspecial[i][1];
  buf[m++] = nspecial[i][2];
  for (k = 0; k < nspecial[i][2]; k++) buf[m++] = special[i][k];

  if (atom->nextra_grow)
    for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
      m += modify->fix[atom->extra_grow[iextra]]->pack_exchange(i,&buf[m]);

  buf[0] = m;
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

///AS  radius[nlocal] = buf[m++];
  rmass[nlocal] = buf[m++];
  density[nlocal] = buf[m++]; 
  omega[nlocal][0] = buf[m++];
  omega[nlocal][1] = buf[m++];
  omega[nlocal][2] = buf[m++];
//AS TODO
  q[nlocal][0] = buf[m++];
  q[nlocal][1] = buf[m++];
  q[nlocal][2] = buf[m++];
  mu[nlocal][0] = buf[m++];
  mu[nlocal][1] = buf[m++];
  mu[nlocal][2] = buf[m++];
  p[nlocal][0] = buf[m++];
  p[nlocal][1] = buf[m++];
  p[nlocal][2] = buf[m++];
  s0[nlocal][0] = buf[m++];
  s0[nlocal][1] = buf[m++];
  s0[nlocal][2] = buf[m++];
  e[nlocal][0] = buf[m++];
  e[nlocal][1] = buf[m++];
  e[nlocal][2] = buf[m++];
 

  molecule[nlocal] = (int) ubuf(buf[m++]).i; //remove?

  num_bond[nlocal] = (int) ubuf(buf[m++]).i;
  for (k = 0; k < num_bond[nlocal]; k++) {
    bond_type[nlocal][k] = (int) ubuf(buf[m++]).i;
    bond_atom[nlocal][k] = (int) ubuf(buf[m++]).i;
  }

  if(atom->n_bondhist)
  {
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
  int n = 0;
///AS It was in AtomSphere  int n = 17 * nlocal;  ///////////////

  for (i = 0; i < nlocal; i++)
  {
    n += 19 + 2*num_bond[i]; ///AS it was 13, we added 6 private variables, so total # is 19
/*AS TODO Calculate size of
  double *q;
  double **mu;
  double *p;
  double *s0;
  double *e;
*/

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

///AS  buf[m++] = radius[i];
  buf[m++] = rmass[i];
  buf[m++] = density[i];
  buf[m++] = omega[i][0];
  buf[m++] = omega[i][1];
  buf[m++] = omega[i][2];
//AS TODO
  buf[m++] = q[i][0];
  buf[m++] = q[i][1];
  buf[m++] = q[i][2]; 
  buf[m++] = mu[i][0];
  buf[m++] = mu[i][1];
  buf[m++] = mu[i][2]; 
  buf[m++] = p[i][0];
  buf[m++] = p[i][1];
  buf[m++] = p[i][2];
  buf[m++] = s0[i][0];
  buf[m++] = s0[i][1];
  buf[m++] = s0[i][2];  
  buf[m++] = e[i][0];
  buf[m++] = e[i][1];
  buf[m++] = e[i][2]; 


  buf[m++] = ubuf(molecule[i]).d; //remove?

  buf[m++] = ubuf(num_bond[i]).d;

  for (k = 0; k < num_bond[i]; k++) {
    buf[m++] = MAX(bond_type[i][k],-bond_type[i][k]);
    buf[m++] = bond_atom[i][k];
  }

  if(atom->n_bondhist)
  {
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

///AS  radius[nlocal] = buf[m++];
  rmass[nlocal] = buf[m++];
  density[nlocal] = buf[m++];
  omega[nlocal][0] = buf[m++];
  omega[nlocal][1] = buf[m++];
  omega[nlocal][2] = buf[m++];
//AS TODO
  q[nlocal][0] = buf[m++];
  q[nlocal][1] = buf[m++];
  q[nlocal][2] = buf[m++]; 
  mu[nlocal][0] = buf[m++];
  mu[nlocal][1] = buf[m++];
  mu[nlocal][2] = buf[m++]; 
  p[nlocal][0] = buf[m++];
  p[nlocal][1] = buf[m++];
  p[nlocal][2] = buf[m++]; 
  s0[nlocal][0] = buf[m++];
  s0[nlocal][1] = buf[m++];
  s0[nlocal][2] = buf[m++]; 
  e[nlocal][0] = buf[m++];
  e[nlocal][1] = buf[m++];
  e[nlocal][2] = buf[m++];  


  molecule[nlocal] = (int) ubuf(buf[m++]).i; //remove?


  num_bond[nlocal] = (int) ubuf(buf[m++]).i;
  for (k = 0; k < num_bond[nlocal]; k++) {
    bond_type[nlocal][k] = (int) ubuf(buf[m++]).i;
    bond_atom[nlocal][k] = (int) ubuf(buf[m++]).i;
  }

  if(atom->n_bondhist)
  {
      if(atom->n_bondhist != (int) ubuf(buf[m++]).i)
          error->all(FLERR,"Íncompatibel restart file: file was created using a bond model with a different number of history values");
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
fprintf(stderr,"AtomVecMCA::create_atom\n");
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

///AS  radius[nlocal] = 0.5;
  density[nlocal] = 1.0;
  rmass[nlocal] = get_init_volume() * density[nlocal]; 
fprintf(stderr, "rmass[%d]= %20.12e mca_radius= %g\n",nlocal,rmass[nlocal],atom->mca_radius);      
  omega[nlocal][0] = 0.0;
  omega[nlocal][1] = 0.0;
  omega[nlocal][2] = 0.0;

  q[nlocal] = 0.4*pow(3.0*get_init_volume()/(4.0*MY_PI), 2.0/3.0) * rmass[nlocal];
  mu[nlocal][0] = 0.0;
  mu[nlocal][1] = 0.0;
  mu[nlocal][2] = 0.0;
  p[nlocal] = 0.0;
  s0[nlocal] = 0.0;
  e[nlocal] = 0.0;

  atom->nlocal++;
}

/* ----------------------------------------------------------------------
   unpack one line from Atoms section of data file
   initialize other atom quantities
------------------------------------------------------------------------- */

void AtomVecMCA::data_atom(double *coord, tagint imagetmp, char **values)
{
fprintf(stderr,"AtomVecMCA::data_atom\n");
  int nlocal = atom->nlocal;
  if (nlocal == nmax) grow(0);

  tag[nlocal] = atoi(values[0]);
  if (tag[nlocal] <= 0)
    error->one(FLERR,"Invalid atom ID in Atoms section of data file");

  type[nlocal] = atoi(values[1]);
  if (type[nlocal] <= 0 || type[nlocal] > atom->ntypes)
    error->one(FLERR,"Invalid atom type in Atoms section of data file");

///AS  radius[nlocal] = 0.5 * atof(values[2]); We have to change other numbers!!!
///AS  if (radius[nlocal] <= 0.0)
///AS    error->one(FLERR,"Invalid radius in Atoms section of data file for 'atom_vec_mca'");

  density[nlocal] = atof(values[2]);
  if (density[nlocal] <= 0.0)
    error->one(FLERR,"Invalid density in Atoms section of data file");

  rmass[nlocal] = get_init_volume() * density[nlocal]; 

  molecule[nlocal] = atoi(values[3]); ///AS We move molecule parameter after all granular parameters

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

  q[nlocal] = 0.4*pow(3.0*get_init_volume()/(4.0*MY_PI), 2.0/3.0) * rmass[nlocal];
  mu[nlocal][0] = 0.0;
  mu[nlocal][1] = 0.0;
  mu[nlocal][2] = 0.0;
  p[nlocal] = 0.0;
  s0[nlocal] = 0.0;
  e[nlocal] = 0.0;

  num_bond[nlocal] = 0;

  atom->nlocal++;
}

/* ----------------------------------------------------------------------
   unpack hybrid quantities from one line in Atoms section of data file
   initialize other atom quantities for this sub-style
------------------------------------------------------------------------- */

int AtomVecMCA::data_atom_hybrid(int nlocal, char **values)
{
/*AS TODO
  double **mu;
  double *p;
  double *s0;
  double *e;
*/
  molecule[nlocal] = atoi(values[0]);

  num_bond[nlocal] = 0;

///AS  radius[nlocal] = 0.5 * atof(values[0]);
///AS  if (radius[nlocal] <= 0.0)
///AS    error->one(FLERR,"Invalid radius in Atoms section of data file");

  density[nlocal] = atof(values[1]);
  if (density[nlocal] <= 0.0)
    error->one(FLERR,"Invalid density in Atoms section of data file");

  rmass[nlocal] = get_init_volume() * density[nlocal]; 
  q[nlocal] = 0.4*pow(3.0*get_init_volume()/(4.0*MY_PI), 2.0/3.0) * rmass[nlocal];

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
///AS ???    buf[i][2] = 2.0*radius[i]; // Why 2.0!!!
///AS ??? mass?
///AS ???    if (radius[i] == 0.0) buf[i][3] = rmass[i];
///AS ???    else
///AS ???      buf[i][3] = rmass[i] / (4.0*MY_PI/3.0 * radius[i]*radius[i]*radius[i]);
    buf[i][2] = rmass[i] / get_init_volume();

/*AS TODO
  double *q; */
    buf[i][3] = x[i][0];
    buf[i][4] = x[i][1];
    buf[i][5] = x[i][2];
    buf[i][6] = ubuf((image[i] & IMGMASK) - IMGMAX).d;
    buf[i][7] = ubuf((image[i] >> IMGBITS & IMGMASK) - IMGMAX).d;
    buf[i][8] = ubuf((image[i] >> IMG2BITS) - IMGMAX).d;

    buf[i][9] = q[i][0];
    buf[i][10] = q[i][1];
    buf[i][11] = q[i][2];

///AS TODO ??? what we need to save else?
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
///AS ???    buf[i][2] = 2.0*radius[i]; // Why 2.0!!!
///AS ??? mass?
///AS ???    if (radius[i] == 0.0) buf[i][3] = rmass[i];
///AS ???    else
///AS ???      buf[i][3] = rmass[i] / (4.0*MY_PI/3.0 * radius[i]*radius[i]*radius[i]);
    buf[i][2] = rmass[i] / get_init_volume();
/*AS TODO
  double *q; */
    buf[i][3] = x[i][0];
    buf[i][4] = x[i][1];
    buf[i][5] = x[i][2];
    buf[i][6] = ubuf((image[i] & IMGMASK) - IMGMAX).d;
    buf[i][7] = ubuf((image[i] >> IMGBITS & IMGMASK) - IMGMAX).d;
    buf[i][8] = ubuf((image[i] >> IMG2BITS) - IMGMAX).d;

    buf[i][9] = q[i][0];
    buf[i][10] = q[i][1];
    buf[i][11] = q[i][2];
///AS TODO ??? what we need to save else?
  }
}

/* ----------------------------------------------------------------------
   pack hybrid atom info for data file
------------------------------------------------------------------------- */

int AtomVecMCA::pack_data_hybrid(int i, double *buf)
{
///AS TODO ??? what we need to save?
///AS TODO ???  buf[0] = 2.0*radius[i]; // Why 2.0!!!
///AS TODO ???  if (radius[i] == 0.0) buf[1] = rmass[i];
///AS TODO ???  else buf[1] = rmass[i] / (4.0*MY_PI/3.0 * radius[i]*radius[i]*radius[i]);
  buf[1] = rmass[i] / get_init_volume();
  return 2;
}

/* ----------------------------------------------------------------------
   write atom info to data file including 3 image flags
------------------------------------------------------------------------- */

void AtomVecMCA::write_data(FILE *fp, int n, double **buf)
{
///AS TODO ??? what we need to save?
  for (int i = 0; i < n; i++)
    fprintf(fp,"%d %d %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %d %d %d\n",
            (int) ubuf(buf[i][0]).i,(int) ubuf(buf[i][1]).i,
            buf[i][2],buf[i][3],
            buf[i][4],buf[i][5],buf[i][6],
            (int) ubuf(buf[i][7]).i,(int) ubuf(buf[i][8]).i,
            (int) ubuf(buf[i][9]).i);
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
///AS ???  if (atom->memcheck("radius")) bytes += memory->usage(radius,nmax);
  if (atom->memcheck("rmass")) bytes += memory->usage(rmass,nmax);
  if (atom->memcheck("omega")) bytes += memory->usage(omega,nmax,3);
  if (atom->memcheck("torque")) bytes += memory->usage(torque,nmax*comm->nthreads,3);

  if (atom->memcheck("q")) bytes += memory->usage(q,nmax);
  if (atom->memcheck("mu")) bytes += memory->usage(mu,nmax,3);
///AS TODO ??? comm->nthreads - do we need to pass them to other processors
  if (atom->memcheck("p")) bytes += memory->usage(p,nmax*comm->nthreads);
  if (atom->memcheck("s0")) bytes += memory->usage(s0,nmax*comm->nthreads);
  if (atom->memcheck("e")) bytes += memory->usage(e,nmax*comm->nthreads);

  if (atom->memcheck("molecule")) bytes += memory->usage(molecule,nmax);
  if (atom->memcheck("nspecial")) bytes += memory->usage(nspecial,nmax,3);
  if (atom->memcheck("special"))  bytes += memory->usage(special,nmax,atom->maxspecial);
  if (atom->memcheck("num_bond")) bytes += memory->usage(num_bond,nmax);
  if (atom->memcheck("bond_type")) bytes += memory->usage(bond_type,nmax,atom->bond_per_atom);
  if (atom->memcheck("bond_atom")) bytes += memory->usage(bond_atom,nmax,atom->bond_per_atom);
  if(atom->n_bondhist) bytes += nmax*sizeof(int);  //!! P.F. not sure about atom->n_bondhist

  return bytes;
}

/* ----------------------------------------------------------------------
  compute initial volume of cellular automaton based on radius and packing
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
  } else if(atom->packing = FCC){
    return 4.0*sqrt(2.0)*mca_radius*mca_radius*mca_radius;
  } else if(atom->packing = HCP) {
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
  } else if(atom->packing = FCC){
    return sqrt(2.0)*mca_radius*mca_radius;
  } else if(atom->packing = HCP) {
    return sqrt(2.0)*mca_radius*mca_radius;
  } else {
    error->all(FLERR,"Illegal packing in AtomVecMCA::get_contact_area()");
    return 1.0;
  }
}
