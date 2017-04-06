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

#include <string.h>
#include <stdlib.h>
#include "fix_mca_setvel.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "domain.h"
#include "region.h"
#include "respa.h"
#include "input.h"
#include "variable.h"
#include "memory.h"
#include "error.h"
#include "force.h"

using namespace LAMMPS_NS;
using namespace FixConst;

enum {
	NONE, CONSTANT, EQUAL, ATOM
};

/* ---------------------------------------------------------------------- */

FixMCASetVel::FixMCASetVel(LAMMPS *lmp, int narg, char **arg) :
		Fix(lmp, narg, arg) {
	if (narg < 6)
		error->all(FLERR, "Illegal fix mca/setvelocity command");

///LAMMPS	dynamic_group_allow = 1;
	vector_flag = 1;
	size_vector = 3;
	global_freq = 1;
	extvector = 1;

	xstr = ystr = zstr = NULL;
	xvalue = yvalue = zvalue = 0.;

	if (strstr(arg[3], "v_") == arg[3]) {
		int n = strlen(&arg[3][2]) + 1;
		xstr = new char[n];
		strcpy(xstr, &arg[3][2]);
	} else if (strcmp(arg[3], "NULL") == 0) {
		xstyle = NONE;
	} else {
		xvalue = force->numeric(FLERR, arg[3]);
		xstyle = CONSTANT;
	}
	if (strstr(arg[4], "v_") == arg[4]) {
		int n = strlen(&arg[4][2]) + 1;
		ystr = new char[n];
		strcpy(ystr, &arg[4][2]);
	} else if (strcmp(arg[4], "NULL") == 0) {
		ystyle = NONE;
	} else {
		yvalue = force->numeric(FLERR, arg[4]);
		ystyle = CONSTANT;
	}
	if (strstr(arg[5], "v_") == arg[5]) {
		int n = strlen(&arg[5][2]) + 1;
		zstr = new char[n];
		strcpy(zstr, &arg[5][2]);
	} else if (strcmp(arg[5], "NULL") == 0) {
		zstyle = NONE;
	} else {
		zvalue = force->numeric(FLERR, arg[5]);
		zstyle = CONSTANT;
	}

	// optional args

	iregion = -1;
	idregion = NULL;

	int iarg = 6;
	while (iarg < narg) {
		if (strcmp(arg[iarg], "region") == 0) {
			if (iarg + 2 > narg)
				error->all(FLERR, "Illegal fix mca/setvelocity command");
			iregion = domain->find_region(arg[iarg + 1]);
			if (iregion == -1)
				error->all(FLERR, "Region ID for fix mca/setvelocity does not exist");
			int n = strlen(arg[iarg + 1]) + 1;
			idregion = new char[n];
			strcpy(idregion, arg[iarg + 1]);
			iarg += 2;
		} else
			error->all(FLERR, "Illegal fix mca/setvelocity command");
	}

	force_flag = 0;
	foriginal[0] = foriginal[1] = foriginal[2] = 0.0;

	maxatom = atom->nmax;
	memory->create(sforce, maxatom, 3, "mca/setvelocity:sforce");
}

/* ---------------------------------------------------------------------- */

FixMCASetVel::~FixMCASetVel() {
	delete[] xstr;
	delete[] ystr;
	delete[] zstr;
	delete[] idregion;
	memory->destroy(sforce);
}

/* ---------------------------------------------------------------------- */

int FixMCASetVel::setmask() {
	int mask = 0;
	//mask |= INITIAL_INTEGRATE;
	mask |= POST_FORCE;
	return mask;
}

/* ---------------------------------------------------------------------- */

int FixMCASetVel::modify_param(int narg, char **arg)
{
	return 0;// Does no work!
	if (narg < 5)
		error->fix_error(FLERR,this, "Illegal fix_modify mca/setvelocity command");

///LAMMPS	dynamic_group_allow = 1;
	vector_flag = 1;
	size_vector = 3;
	global_freq = 1;
	extvector = 1;

///	xstr = ystr = zstr = NULL;

	int iarg = 2;
	if (strstr(arg[iarg], "v_") == arg[iarg]) {
		int n = strlen(&arg[iarg][2]) + 1;
		if (xstr) delete[] xstr;
		xstr = new char[n];
		strcpy(xstr, &arg[iarg][2]);
	} else if (strcmp(arg[iarg], "NULL") == 0) {
		xstyle = NONE;
	} else {
		xvalue = force->numeric(FLERR, arg[iarg]);
		xstyle = CONSTANT;
	}
	iarg = 3;
	if (strstr(arg[iarg], "v_") == arg[iarg]) {
		int n = strlen(&arg[iarg][2]) + 1;
		if (ystr) delete[] ystr;
		ystr = new char[n];
		strcpy(ystr, &arg[iarg][2]);
	} else if (strcmp(arg[iarg], "NULL") == 0) {
		ystyle = NONE;
	} else {
		yvalue = force->numeric(FLERR, arg[iarg]);
		ystyle = CONSTANT;
	}
	iarg = 4;
	if (strstr(arg[iarg], "v_") == arg[iarg]) {
		int n = strlen(&arg[iarg][2]) + 1;
		if (zstr) delete[] zstr;
		zstr = new char[n];
		strcpy(zstr, &arg[iarg][2]);
	} else if (strcmp(arg[iarg], "NULL") == 0) {
		zstyle = NONE;
	} else {
		zvalue = force->numeric(FLERR, arg[iarg]);
		zstyle = CONSTANT;
	}

	// optional args

	iregion = -1;
	idregion = NULL;

	iarg = 5;
	while (iarg < narg) {
		if (strcmp(arg[iarg], "region") == 0) {
			if (iarg + 2 > narg)
				error->fix_error(FLERR,this, "Illegal fix_modify mca/setvelocity command");
			iregion = domain->find_region(arg[iarg + 1]);
			if (iregion == -1)
				error->fix_error(FLERR,this, "Region ID for fix_modify mca/setvelocity does not exist");
			int n = strlen(arg[iarg + 1]) + 1;
			if (idregion) delete[] idregion;
			idregion = new char[n];
			strcpy(idregion, arg[iarg + 1]);
			iarg += 2;
		} else
			error->fix_error(FLERR,this, "Illegal fix_modify mca/setvelocity command");
	}

	force_flag = 0;
	foriginal[0] = foriginal[1] = foriginal[2] = 0.0;

///	maxatom = atom->nmax;
///	memory->create(sforce, maxatom, 3, "mca/setvelocity:sforce");
	init();
	return iarg;
}

void FixMCASetVel::init() {
	// check variables

	if (xstr) {
		xvar = input->variable->find(xstr);
		if (xvar < 0)
			error->all(FLERR, "Variable name for fix mca/setvelocity does not exist");
		if (input->variable->equalstyle(xvar))
			xstyle = EQUAL;
		else if (input->variable->atomstyle(xvar))
			xstyle = ATOM;
		else
			error->all(FLERR, "Variable for fix mca/setvelocity is of invalid style");
	}
	if (ystr) {
		yvar = input->variable->find(ystr);
		if (yvar < 0)
			error->all(FLERR, "Variable name for fix mca/setvelocity does not exist");
		if (input->variable->equalstyle(yvar))
			ystyle = EQUAL;
		else if (input->variable->atomstyle(yvar))
			ystyle = ATOM;
		else
			error->all(FLERR, "Variable for fix mca/setvelocity is of invalid style");
	}
	if (zstr) {
		zvar = input->variable->find(zstr);
		if (zvar < 0)
			error->all(FLERR, "Variable name for fix mca/setvelocity does not exist");
		if (input->variable->equalstyle(zvar))
			zstyle = EQUAL;
		else if (input->variable->atomstyle(zvar))
			zstyle = ATOM;
		else
			error->all(FLERR, "Variable for fix mca/setvelocity is of invalid style");
	}

	// set index and check validity of region

	if (iregion >= 0) {
		iregion = domain->find_region(idregion);
		if (iregion == -1)
			error->all(FLERR, "Region ID for fix mca/setvelocity does not exist");
	}

	if (xstyle == ATOM || ystyle == ATOM || zstyle == ATOM)
		varflag = ATOM;
	else if (xstyle == EQUAL || ystyle == EQUAL || zstyle == EQUAL)
		varflag = EQUAL;
	else
		varflag = CONSTANT;

	if (strstr(update->integrate_style, "respa"))
		nlevels_respa = ((Respa *) update->integrate)->nlevels;

	// cannot use non-zero forces for a minimization since no energy is integrated
	// use fix addforce instead

	int flag = 0;
	if (update->whichflag == 2) {
		if (xstyle == EQUAL || xstyle == ATOM)
			flag = 1;
		if (ystyle == EQUAL || ystyle == ATOM)
			flag = 1;
		if (zstyle == EQUAL || zstyle == ATOM)
			flag = 1;
		if (xstyle == CONSTANT && xvalue != 0.0)
			flag = 1;
		if (ystyle == CONSTANT && yvalue != 0.0)
			flag = 1;
		if (zstyle == CONSTANT && zvalue != 0.0)
			flag = 1;
	}
	if (flag)
		error->all(FLERR, "Cannot use non-zero forces in an energy minimization");
}

/* ---------------------------------------------------------------------- */

void FixMCASetVel::setup(int vflag) {
fprintf(logfile, "FixMCASetVel::setup() \n"); ///AS DEBUG

	if (strstr(update->integrate_style, "verlet"))
		post_force(vflag); /// why??
	else
		for (int ilevel = 0; ilevel < nlevels_respa; ilevel++) {
			((Respa *) update->integrate)->copy_flevel_f(ilevel);
			post_force_respa(vflag, ilevel, 0);
			((Respa *) update->integrate)->copy_f_flevel(ilevel);
		}
}

/* ---------------------------------------------------------------------- */

void FixMCASetVel::min_setup(int vflag) {
	post_force(vflag);/// why??
}

/* ---------------------------------------------------------------------- */

//void FixMCASetVel::initial_integrate(int vflag) {
void FixMCASetVel::post_force(int vflag) {
	double **x = atom->x;
	double **f = atom->f;
	double **v = atom->v;
        double **omega = atom->omega;
	int *mask = atom->mask;
	int nlocal = atom->nlocal;

	// update region if necessary

	Region *region = NULL;
	if (iregion >= 0) {
		region = domain->regions[iregion];
///LAMMPS		region->prematch();
	}

	// reallocate sforce array if necessary

	if (varflag == ATOM && nlocal > maxatom) {
		maxatom = atom->nmax;
		memory->destroy(sforce);
		memory->create(sforce, maxatom, 3, "mca/setvelocity:sforce");
	}

	foriginal[0] = foriginal[1] = foriginal[2] = 0.0;
	force_flag = 0;

	if (varflag == CONSTANT) {
		for (int i = 0; i < nlocal; i++)
			if (mask[i] & groupbit) {
				if (region && !region->match(x[i][0], x[i][1], x[i][2]))
					continue;
				foriginal[0] += f[i][0];
				foriginal[1] += f[i][1];
				foriginal[2] += f[i][2];
				if (xstyle) {
					v[i][0] = xvalue;
//                                        omega[i][0] = omega[i][1] = omega[i][2] = 0.0;
					f[i][0] = 0.0;
				}
				if (ystyle) {
					v[i][1] = yvalue;
					f[i][1] = 0.0;
//                                        omega[i][0] = omega[i][1] = omega[i][2] = 0.0;
				}
				if (zstyle) {
					v[i][2] = zvalue;
//                                        omega[i][0] = omega[i][1] = omega[i][2] = 0.0;
					f[i][2] = 0.0;
				}
			}

		// variable force, wrap with clear/add

	} else {

		modify->clearstep_compute();

		if (xstyle == EQUAL)
			xvalue = input->variable->compute_equal(xvar);
		else if (xstyle == ATOM)
			input->variable->compute_atom(xvar, igroup, &sforce[0][0], 3, 0);
		if (ystyle == EQUAL)
			yvalue = input->variable->compute_equal(yvar);
		else if (ystyle == ATOM)
			input->variable->compute_atom(yvar, igroup, &sforce[0][1], 3, 0);
		if (zstyle == EQUAL)
			zvalue = input->variable->compute_equal(zvar);
		else if (zstyle == ATOM)
			input->variable->compute_atom(zvar, igroup, &sforce[0][2], 3, 0);

		modify->addstep_compute(update->ntimestep + 1);

		//fprintf(logfile, "setting velocity at timestep %d\n", update->ntimestep);

		for (int i = 0; i < nlocal; i++)
			if (mask[i] & groupbit) {
				if (region && !region->match(x[i][0], x[i][1], x[i][2]))
					continue;
				foriginal[0] += f[i][0];
				foriginal[1] += f[i][1];
				foriginal[2] += f[i][2];
				if (xstyle == ATOM) {
					v[i][0] = sforce[i][0];
//                                        omega[i][0] = omega[i][1] = omega[i][2] = 0.0;
					f[i][0] = 0.0;
				} else if (xstyle) {
					v[i][0] = xvalue;
//                                        omega[i][0] = omega[i][1] = omega[i][2] = 0.0;
					f[i][0] = 0.0;
				}

				if (ystyle == ATOM) {
					v[i][1] = sforce[i][1];
//                                        omega[i][0] = omega[i][1] = omega[i][2] = 0.0;
					f[i][1] = 0.0;
				} else if (ystyle) {
					v[i][1] = yvalue;
//                                        omega[i][0] = omega[i][1] = omega[i][2] = 0.0;
					f[i][1] = 0.0;
				}

				if (zstyle == ATOM) {
					v[i][2] = sforce[i][2];
//                                        omega[i][0] = omega[i][1] = omega[i][2] = 0.0;
					f[i][2] = 0.0;
				} else if (zstyle) {
					v[i][2] = zvalue;
//                                        omega[i][0] = omega[i][1] = omega[i][2] = 0.0;
					f[i][2] = 0.0;
				}

			}
	}
}

/* ----------------------------------------------------------------------
 return components of total force on fix group before force was changed
 ------------------------------------------------------------------------- */

double FixMCASetVel::compute_vector(int n) {
// only sum across procs one time

	if (force_flag == 0) {
		MPI_Allreduce(foriginal, foriginal_all, 3, MPI_DOUBLE, MPI_SUM, world);
		force_flag = 1;
	}
	return foriginal_all[n];
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based array
 ------------------------------------------------------------------------- */

double FixMCASetVel::memory_usage() {
	double bytes = 0.0;
	if (varflag == ATOM)
		bytes = atom->nmax * 3 * sizeof(double);
	return bytes;
}
