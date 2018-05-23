/**************************************************************************
 * Perpendicular Laplacian inversion. 
 *                           Using Algebraic multigrid Solver
 *                              with PETSc library
 *
 * Equation solved is:
 *  d*\nabla^2_\perp x + (1/c1)\nabla_perp c2\cdot\nabla_\perp x + a x = b
 *
 **************************************************************************
 * Copyright 2018 K.S. Kang kskang@ipp.mpg.de
 *
 * Contact: Ben Dudson, bd512@york.ac.uk
 * 
 * This file is part of BOUT++.
 *
 * BOUT++ is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * BOUT++ is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with BOUT++.  If not, see <http://www.gnu.org/licenses/>.
 *
 **************************************************************************/

#include "petscamg.hxx"

void LaplacePetscAmg::settingSolver(int kflag){
  //  Timer timer("invert");
  TRACE("LaplacePetscAmg::settingSolver(int)");
  
  if(!opts) {
    // If no options supplied, use default
    opts = Options::getRoot()->getSection("petscamg");
  }

  //////////////////////////////////////////////////
  // Set up KSP
  
  // Declare KSP Context 
  KSPCreate( commX, &ksp ); 
  KSPGetPC(ksp, &pc);
  
  // Configure Linear Solver
  if(kflag == 0) KSPSetOperators( ksp, MatA, MatA );
  else KSPSetOperators( ksp, MatA, MatP );
   // Convergence Parameters. Solution is considered converged if |r_k| < max( rtol * |b| , atol )
    // where r_k = b - Ax_k. The solution is considered diverged if |r_k| > dtol * |b|.
  
  if(soltype == "direct") {
    KSPSetType(ksp,KSPPREONLY);
    PCSetType(pc,PCLU);
  }
  else {
    KSPSetType( ksp, KSPGMRES );
    KSPSetInitialGuessNonzero( ksp, (PetscBool) true );
    if(soltype == "gmres") {
      PCSetType(pc,PCILU);
    }
    else {
      
      if(soltype == "gamg" ) {
	PCSetType(pc,PCGAMG);
      }
      else if(soltype == "gamggeo" ) {
	PCSetType(pc,PCGAMG);
        PCGAMGSetType(pc,PCGAMGGEO);
      }
      else if(soltype == "gamgag") {
        PCSetType(pc,PCGAMG);
        PCGAMGSetType(pc,PCGAMGAGG);
      }
      else if(soltype == "gamgag1") {
        PCSetType(pc,PCGAMG);
        PCGAMGSetType(pc,PCGAMGAGG);
	PCGAMGSetNSmooths(pc,1);
      }
      PCMGSetCycleType(pc,PC_MG_CYCLE_V);
      PCGAMGSetNlevels(pc,mglevel);
      PCGAMGSetNSmooths(pc,2);
    }
    if(rightpre) KSPSetPCSide(ksp, PC_RIGHT); // Right preconditioning
    else          KSPSetPCSide(ksp, PC_LEFT);  // Left preconditioning

  }
  KSPSetTolerances(ksp,rtol,atol,dtol,maxits);  
  KSPSetFromOptions( ksp );
}

const FieldPerp LaplacePetscAmg::solve(const FieldPerp &rhs, const FieldPerp &x0) {
  // Timer timer("invert");
  TRACE("LaplacePetscAmg::solve(const FieldPerp, const FieldPerp)");
  
  // Load initial guess x0 into xs and rhs into bs
  Mesh *mesh = rhs.getMesh();  // Where to get initializing LaplacePetscAmg
  Coordinates *coords = mesh->coordinates();
  int yindex = rhs.getIndex();
  int ind,i2,i,k,k2;
  PetscScalar val;
  // mxstart = xgstart;
  // mzstart = zgstart;
  if ( global_flags & INVERT_START_NEW ) {
    // set initial guess to zero
    for (i=0; i < Nx_local; i++) {
      for (k=0; k < Nz_local; k++) {
        ind = gindices[(i+lxs)*nzt+k+lzs];
        val = 0.;
        VecSetValues( xs, 1, &ind, &val, INSERT_VALUES );
      
        val = rhs(i+mxstart,k+mzstart);
        VecSetValues( bs, 1, &ind, &val, INSERT_VALUES );
      }
    }
  }
  else {
    // Read initial guess into local array, ignoring guard cells
    for (i=0; i < Nx_local; i++) {
      for (k=0; k < Nz_local; k++) {
        ind = gindices[(i+lxs)*nzt+k+lzs];
        val = x0(i+mxstart,k+mzstart);
        VecSetValues( xs, 1, &ind, &val, INSERT_VALUES );
      
        val = rhs(i+mxstart,k+mzstart);
        VecSetValues( bs, 1, &ind, &val, INSERT_VALUES );
      }
    }
  }
  // For the boundary conditions 
  BoutReal tval[nzt],dval,ddx_C,ddz_C;
  if (mesh->firstX()) {
    i2 = mesh->xstart;
    if ( inner_boundary_flags & INVERT_AC_GRAD ) {
      // Neumann boundary condition
      if ( inner_boundary_flags & INVERT_SET ) {
        // sEY THE VALUE guard cells specify gradient to set at inner boundary
	// tval = df/dn = (v_ghost - v_in)/distance 
        for (k = 0; k < nzt; k++) {
	  tval[k] = -x0(i2-1, k+mzstart-lzs)*sqrt(coords->g_11(i2, yindex))*coords->dx(i2, yindex); 
        }
      }
      else {
        // zero gradient inner boundary condition
        for (int k = 0; k<nzt; k++) {
          // set inner guard cells
          tval[k] = 0.0;
        }
      }
    }
    else {      // Dirichlet boundary condition
      if ( inner_boundary_flags & INVERT_SET ) {
        // guard cells of x0 specify value to set at inner boundary
	// tval = f = (v_ghost + v_in)/2.0
        for (int k = 0; k < nzt; k++) {
          tval[k] = 2.*x0(i2-1, k+mzstart-lzs); 
        // this is the value to set at the inner boundary
        }
      }
      else {
        // zero value inner boundary condition
        for (int k=0; k < nzt; k++) {
          // set inner guard cells
          tval[k] = 0.;
        }
      }
    }
    for(k = 0;k < Nz_local;k++) {
      k2 = k + mzstart;
      ddx_C = (C2(i2+1, yindex, k2) - C2(i2-1, yindex, k2))/2./coords->dx(i2, yindex)/C1(i2, yindex, k2);
      ddz_C = (C2(i2, yindex, k2+nzt) - C2(i2, yindex, k2-nzt)) /2./coords->dz/C1(i2, yindex, k2);
      dval = D(i2, yindex, k2)*coords->g11(i2, yindex)/coords->dx(i2, yindex)/coords->dx(i2, yindex);
      dval -= (D(i2, yindex, k2)*2.*coords->G1(i2, yindex) + coords->g11(i2, yindex)*ddx_C
                   + coords->g13(i2, yindex)*ddz_C)/coords->dx(i2, yindex)/2.0;
      val = -tval[k]*dval;
      dval = D(i2, yindex, k2)*coords->g13(i2, yindex)/coords->dx(i2, yindex)/coords->dz/4.;
      if(lzs == 0 && k == 0) val -= dval*tval[nzt-1];
      else val -= dval*tval[k-1];
      if(lzs == 0 && k == nzt-1) val += dval*tval[0];
      else val += dval*tval[k+1];
      ind = gindices[k+lzs];
      VecSetValues( bs, 1, &ind, &val, ADD_VALUES );
    }  
  }
  if (mesh->lastX()) {
    i2 = mesh->xend;
    if ( outer_boundary_flags & INVERT_AC_GRAD ) {
      // Neumann boundary condition
      if ( inner_boundary_flags & INVERT_SET ) {
        // guard cells of x0 specify gradient to set at outer boundary
	// tval = df/dn = (v_ghost - v_in)/distance 
        for (k= 0; k < nzt; k++) {
          tval[k] = x0(i2+1, k+mzstart-lzs)*sqrt(coords->g_11(i2, yindex))*coords->dx(i2, yindex); 
        // this is the value to set the gradient to at the outer boundary
        }
      }
      else {
        // zero gradient outer boundary condition
        for (k=0; k<nzt; k++) {
          // set outer guard cells
          tval[k] = 0.;
        }
      }
    }
    else {
      // Dirichlet boundary condition
      if ( outer_boundary_flags & INVERT_SET ) {
        // guard cells of x0 specify value to set at outer boundary
        for (k=0; k< nzt; k++) {
          int k2 = k-1;
          tval[k]=2.*x0(i2+1, k+mzstart-lzs); 
          // this is the value to set at the outer boundary
        }
      }
      else {
        // zero value inner boundary condition
        for (int k=0; k<nzt; k++) {
          // set outer guard cells
          tval[k] = 0.;
        }
      }
    }
    for(k = 0;k < Nz_local;k++) {
      k2 = k+mzstart;
      ddx_C = (C2(i2+1, yindex, k2) - C2(i2-1, yindex, k2))/2./coords->dx(i2, yindex)/C1(i2, yindex, k2);
      ddz_C = (C2(i2, yindex, k2+nzt) - C2(i2, yindex, k2-nzt)) /2./coords->dz/C1(i2, yindex, k2);
      dval = D(i2, yindex, k2)*coords->g11(i2, yindex)/coords->dx(i2, yindex)/coords->dx(i2, yindex);
      dval += (D(i2, yindex, k2)*2.*coords->G1(i2, yindex) + coords->g11(i2, yindex)*ddx_C
                   + coords->g13(i2, yindex)*ddz_C)/coords->dx(i2, yindex)/2.0;
      val = -tval[k]*dval;
      dval = D(i2, yindex, k2)*coords->g13(i2, yindex)/coords->dx(i2, yindex)/coords->dz/4.;
      if(lzs == 0 && k == 0) val += dval*tval[nzt-1];
      else val += dval*tval[k-1];
      if(lzs == 0 && k == nzt-1) val -= dval*tval[0];
      else val -= dval*tval[k+1];
      ind = gindices[(nxt-1)*nzt+k+lzs];
      VecSetValues( bs, 1, &ind, &val, ADD_VALUES );
    }      
  }
  
  // Assemble RHS Vector
  VecAssemblyBegin(bs);
  VecAssemblyEnd(bs);

  // Assemble Trial Solution Vector
  VecAssemblyBegin(xs);
  VecAssemblyEnd(xs);
  
  // Solve the system
  KSPSolve( ksp, bs, xs );
  
  KSPConvergedReason reason;
  KSPGetConvergedReason( ksp, &reason );
  
  if(reason <= 0) {
    throw BoutException("LaplacePetscAmg failed to converge. Reason %d", reason);
  }
  
  //////////////////////////
  // Copy data into result
  
  FieldPerp result(mesh);
  result.allocate();
  
  for(i = 0;i < Nx_local;i++) {
    for(k= 0;k < Nz_local;k++) {
      ind = gindices[(i+lxs)*nzt+k+lzs];
      VecGetValues(xs, 1, &ind, &val );
      result(i+mesh->xstart,k+mzstart) = val;
    }
  }
  
  // Inner X boundary approximations on guard cells
  // Need to modify
  
  if(mesh->firstX()) {
    i2 = mesh->xstart;
    if ( inner_boundary_flags & INVERT_AC_GRAD ) {
      // Set THE VALUE guard cells specify gradient to set at inner boundary
      // tval = df/dn = (v_ghost - v_in)/distance 
      for (k = 0; k < Nz_local; k++) {
	val = -x0(i2-1, k+mzstart)*sqrt(coords->g_11(i2, yindex))*coords->dx(i2, yindex); 
        result(i2-1,k+mzstart) = val + result(i2,k+mzstart);
      }
    }
    else {      // Dirichlet boundary condition
        // guard cells of x0 specify value to set at inner boundary
	// tval = f = (v_ghost + v_in)/2.0
      for (int k = 0; k < Nz_local; k++) {
          result(i2-1,k+mzstart) = 2.*x0(i2-1, k+mzstart) - result(i2,k+mzstart); 
        // this is the value to set at the inner boundary
      }
    }
  }

  // Outer X boundary
  if (mesh->lastX()) {
    i2 = mesh->xend;
    if ( outer_boundary_flags & INVERT_AC_GRAD ) {
      // Neumann boundary condition
        // guard cells of x0 specify gradient to set at outer boundary
	// tval = df/dn = (v_ghost - v_in)/distance 
      for (k= 0; k < Nz_local; k++) {
        tval[k] = x0(i2+1, k+mzstart)*sqrt(coords->g_11(i2, yindex))*coords->dx(i2, yindex); 
        result(i2+1,k+mzstart) = val + result(i2,k+mzstart);
        // this is the value to set the gradient to at the outer boundary
      }
    }
    else {
      // Dirichlet boundary condition
      // guard cells of x0 specify value to set at outer boundary
      for (k=0; k< Nz_local; k++) {
        result(i2+1,k+mzstart) = 2.*x0(i2+1, k+mzstart-lzs) - result(i2,k+mzstart); 
          // this is the value to set at the outer boundary
      }

    }
  }
  result.setIndex(yindex);
  // Set the index of the FieldPerp to be returned
  return result;
}

