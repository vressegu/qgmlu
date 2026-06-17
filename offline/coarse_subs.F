c**************************************************************************
c
c     This module includes all subroutines of filtering for performing
c     the program coarse_grain.
c
c     Written by Long Li, 10 Aug. 2020.  
c      
c**************************************************************************      
c
      MODULE coarse_subs

      IMPLICIT NONE
      PRIVATE

*     Subroutines      
      PUBLIC :: gaussf, pfilt, pfilt_cycl, p2uv

      CONTAINS

c**************************************************************************      
c
      SUBROUTINE gaussf (kerl, wd)

*     Construct 2D Gaussian filter kernel kerl of width wd. 
*     The total weight wt is also calculated.

      IMPLICIT NONE    
      
*     I/O arguments
      integer, INTENT(IN) :: wd
      double precision, INTENT(OUT) :: kerl(wd+1,wd+1)

*     Local variables      
      integer i,j,c
      double precision pi,wt

      wt = 0.0d0
      pi = 3.14159265358979324D0
      c = wd/2 + 1 

      do i=1,wd+1  
        do j=1,wd+1
*         Gaussian convolution kernel        
          kerl(i,j) = ( 6.0d0/(pi*(wd**2)) ) *
     &                exp( - 6.0d0*( (i-c)**2 + (j-c)**2 ) / (wd**2) ) 
          wt = wt + kerl(i,j)
        enddo
      enddo

*     Normalization  
      do i=1,wd+1  
        do j=1,wd+1
          kerl(i,j) = kerl(i,j)/wt
        enddo
      enddo
      
      END SUBROUTINE gaussf
c
c**************************************************************************
c      
      SUBROUTINE pfilt (pf, pr, p, fk, wd)

*     Coarse-graining of eddy-resolving snapshot.

*     Modules          
      USE param

      IMPLICIT NONE

*     I/O arguments
      integer, INTENT(IN) :: wd
      double precision, INTENT(IN) :: fk(wd+1,wd+1),p(nxpo,nypo)
      double precision, INTENT(OUT), DIMENSION(nxpoc,nypoc) :: pf,pr
*
*     fk is filter kernel of total weight wt and width wd
*     p is eddy-resolving presure (m^2/s^2)
*     pf and pr are corase-grained and residual pressure (m^2/s^2)

*     Local variables
      integer nh,i,j,ii,jj,il,jl,ig,jg
      double precision psum,wsum,rdom,pm,pfm,bcfc,pbc

      nh = wd/2 + 1 !! size of half-kernel
      rdom = 1.0d0/dble(nso*nso)
      bcfc = 1.0d0/dble(nxtoc + nytoc - 1)
      pf = 0.0d0

!$OMP PARALLEL DEFAULT (NONE)
!$OMP&         PRIVATE (i,j,ii,jj,il,jl,ig,jg,psum,wsum)
!$OMP&         SHARED  (p,pf,pr,fk,wd,nh,pm,rdom,pfm,pbc,bcfc)

*     Original mass
*     -------------      
      call genint (p, nxpo, nypo, 0.5d0, 0.5d0, pm)

*     Filtering inner points     
*     ----------------------
!$OMP DO SCHEDULE (STATIC)      
      do i=2,nxpoc-1 !! coarse-grid index
        ii = 1 + (i-1)*nso !! fine-grid index
        do j=2,nypoc-1
          jj = 1 + (j-1)*nso
          psum = 0.0d0 !! convoluted state
          wsum = 0.0d0 !! accounted weight
          do il=1,wd+1 !! local index (within kernel)
            ig = ii + il - nh !! global index (fine-grid)
            if ( (ig.lt.1) .or. (ig.gt.nxpo) ) then
              cycle !! skip iter.
            endif
            do jl=1,wd+1
              jg = jj + jl - nh
              if ( (jg.lt.1) .or. (jg.gt.nypo) ) then
                cycle 
              endif  
*             Convolution of p by fk
              psum = psum + fk(il,jl)*p(ig,jg)
              wsum = wsum + fk(il,jl)            
            enddo
          enddo
*         Normalization
          if (rescale) then          
            pf(i,j) = psum/wsum
          else
            pf(i,j) = psum
          endif
        enddo
      enddo
!$OMP END DO      

*     Filling boundary values
*     -----------------------     
      call genint (pf, nxpoc, nypoc, 0.5d0, 0.5d0, pfm)

!$OMP SINGLE
      pbc = bcfc*(rdom*pm - pfm)
!$OMP END SINGLE

!$OMP DO SCHEDULE (STATIC)      
      do j=1,nypoc
        pf(1,j) = pbc
        pf(nxpoc,j) = pbc
      enddo
!$OMP END DO NOWAIT     

!$OMP DO SCHEDULE (STATIC)      
      do i=2,nxtoc
        pf(i,1) = pbc
        pf(i,nypoc) = pbc
      enddo
!$OMP END DO

*     Derive residuals
*     ----------------      
!$OMP DO SCHEDULE (STATIC)      
      do j=1,nypoc
        jj = 1 + (j-1)*nso
        do i=1,nxpoc
          ii = 1 + (i-1)*nso
          pr(i,j) = p(ii,jj) - pf(i,j)
        enddo  
      enddo
!$OMP END DO  
      
!!      call genint (pf, nxpoc, nypoc, 0.5d0, 0.5d0, pfm)
!$OMP END PARALLEL

!!      print *, 'Original mass = ',pm/dble(nxto*nyto)
!!      print *, 'Filtered mass = ',pfm/dble(nxtoc*nytoc)

      END SUBROUTINE pfilt
c
c**************************************************************************
c      
      SUBROUTINE pfilt_cycl (pf, pr, p, fk, wd)

*     Coarse-graining of eddy-resolving snapshot in the cyclic channel.

*     Modules          
      USE param, ONLY : nxpo,nypo,nxpoc,nypoc,nso,rescale

      IMPLICIT NONE

*     I/O arguments
      integer, INTENT(IN) :: wd
      double precision, INTENT(IN) :: fk(wd+1,wd+1),p(nxpo,nypo)
      double precision, INTENT(OUT), DIMENSION(nxpoc,nypoc) :: pf,pr
*
*     fk is filter kernel of total weight wt and width wd
*     p is eddy-resolving presure (m^2/s^2)
*     pf and pr are corase-grained and residual pressure (m^2/s^2)

*     Local variables
      integer nh,i,j,ii,jj,il,jl,ig,jg
      double precision psum,wsum

      nh = wd/2 + 1 !! size of half-kernel

!$OMP PARALLEL DEFAULT (NONE)
!$OMP&         PRIVATE (i,j,ii,jj,il,jl,ig,jg,psum,wsum)
!$OMP&         SHARED  (p,pf,pr,fk,wd,nh)

*     Filtering inner points     
*     ----------------------
!$OMP DO SCHEDULE (STATIC)      
      do i=1,nxpoc-1 !! coarse-grid index
        ii = 1 + (i-1)*nso !! fine-grid index
        do j=2,nypoc-1
          jj = 1 + (j-1)*nso
          psum = 0.0d0 !! convoluted state
          wsum = 0.0d0 !! accounted weight
          do jl=1,wd+1 !! local index (within kernel)
            jg = jj + jl - nh !! global index (fine-grid)
            if ( (jg.lt.1) .or. (jg.gt.nypo) ) then
              cycle !! skip iter.
            endif
            do il=1,wd+1
              ig = ii + il - nh
              if (jg.lt.1) then
                  psum = psum + fk(il,jl)*p(ig+nxpo-1,jg)
              else if (jg.gt.nypo) then
                  psum = psum + fk(il,jl)*p(ig-nxpo+1,jg)
              else
                  psum = psum + fk(il,jl)*p(ig,jg)
              endif  
              wsum = wsum + fk(il,jl)            
            enddo
          enddo
*         Rescale pf by wt
          if (rescale) then
            pf(i,j) = psum/wsum
          else
            pf(i,j) = psum
          endif  
*         Substract pr
          pr(i,j) = p(ii,jj) - pf(i,j)
        enddo
      enddo
!$OMP END DO NOWAIT      

*     Filling boundary values
*     -----------------------     
*     Update N/S value
!$OMP DO SCHEDULE (STATIC)      
      do i=1,nxpoc
        ii = 1 + (i-1)*nso
        pf(i,1) = 0.0d0!!p(ii,1)*wt 
        pr(i,1) = p(ii,1) - pf(i,1)
        pf(i,nypoc) = 0.0d0!!p(ii,nypo)*wt
        pr(i,nypoc) = p(ii,nypo) - pf(i,nypoc)
      enddo
!$OMP END DO NOWAIT     
*     Update W/E value      
!$OMP DO SCHEDULE (STATIC)      
      do j=2,nypoc-1
        pf(nxpoc,j) = pf(1,j)
        pr(nxpoc,j) = pr(1,j)
      enddo
!$OMP END DO      
!$OMP END PARALLEL

      END SUBROUTINE pfilt_cycl
c
c**************************************************************************      
c
      SUBROUTINE p2uv (u, v, p, rdxf0)

*     Compute geostrophic velocities from pressure

*     Modules
      USE param

      IMPLICIT NONE    
      
*     I/O arguments
      double precision, INTENT(OUT) :: u(nxpoc,nytoc),v(nxtoc,nypoc)
      double precision, INTENT(IN) :: p(nxpoc,nypoc),rdxf0

*     Local variables      
      integer i,j

!$OMP PARALLEL DEFAULT (NONE)
!$OMP&         PRIVATE (i,j)
!$OMP&         SHARED  (u,v,p,rdxf0)
      
*     Zonal velocity  
!$OMP DO SCHEDULE (STATIC)      
      do i=1,nxpoc
        do j=1,nytoc
          u(i,j) = -rdxf0*( p(i,j+1) - p(i,j) )
        enddo
      enddo
!$OMP END DO NOWAIT      
   
*     Meridional velocity      
!$OMP DO SCHEDULE (STATIC)      
      do j=1,nypoc
        do i=1,nxtoc
          v(i,j) =  rdxf0*( p(i+1,j) - p(i,j) )
        enddo
      enddo
!$OMP END DO 
!$OMP END PARALLEL

      END SUBROUTINE p2uv
*
c***********************************************************************
*
      SUBROUTINE genint (val, nx, ny, facwe, facsn, answer)
*
*     Computes area integral of internal values of field
*     val(nx,ny). facwe controls contribution from western
*     & eastern boundaries; facsn controls contribution from
*     southern & northern boundaries. Integral returned as answer
*     Modified version with reduced accumulator error
*
*     N.B. Parallelised using "orphaned directives";
*     only intended for use in the parallel regions
*     established earlier in calling subroutine monnc_comp

      IMPLICIT NONE

*     Subroutine arguments
      integer nx,ny
      double precision val(nx,ny),facwe,facsn,answer
*
*     Local variables
      integer i,j
      double precision sumi,xxs,xxn
*
!$OMP SINGLE
      answer = 0.0d0
!$OMP END SINGLE
*     Inner points + facwe*(W & E boundaries)
!$OMP DO SCHEDULE (STATIC)
!$OMP&   PRIVATE (i,j,sumi)
!$OMP&   REDUCTION (+:answer)
      do j=2,ny-1
        sumi = facwe*val(1,j)
        do i=2,nx-1
          sumi = sumi + val(i,j)
        enddo
        sumi = sumi + facwe*val(nx,j)
        answer = answer + sumi
      enddo
!$OMP END DO

!$OMP SINGLE
*     N & S boundary contributions
      xxs = facwe*val(1, 1)
      xxn = facwe*val(1,ny)
*     Inner points
      do i=2,nx-1
        xxs = xxs + val(i, 1)
        xxn = xxn + val(i,ny)
      enddo
      xxs = xxs + facwe*val(nx, 1)
      xxn = xxn + facwe*val(nx,ny)

      answer = answer + facsn*( xxs + xxn )
!$OMP END SINGLE

      END SUBROUTINE genint
c      
c**************************************************************************      
c
      END MODULE coarse_subs    
c      
c**************************************************************************
