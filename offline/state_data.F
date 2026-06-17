c**************************************************************************
c
c     This module is used to allocate state variables for performing
c     the program coarse_grain.
c
c     Written by Long Li, 10 Aug. 2020.  
c      
c**************************************************************************      
c
      MODULE state

      USE param
      
      IMPLICIT NONE

      PUBLIC
      SAVE

      double precision xpo(nxpo),ypo(nypo),zo(nlo),tyrs(nto),
     &                 xpoc(nxpoc),ypoc(nypoc),xtoc(nxtoc),ytoc(nytoc)
      double precision, DIMENSION(nxpoc,nypoc,nlo) :: pf,pr
      double precision, DIMENSION(nxpoc,nytoc,nlo) :: uf,ur
      double precision, DIMENSION(nxtoc,nypoc,nlo) :: vf,vr
*
*     xpo and ypo are fine P-grid axis (km)
*     zo is vertical axis (km)
*     tyrs is time axis (years) 
*     xpoc and ypoc are coarse P-grid axis (km)
*     xtoc and ytoc are coarse T-grid axis (km)
*     pf and pr are the corase-grained and residual pressure (m^2/s^2)
*     uf and ur are the corase-grained and residual zonal velocity (m/s)
*     vf and vr are the corase-grained and residual meridional velocity (m/s)

      END MODULE state   
c      
c**************************************************************************
