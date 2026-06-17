c**************************************************************************
c
c     This module is used to allocate input parameters for performing
c     the program coarse_grain.
c
c     Written by Long Li, 10 Aug. 2020.  
c      
c**************************************************************************      
c
      MODULE param

      IMPLICIT NONE

      PUBLIC
      SAVE

*     I/O directory
*     -------------
      character (len=*) :: iodir,innam,outnam
      PARAMETER ( innam = 'ocpo.nc', outnam = 'ocfluc40_f200.nc',
     &            iodir = '/media/loli/Long/q-gcm/' //
     &                    'double_gyre_ocean_only/' //
     &                    'REF5/data1d/yrs114-115/' )  
      logical :: cyclic,rescale
      PARAMETER ( cyclic = .FALSE., rescale = .FALSE. ) 

*     Array dimensions    
*     ----------------
      integer :: nxto,nyto,nlo,nto,nso,fwid
      PARAMETER ( nxto = 768, nyto = 960, nlo = 3, nto = 366, 
     &            nso = 8, fwid = 40 )      
*
*     nxto, nyto are the numbers of oceanic gridcells W/E and N/S.
*     nlo is the number of oceanic layers.
*     nto is the number of snapshots
*     nso is the number of oceanic gridpoints for subsampling. 
*     nso needs to be a factor of both nxto and nyto.
*     fwid is the low-pass filter width (240km as default)

*     Ohter physical constant
*     -----------------------
      double precision :: fnot
      PARAMETER ( fnot = 9.37456D-05 ) !! 40 North
*
*     fnot is the Coriolis parameter (s^-1)

*     Derived grid parameters (do not alter)
*     -------------------------------------- 
      integer :: nxpo,nypo,nxtoc,nytoc,nxpoc,nypoc
      PARAMETER ( nxpo = nxto + 1, nypo = nyto + 1,
     &            nxtoc = nxto/nso, nytoc = nyto/nso,
     &            nxpoc = nxtoc + 1, nypoc = nytoc + 1 )
*
*     nxpo, nypo are the numbers of oceanic p points W/E and N/S.      
*     nxtoc, nytoc are the numbers of p points in coarse grid.
*     nxpoc, nypoc are the corresponding number of T grid points.      
*     nfto is the width of the filter kernel (oceanic gridcells).
*     nfpo is the number of p grid points in each axis of kernel.
*     nhpo is the number of p grid points in each half of axis.      

      END MODULE param    
c      
c**************************************************************************
 
