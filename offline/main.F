c**************************************************************************
c
c     This program is used to perform the coarse-graining procedure of
c     eddy-resolving snapshots in order to construct the POD hereafter.
c
c     Written by Long Li, 10 Aug. 2020.  
c      
c**************************************************************************
c      
      PROGRAM main

*     Modules
      USE param    
      USE state
      USE ionc_subs
      USE coarse_subs

      IMPLICIT NONE
      INCLUDE 'netcdf.inc'

*     Local variables
      integer i,j
      double precision dxoc,rdxof0,gfk(2*fwid+1,2*fwid+1),
     &                 po(nxpo,nypo,nlo) 
*      
*     po is fine-grid pressure (m^2/s^2)      

*     NetCDF variables
      character (len=*), parameter :: subnam = 'main'
      integer status

*     OpenMP variables
!$    integer nprocs, OMP_GET_NUM_PROCS, nthmax, OMP_GET_MAX_THREADS
      
      print *
      print *, '*******************************************************'
      print *, '****  Coarse-graining of eddy-resolving snapshots   ***'
      print *, '*******************************************************'

*     Examine OpenMP environment
*     --------------------------      
!$    nprocs = OMP_GET_NUM_PROCS()
!$    nthmax = OMP_GET_MAX_THREADS()
!$    write(*,*) ' ' 
!$    write(*,*) ' OpenMP parallelism is activated'
!$    write(*,'(a,i5)') ' No. of processors available = ', nprocs
!$    write(*,'(a,i3)') ' Max. no. of threads available = ', nthmax

*     Initialize inputs
*     -----------------
*     Open netCDF file 
      status = nf_open (trim(iodir)//innam, nf_nowrite, incid)
      if ( status.ne.NF_NOERR ) call handle_err (status, subnam)
      call read_ini
      print *
      print *, 'Input file initialized'

*     Construct coarse grid
*     ---------------------
*     x axis of p grid      
      do i=1,nxpoc 
        xpoc(i) = xpo(1+(i-1)*nso)
      enddo
*     y axis of p grid        
      do j=1,nypoc
        ypoc(j) = ypo(1+(j-1)*nso)
      enddo
*     x axis of T grid      
      do i=1,nxtoc
        xtoc(i) = 0.5d0*( xpoc(i) + xpoc(i+1) )
      enddo
*     y axis of T grid        
      do j=1,nytoc
        ytoc(j) = 0.5d0*( ypoc(j) + ypoc(j+1) )
      enddo

*     Initialize outputs
*     ------------------
*     Create netCDF file
      status = nf_create (trim(iodir)//outnam, 
     &                    ior(nf_clobber,nf_64bit_offset), oncid)
      if ( status.ne.NF_NOERR ) call handle_err (status, subnam)
      call save_ini
      print *
      print *, 'Output file created'

*     Derived parameters      
      dxoc = 1.0d3*( xpoc(2) - xpoc(1) ) !! (m)
      rdxof0 = 1.0d0/( fnot*dxoc ) !! (m^-1 s)

*     Coarse-graining of pressure snapshots
*     -------------------------------------
      print *
      print *, 'Generate Gaussian filter kernel'
      call gaussf (gfk, 2*fwid)

      print *
      print *, 'Coarse-graining procedure'
      print *, '=========================' 
      do i=1,nto
        write(*,'(a,f9.3)') '  Time (years) = ', tyrs(i)
*       Read eddy-resolving pressure
        call read_out (po, i)
        do j=1,nlo
*         Coarse-graining of po
          if ( cyclic ) then
              call pfilt_cycl (pf(1,1,j), pr(1,1,j), po(1,1,j), 
     &                         gfk, 2*fwid)
          else
              call pfilt (pf(1,1,j), pr(1,1,j), po(1,1,j), 
     &                    gfk, 2*fwid) 
          endif    
*         Compute geostrophic velocity 
          call p2uv (uf(1,1,j), vf(1,1,j), pf(1,1,j), rdxof0)
          call p2uv (ur(1,1,j), vr(1,1,j), pr(1,1,j), rdxof0)
        enddo
*       Save output file
        call save_out (i)
      enddo

*     Close the file and freeing all resources
      status = nf_close (incid)
      if ( status.ne.NF_NOERR ) call handle_err (status, subnam)
      status = nf_close (oncid)
      if ( status.ne.NF_NOERR ) call handle_err (status, subnam)

      print *
      print *, '*******************************************************'
      print *, '******************  END of program   ******************'
      print *, '*******************************************************'

      END PROGRAM main    
c
c**************************************************************************
